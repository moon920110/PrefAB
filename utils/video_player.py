"""
    NEED TO BE RUN BY LOCAL INTERPRETER
"""
import os
import sys
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QSlider
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage

from dataloader.again_reader import AgainReader


class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()

        with open('../config/config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.config['data']['path'] = '/Users/supermoon/Documents/Research/Affective game/AGAIN'
        self.again = AgainReader(self.config).game_info_by_name('Shootout')
        self.again['time_index'] = self.again['time_index'].apply(
            lambda x: sum([a * b for a, b in zip([3600, 60, 1], map(float, x[7:].split(':')))]))
        self.players = self.again['player_id'].unique()
        self.p_idx = 0
        self.file = None

        # Layout and GUI elements
        self.layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.play_button = QPushButton('Play')
        self.next_button = QPushButton('Next')
        self.slider = QSlider(Qt.Horizontal)
        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.play_button)
        self.layout.addWidget(self.next_button)
        self.layout.addWidget(self.slider)

        # Arousal graph setup
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Set layout
        self.setLayout(self.layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.slider.valueChanged.connect(self.set_frame)

        self.play_button.clicked.connect(self.play_video)
        self.next_button.clicked.connect(self.set_new_player_data)

        self.cap = None
        self.frame_count = 0
        self.frame_rate = 0
        self.sampling_rate = 0

        self.set_new_player_data()

    def set_new_player_data(self):
        player = self.players[self.p_idx]
        self.p_idx += 1
        self.arousal_data = self.again[self.again['player_id'] == player]
        self.arousal_data = self.arousal_data.sort_values('time_index')
        self.file = f'{player}_{self.config["game_name"][self.arousal_data["game"].unique()[0]]}_{self.arousal_data["session_id"].unique()[0]}'

        self.set_new_video()

    def set_new_video(self):
        # Video capture
        self.cap = cv2.VideoCapture(os.path.join(self.config['data']['path'], 'videos', self.file + '.webm'))
        self.frame_count = self.count_frames()
        self.frame_rate = int(self.frame_count / self.arousal_data['time_index'].max())
        print(f'Frame rate: {self.frame_rate}')
        self.sampling_rate = int(self.frame_count / len(self.arousal_data))
        print(f'Sampling rate: {self.sampling_rate}')
        self.cap = cv2.VideoCapture(os.path.join(self.config['data']['path'], 'videos', self.file + '.webm'))
        # self.frame_count = 2000
        self.slider.setMaximum(self.frame_count)


    def count_frames(self):
        count = 0
        while True:
            ret, _ = self.cap.read()
            if not ret:
                break
            count += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f'Frame count: {count}')
        return count

    def play_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play')
        else:
            self.timer.start(self.frame_rate)  # Adjust the timer to match the video frame rate
            self.play_button.setText('Pause')

    def set_frame(self, value):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
        self.next_frame()

    def next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Update slider position
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.blockSignals(True)
            self.slider.setValue(current_frame)
            self.slider.blockSignals(False)

            # Update video frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))

            # Update arousal graph
            self.update_graph(current_frame)

    def update_graph(self, current_frame):
        # Map current frame to arousal data index
        if current_frame % self.sampling_rate == 0:
            data_index = int((current_frame / self.sampling_rate))
            relevant_data = self.arousal_data['arousal'][:data_index]
            time_index = self.arousal_data['time_index'][:data_index]
            # time_index = np.arange(1, data_index+1)

            try:
                self.ax.clear()
                self.ax.plot(time_index, relevant_data, label='Arousal')
                self.ax.axvline(x=data_index, color='r', linestyle='--')
                self.ax.set_xlim(0, self.arousal_data['time_index'].max())
                self.ax.set_ylim(0, 1)
                self.canvas.draw()
            except Exception as e:
                print(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
