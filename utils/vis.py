import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoFileClip
import cv2


def plot_ordinal_arousal(data, title):
    games = data['[control]game'].unique()

    fig_num = 1
    for game in games:
        game_data = data[data['game'] == game].groupby('player_id')

        plt.figure(fig_num)
        plt.title(f'{title} - {game}')
        i = 0
        for player_id, player_data in game_data:
            # compare arousal at a time point with the previous time point. If it is higher, it is 1, if it is lower, it is 0, and if it is the same, it is 0.5.
            ordinal_arousal = player_data['arousal'].diff().apply(lambda x: 1 if x > 0 else 0 if x < 0 else 0.5)
            player_data['ordinal_arousal'] = ordinal_arousal

            plt.plot(player_data['time_index'], player_data['ordinal_arousal'])
            i += 1
            if i > 1:
                break
        fig_num += 1
        # plt.savefig(f'data/{title}-{game}.png')
    plt.show()


def plot_arousal(data, title):
    games = data['game'].unique()

    fig_num = 1
    for game in games:
        game_data = data[data['game'] == game].groupby('player_id')
        # fig, ax = plt.subplots()
        # ax.title(f'{title}_{game}')
        plt.figure(fig_num)
        plt.title(f'{title}_{game}')
        i = 0
        for player_id, player_data in game_data:
            plt.plot(player_data['time_index'], player_data['arousal'])
            i += 1
            if i > 10:
                break
        fig_num += 1
    plt.show()


def plot_arousal_with_video(game_data, video_path, config):
    player_ids = game_data['player_id'].unique()

    player_data = game_data[game_data['player_id'] == player_ids[0]]
    parc_id = player_data['player_id'].unique()[0]
    game_name = config['game_name'][player_data['game'].unique()[0]]
    session_id = player_data['session_id'].unique()[0]
    video_name = f'{parc_id}_{game_name}_{session_id}.mp4'

    print(f'read: {os.path.join(video_path, video_name)}')

    video = VideoFileClip(os.path.join(video_path, video_name))
    duration = video.duration
    # video.preview()
    # cap = cv2.VideoCapture(os.path.join(video_path, video_name))
    # if not cap.isOpened():
    #     print('Error: Unable to open video')

    fig, ax = plt.subplots()

    # def get_frame():
    #     ret, frame = cap.read()
    #     if not ret:
    #         return None
    #     return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # frame = get_frame()
    im = ax.imshow(video.get_frame(0))

    def update(t):
        frame = video.get_frame(t)
        im.set_array(frame)
        return im

    ani = FuncAnimation(fig, update, frames=np.linspace(0, duration, int(duration*10)))
    # plt.plot(player_data['[control]time_index'], player_data['[output]arousal'])
    plt.show()
    # cap.release()
    # cv2.destroyAllWindows()
