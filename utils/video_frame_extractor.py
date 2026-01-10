import os
import argparse
import yaml
from tqdm import tqdm
import h5py
import cv2
import subprocess

from dataloader.again_reader import AgainReader


def cut_video(input_video, start, end, output_video):
    command = [
        'ffmpeg',
        '-i', input_video,  # 입력 영상 파일
        '-ss', str(start),  # 시작 시간 (초 단위)
        '-to', str(end),  # 끝 시간 (초 단위)
        '-c:v', 'libx264',  # 비디오 코덱 설정
        '-c:a', 'aac',  # 오디오 코덱 설정
        '-strict', 'experimental',  # FFmpeg 설정 (필요한 경우)
        '-y',  # 덮어쓰기 옵션
        output_video
    ]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def parse_images_from_video_by_timestamp(data, video_full_path, output_path, video_name, transform=False, compression='gzip', compression_level=4):
    cap = cv2.VideoCapture(video_full_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    with h5py.File(output_path, 'a') as f:
        if video_name in f:
            del f[video_name]

        video_group = f.create_group(video_name)

        for _, row in data.iterrows():
            time_stamp = row['time_stamp']
            frame_offset = int(fps * time_stamp)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
            ret, frame = cap.read()

            if ret:
                time_index = row['time_index']
                if transform:
                    # resize image to 1/2 size
                    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_group.create_dataset(f'{time_index}_{time_stamp}', data=frame, dtype='uint8',
                                           compression=compression, compression_opts=compression_level)

    cap.release()


def parse_AGAIN_images(again, config):
    video_path = os.path.join(config['data']['path'], config['data']['vision']['video'])
    out_dir = os.path.join(config['data']['path'], config['data']['vision']['frame'])
    print(f'Extract images from video: {video_path} -> {out_dir}')

    sessions = again['session_id'].unique()

    for s_i, session in tqdm(enumerate(sessions), total=len(sessions)):
        data = again[(again['session_id'] == session)]

        game = data['game'].unique()[0]
        game_name = config['game_name'][game]
        player = data['player_id'].unique()[0]

        video = f'{player}_{game_name}_{session}.mp4'
        video_name = os.path.splitext(video)[0]
        video_full_path = os.path.join(video_path, video)

        parse_images_from_video_by_timestamp(data, video_full_path, out_dir, video_name, transform=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PrefAB prototype')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    again = AgainReader(config=config).again

    parse_AGAIN_images(again=again, config=config)
