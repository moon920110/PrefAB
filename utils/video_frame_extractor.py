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


def parse_images_from_video_by_timestamp(data, video_full_path, video_out_path, transform=False, compression='gzip', compression_level=9):
    cap = cv2.VideoCapture(video_full_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    with h5py.File(video_out_path, 'w') as f:
        frame_group = f.create_group('frames')

        for _, row in tqdm(data.iterrows(), desc=f'Processing frames'):
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
                frame_group.create_dataset(f'{time_index}_{time_stamp}', data=frame, dtype='uint8',
                                           compression=compression, compression_opts=compression_level)

    cap.release()


def parse_custom_images(video_path, out_dir, again, config):
    pass


def parse_AGAIN_images(again, config, logger):
    video_path = os.path.join(config['data']['path'], config['data']['vision']['video'])
    out_dir = os.path.join(config['data']['path'], config['data']['vision']['frame'])

    logger.info(f'Extract images from video: {video_path} -> {out_dir}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    games = again['game'].unique()
    players = again['player_id'].unique()

    for g_i, game in enumerate(games):
        for p_i, player in enumerate(players):
            data = again[(again['game'] == game) & (again['player_id'] == player)]
            if len(data) == 0:
                print(f'There is no data for {game} ({g_i + 1}/{len(games)}) - {player} ({p_i + 1}/{len(players)})')
                continue
            session_id = data['session_id'].unique()[0]
            game_name = config['game_name'][game]

            video = f'{player}_{game_name}_{session_id}.mp4'
            video_name = os.path.splitext(video)[0]
            video_full_path = os.path.join(video_path, video)
            video_out_path = os.path.join(out_dir, f'{video_name}.h5')

            if os.path.exists(video_out_path):
                continue

            parse_images_from_video_by_timestamp(data, video_full_path, video_out_path, transform=True)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PrefAB prototype')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    again = AgainReader(config=config)
    again_shooter = again.game_info_by_genre('Shooter')

    parse_AGAIN_images(again=again, config=config)
