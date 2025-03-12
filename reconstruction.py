import os
import logging
import argparse

import pandas as pd
import numpy as np
import yaml

from scipy.interpolate import PchipInterpolator
from dataloader.again_reader import AgainReader, CustomReader
from dataloader.dataset import TestDataset
from utils.preprocessing import resample_arousal, get_intensity
from utils.video_frame_extractor import parse_AGAIN_images, cut_video
from utils.stats import find_significant_peaks_and_valleys
from utils.utils import convert_time_to_frame
from executor.tester import RanknetTester


def interpolate(config, logger):
    player = config['experiment']['player']
    session = config['experiment']['session']
    game_name = config['game_name'][config['experiment']['game']]
    clip_path = os.path.join(config['data']['path'], 'video_clips', f'{player}_{session}')
    roi_list = pd.read_csv(os.path.join(clip_path, 'roi.csv'))
    clean_path = os.path.join(config['data']['path'], 'clean_data', f'{player}_{game_name}_{session}_clean.csv')
    clean_data = pd.read_csv(clean_path)
    annotation_data_all = pd.read_csv(os.path.join(clip_path, f'{player}_{session}_arousal.csv'))
    anno_sessions = annotation_data_all['SessionID'].unique()

    clean_data['arousal'] = 0.
    clean_data['time_index'] = pd.to_timedelta(clean_data['time_index'])
    logger.info(f'player: {player}, session: {session}, game: {game_name}')

    # video name에서 시작 시간과 끝 시간을 추출
    for i, roi in roi_list.iterrows():
        session_id = anno_sessions[i]
        annotation_data = annotation_data_all[annotation_data_all['SessionID'] == session_id]
        annotation_data = resample_arousal(annotation_data)
        annotation_data[-1] = annotation_data[-2]

        idx = i + 1
        start_time = roi['start']
        end_time = roi['end']
        logger.info(f'Processing {idx}th video clip: {start_time} ~ {end_time}')

        start_timedelta = pd.to_timedelta(start_time, unit='s')
        annotation_data.index = annotation_data.index + start_timedelta
        start_arousal_offset = clean_data.loc[clean_data['time_index'] == start_timedelta, 'arousal'].values[0]

        logger.info(f'start_arousal_offset: {start_arousal_offset}, annotation_data: {annotation_data}')
        for time_index, arousal in annotation_data.items():
            # print(f'time_index: {time_index}, start_arousal: {start_arousal_offset}, arousal: {arousal}, sum: {start_arousal_offset + arousal}')
            clean_data.loc[clean_data['time_index'] == time_index, 'arousal'] = arousal + start_arousal_offset

        end_timedelta = pd.to_timedelta(end_time, unit='s')

        if end_timedelta >= clean_data['time_index'].iloc[-1]:
            end_timedelta = clean_data['time_index'].iloc[-1]

        end_index = clean_data.loc[clean_data['time_index'] == end_timedelta].index[0]
        last_4_data = annotation_data.values
        gradient = np.diff(last_4_data).sum()
        # gradient = last_4_data.mean()
        if end_index < len(clean_data):
            for i in range(end_index, len(clean_data)):
                # print(f'prev_arousal: {clean_data.loc[i - 1, "arousal"]}, gradient: {gradient}, sum: {clean_data.loc[i - 1, "arousal"] + gradient}')
                # clean_data.loc[i, 'arousal'] = clean_data.loc[i - 1, 'arousal'] + gradient
                clean_data.loc[i, 'arousal'] = start_arousal_offset + gradient
                # print(start_arousal_offset, clean_data.loc[i, 'arousal'], gradient)

    clean_data['arousal'] = get_intensity(clean_data, ['arousal'])
    clean_data.to_csv(clean_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PrefAB prototype')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(conf['train']['log_dir'], 'exp_interpolation.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    interpolate(conf, logger)