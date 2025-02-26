import os
import logging
import argparse

import numpy as np
import yaml

from dataloader.again_reader import AgainReader, CustomReader
from dataloader.dataset import TestDataset
from utils.preprocessing import cleaning_logs
from utils.video_frame_extractor import parse_AGAIN_images, cut_video
from utils.stats import find_significant_peaks_and_valleys
from executor.tester import RanknetTester


def inference(config, logger):
    player = config['experiment']['player']
    session = config['experiment']['session']
    game_name = config['game_name'][config['experiment']['game']]
    video_path = os.path.join(
        config['data']['path'],
        config['data']['vision']['video'],
        f'{player}_{game_name}_{session}.mp4'
    )
    clip_path = os.path.join(config['data']['path'], 'video_clips')

    # v cleaning
    cleaned_log = cleaning_logs(config, logger)

    # v frame extraction
    parse_AGAIN_images(cleaned_log, config, logger)

    # v inference
    data, numeric_columns, bio_size = CustomReader(cleaned_log, config, logger).prepare_dataset()
    test_samples = TestDataset(data, numeric_columns, config)
    tester = RanknetTester(test_samples, bio_size, config=config, logger=logger)
    output = tester.inference()
    print(output)

    # find inflection points
    peaks, valleys = find_significant_peaks_and_valleys(output, threshold=0.5)

    inflections = np.concatenate([peaks, valleys])
    inflections = np.sort(inflections)
    inflections = np.unique(inflections)
    roi_list = [[0, 12]]
    for i in range(len(inflections)):
        if roi_list[-1][1] >= inflections[i] - 6:
            roi_list[-1][1] = inflections[i] + 6
        else:
            roi_list.append([inflections[i]-6, inflections[i]+6])
    print(roi_list)

    # cut video
    for i, roi in enumerate(roi_list):
        video_clip = os.path.join(clip_path, f'{player}_{game_name}_{session}_{i}.mp4')
        cut_video(video_path, roi[0], roi[1], video_clip)
        print(f'Video clip {i} is saved at {video_clip}')


def interpolate(config):
    # interpolation
    # reconstruction graph
    pass


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

    fh = logging.FileHandler(os.path.join(conf['train']['log_dir'], 'exp_log.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # again = AgainReader(config=conf).game_info_by_name('TopDown')
    # print(again.columns)

    inference(conf, logger)
    # interpolate(config)