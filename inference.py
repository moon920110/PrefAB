import os
import logging
import argparse

import numpy as np
import yaml

from dataloader.again_reader import CustomReader
from dataloader.dataset import TestDataset
from utils.preprocessing import cleaning_logs
from utils.video_frame_extractor import parse_AGAIN_images, cut_video
from utils.stats import find_significant_peaks_and_valleys
from utils.utils import convert_frame_to_time
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
    clip_path = os.path.join(config['data']['path'], 'video_clips', f'{player}_{session}')
    if not os.path.exists(clip_path):
        os.makedirs(clip_path)

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
    inflections = np.unique(inflections)
    inflections = np.sort(inflections)
    inflections += 12  # prediction starts from 3 second. 12 frames = 3 seconds
    print(inflections)

    roi_list = [[1, 4]]  # seconds
    for i in range(len(inflections)):
        inflection_time = convert_frame_to_time(inflections[i])
        if roi_list[-1][1] >= inflection_time - 1.5:
            roi_list[-1][1] = inflection_time + 1.5
        else:
            roi_list.append([inflection_time - 1.5, inflection_time + 1.5])
    print(roi_list)

    # cut video
    for i, roi in enumerate(roi_list):
        video_clip = os.path.join(clip_path, f'{player}_{game_name}_{session}_{i}.mp4')
        cut_video(video_path, roi[0], roi[1], video_clip)
        print(f'Video clip {i} is saved at {video_clip}')


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

    fh = logging.FileHandler(os.path.join(conf['train']['log_dir'], 'exp_inference.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    inference(conf, logger)