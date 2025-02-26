import os
import logging
import argparse

import yaml

from dataloader.again_reader import AgainReader, CustomReader
from dataloader.dataset import TestDataset
from utils.preprocessing import cleaning_logs
from utils.video_frame_extractor import parse_AGAIN_images, cut_video
from executor.tester import RanknetTester


def inference(config, logger):
    # v cleaning
    cleaned_log = cleaning_logs(config, logger)
    # v frame extraction
    parse_AGAIN_images(cleaned_log, config, logger)
    # v inference
    data, numeric_columns, bio_size = CustomReader(cleaned_log, config, logger).prepare_dataset()
    test_samples = TestDataset(data, numeric_columns, config)
    tester = RanknetTester(test_samples, bio_size, config=config, logger=logger)
    tester.test()
    # find inflection points
    # clip video -> video cutter 참조
    pass


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