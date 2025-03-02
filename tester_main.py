import os
import argparse
import logging

import yaml
import torch
import pandas as pd

from utils.stats import find_significant_peaks_and_valleys, inflection_comparison, get_dtw_cluster, reconstruct_state_via_interpolation
from utils.video_frame_extractor import parse_AGAIN_images
from utils.utils import create_new_filename, h5reader
from dataloader.dataset import PairDataset, TestDataset
from dataloader.again_reader import AgainReader
from executor.tester import RanknetTester
from utils.preprocessing import integrate_arousal


def dtw_cluster_demo(save=False):
    with open('./config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    game = 'TopDown'
    again_reader = AgainReader(config)
    custom_again_reader = AgainReader(config, again_file_name='clean_data_custom.csv')

    data = again_reader.game_info_by_name(game)
    custom_data = custom_again_reader.game_info_by_name(game)
    data = pd.concat([data, custom_data], ignore_index=True)

    clusters = get_dtw_cluster(data, config)
    if save:
        clusters = pd.DataFrame(clusters.items(), columns=['session_id', 'cluster'])
        clusters.to_csv(os.path.join(config['data']['path'], 'cluster', f'cluster_{game}.csv'), index=False)
    return clusters


def find_peak_demo():
    with open('config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config['data']['path'] = 'data/'
    game = 'Shootout'
    print(f'read data {game}')
    again_reader = AgainReader(config)
    # data = again_reader.game_info_by_name(game)
    data = again_reader.game_info_by_name('Shootout')
    print(f'len data: {len(data)}')

    sessions = data['session_id'].unique()
    for i, session in enumerate(sessions):
        print(f'session: {session}')
        session_data = data[data['session_id'] == session]
        find_significant_peaks_and_valleys(session_data['arousal'].values, threshold=0.5)

        if i > 20:
            break


def post_analysis_demo(case='None'):
    dirs = [['/home/jovyan/projects/PrefAB/log/prefab_bio_film_cluster_aux_TinyCars', False]]
    for item in dirs:
        # inflection_comparison(dir)
        if case == 'Comparison':
            inflection_comparison(item[0], True, item[1])
        elif case == 'Reconstruction':
            reconstruct_state_via_interpolation(item[0], True, item[1])



def video_fram_extractor_main():
    parser = argparse.ArgumentParser(description='PrefAB prototype')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config, encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    again = AgainReader(config=config)

    parse_AGAIN_images(video_path=os.path.join(again.data_path, config['data']['vision']['video']),
                                         out_dir=os.path.join(again.data_path, config['data']['vision']['frame']),
                                         again=again.again,
                                         config=config,
                                         )


def tsne_demo():
    parser = argparse.ArgumentParser(description='PrefAB prototype')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists(config['test']['log_dir']):
        os.makedirs(config['test']['log_dir'])
    config['test']['new_exp'] = create_new_filename(config['test']['log_dir'], config['test']['exp'])

    if not os.path.exists(os.path.join(config['test']['log_dir'], f"{config['test']['new_exp']}")):
        os.makedirs(os.path.join(config['test']['log_dir'], f"{config['test']['new_exp']}"))

    dataset, numeric_columns, bio_features_size = AgainReader(config).prepare_sequential_ranknet_dataset()
    train_size = int(len(dataset) * config['train']['train_ratio'])
    test_size = len(dataset) - train_size
    train_samples, test_samples = torch.utils.data.random_split(dataset,
                                                                [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(
                                                                    config['test']['seed'])
                                                                )

    # train_dataset = PairDataset(train_samples, numeric_columns, bio_features_size, config)
    test_dataset = TestDataset(test_samples, numeric_columns, config)
    # test_dataset = TestDataset(train_samples, numeric_columns, config)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(config['test']['log_dir'], f"{config['test']['new_exp']}", 'log.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    tester = RanknetTester(test_dataset, bio_features_size, config=config, logger=logger)
    tester.test()
    logger.info("Testing is done!")


def integrate_arousal_test():
    with open('config/config.yaml', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    integrate_arousal(config)


if __name__ == '__main__':
    # post_analysis_demo('Comparison')
    # tsne_demo()
    print(dtw_cluster_demo(True))
    # video_fram_extractor_main()
    # h5reader('data/frame_data/p1_topdown_s1.h5', 'frames')
    # integrate_arousal_test()