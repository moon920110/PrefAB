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


def tsne_demo(config, val_type):
    # trainset / testset

    if not os.path.exists(config['test']['log_dir']):
        os.makedirs(config['test']['log_dir'])
    config['test']['new_exp'] = create_new_filename(config['test']['log_dir'], f"{config['test']['exp']}_{val_type}")

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
    if val_type == 'train':
        test_dataset = TestDataset(train_samples, numeric_columns, config)
    else:
        test_dataset = TestDataset(test_samples, numeric_columns, config)

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

def time_efficiency_demo():
    root = '/home/jovyan/data/exp2'
    players = [f'p{i}' for i in range(1, 26)]
    sessions = [f's{i}' for i in range (1, 4)]

    results = {}
    for player in players:
        for session in sessions:
            dir_name = os.path.join(root, player, f'{player}_{session}')
            if os.path.exists(dir_name):
                efficiency, total_duration, clip_duration = compute_time_efficiency(root, player, session)
                print(f'{player}_{session}: efficiency: {efficiency} ({clip_duration} / {total_duration})')
                result = {
                    'efficiency': efficiency,
                    'clip_duration': clip_duration,
                    'total_duration': total_duration,
                }
                results[f'{player}_{session}'] = result
    results_df = pd.DataFrame(results)
    # transpose results_df
    results_df = results_df.T
    results_df.to_csv(os.path.join(root, 'time_efficiency.csv'))

    print(results_df)
    avg_efficiency = results_df['efficiency'].mean()
    avg_clip_duration = results_df['clip_duration'].mean()
    avg_total_duration = results_df['total_duration'].mean()
    print(f'average efficiency: {avg_efficiency} / average clip duration: {avg_clip_duration} / average total duration: {avg_total_duration}')


def auto_test(config):
    games = {
        'TinyCars': 'tinycars',
        'Solid': 'solid',
        # 'ApexSpeed': 'apex',
        # 'Heist!': 'heist',
        # 'Shootout': 'shootout',
        # 'TopDown': 'topdown',
        # "Run'N'Gun": 'runngun',
        # "Pirates!": 'pirates',
        # "Endless": 'endless'
    }

    for game, acronym in games.items():
        exps = [[f'regression_{acronym}_20', 'non_ordinal'],
                # [f'prefab_{acronym}_re', 'prefab'],
                # [f'prefab_v2_{acronym}', 'prefab']
                ]

        config['train']['game'] = game
        for exp, mode in exps:
            config['test']['exp'] = exp
            config['train']['mode'] = mode
            config['test']['mode'] = mode
            if not os.path.exists(os.path.join(config['test']['log_dir'], f"{exp}_train")):
                print(f'exp: {exp} for trainset')
                try:
                    tsne_demo(config, 'train')
                except:
                    os.remove(os.path.join(config['test']['log_dir'], f"{exp}_train"))
            if not os.path.exists(os.path.join(config['test']['log_dir'], f"{exp}_test")):
                print(f'exp: {exp} for testset')
                try:
                    tsne_demo(config, 'test')
                except:
                    shutil.rmtree(os.path.join(config['test']['log_dir'], f"{exp}_test"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PrefAB prototype')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    post_analysis_demo('Comparison', False)
    # tsne_demo(config, 'test')
    # tsne_demo(config, 'train')
    # time_efficiency_demo()
    # video_frame_extractor_main()
    # h5reader('data/frame_data/p1_topdown_s1.h5', 'frames')
    # integrate_arousal_test()