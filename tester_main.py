import os
import argparse

from dataloader.again_reader import AgainReader
import yaml

from utils.stats import find_significant_peaks_and_valleys, inflection_comparison, get_dtw_cluster, reconstruct_state_via_interpolation
from utils.video_frame_extractor import parse_images_from_video_by_timestamp


def dtw_cluster_demo():
    with open('../config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    game = 'Shootout'
    again_reader = AgainReader(config)
    # data = again_reader.game_info_by_name(game)
    data = again_reader.game_info_by_name('Shootout')
    print(f'len data: {len(data)}')

    find_significant_peaks_and_valleys(data['arousal'].values)
    get_dtw_cluster(data, config)


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
    dirs = [['log/prefab-topdown', True]]
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

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    again = AgainReader(config=config)

    parse_images_from_video_by_timestamp(video_path=os.path.join(again.data_path, config['data']['vision']['video']),
                                         out_dir=os.path.join(again.data_path, config['data']['vision']['frame']),
                                         again=again.again,
                                         config=config,
                                         transform=True,
                                         )


if __name__ == '__main__':
    post_analysis_demo('Comparison')
