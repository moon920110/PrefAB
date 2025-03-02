import os
import pandas as pd

from utils.preprocessing import cleaning_logs, integrate_arousal
from utils.video_frame_extractor import parse_AGAIN_images


def preprocess(config, logger):
    player = config['experiment']['player']
    session = config['experiment']['session']
    logger.info(f'player: {player}, session: {session}')

    cleaned_log = cleaning_logs(config, logger)
    parse_AGAIN_images(cleaned_log, config, logger)

    integrate_arousal(config)


def migrate_clean_data():
    root = 'data/clean_data'
    main_clean = None
    for file in os.listdir(root):
        if file == 'clean_data_custom.csv' or file == 'clean_data.csv':
            continue
        if main_clean is None:
            main_clean = pd.read_csv(os.path.join(root, file))
        else:
            df = pd.read_csv(os.path.join(root, file))
            main_clean = pd.concat([main_clean, df], ignore_index=True)
    main_clean.to_csv(os.path.join(root, 'clean_data_custom.csv'), index=False)

    sessions = main_clean['session_id'].unique()
    sessions = [session for session in sessions if 's4' in session]

    data_s4 = main_clean[main_clean['session_id'].isin(sessions)]
    data_not_s4 = main_clean[~main_clean['session_id'].isin(sessions)]
    data_s4.to_csv(os.path.join(root, 'clean_data_custom_test.csv'), index=False)
    data_not_s4.to_csv(os.path.join(root, 'clean_data_custom_train.csv'), index=False)


if __name__ == '__main__':
    import yaml
    import logging

    with open('./config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(config['train']['log_dir'], 'exp_inference.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    players = [f'p{i}' for i in range(1, 26)]
    sessions = [f's{i}' for i in range(1, 5)]
    for player in players:
        for session in sessions:
            config['experiment']['player'] = player
            config['experiment']['session'] = f'{player}{session}'
            if not os.path.exists(os.path.join('data/clean_data', f'{player}_topdown_{player}{session}_clean.csv')):
                preprocess(config, logger)

    migrate_clean_data()
