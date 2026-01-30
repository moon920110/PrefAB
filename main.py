import os
import logging
import argparse
import yaml

import torch

from accelerate import Accelerator
from accelerate.utils import set_seed, broadcast_object_list
from accelerate.logging import get_logger

from dataloader.dataset import PairDataset, TestDataset
from dataloader.again_reader import AgainReader
from executor.ranknet_trainer import RanknetTrainer
from executor.cardinal_trainer import CardinalTrainer
from utils.vis import *
from utils.utils import *

logger = get_logger(__name__)


def load_dataset(config, logger):
    all_dataset, numeric_columns, bio_features_size, game_metadata = AgainReader(config).prepare_sequential_ranknet_dataset()
    if config['train']['generalization']['activate']:
        logger.info("Generalization test")
        train_list = config['train']['generalization']['train_games']

        games_arr = np.array(game_metadata)
        train_mask = np.isin(games_arr, train_list)

        all_dataset_np = np.array(all_dataset, dtype=object)
        train_samples = all_dataset_np[train_mask].tolist()
        test_samples = all_dataset_np[~train_mask].tolist()

        if len(test_samples) == 0:
            train_size = int(len(all_dataset) * config['train']['train_ratio'])
            test_size = len(all_dataset) - train_size
            train_samples, test_samples = torch.utils.data.random_split(all_dataset, [train_size, test_size])
            pass

        logger.info(f"Train Games: {list(set(games_arr[train_mask]))}")
        logger.info(f"Test Games: {list(set(games_arr[~train_mask]))}")

        assert len(train_samples) > 0, f"Train set is empty. Check game names: {train_list}"
        assert len(test_samples) > 0, f"Test set is empty. All games are assigned to train set"
    else:  # train and test in a single game
        train_size = int(len(all_dataset) * config['train']['train_ratio'])
        test_size = len(all_dataset) - train_size
        train_samples, test_samples = torch.utils.data.random_split(all_dataset, [train_size, test_size])

    train_dataset = PairDataset(train_samples, numeric_columns, bio_features_size, config)
    test_dataset = TestDataset(test_samples, numeric_columns, config)

    return train_dataset, test_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PrefAB')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    accelerator = Accelerator(
        mixed_precision=config['train']['mixed_precision'],
    )

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # config['train']['exp'] = create_new_filename(config['train']['log_dir'], config['train']['exp'])
    exp_name = [config['train']['exp']]
    if accelerator.is_main_process:
        new_filename = create_new_filename(config['train']['log_dir'], config['train']['exp'])
        exp_name = [new_filename]

        if not os.path.exists(config['train']['log_dir']):
            os.makedirs(config['train']['log_dir'], exist_ok=True)

        if not os.path.exists(os.path.join(config['train']['log_dir'], f"{new_filename}")):
            os.makedirs(os.path.join(config['train']['log_dir'], f"{new_filename}"), exist_ok=True)

        fh = logging.FileHandler(os.path.join(config['train']['log_dir'], f"{new_filename}", 'log.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.logger.addHandler(fh)
    broadcast_object_list(exp_name, from_process=0)
    config['train']['exp'] = exp_name[0]

    set_seed(config['train']['seed'])

    train, test = load_dataset(config, logger)
    if config['train']['mode'] == 'non_ordinal':
        trainer = CardinalTrainer(train, test, config=config, logger=logger, accelerator=accelerator)
    else:
        trainer = RanknetTrainer(train, test, config=config, logger=logger, accelerator=accelerator)
    trainer.train()
    logger.info("Training is done!")