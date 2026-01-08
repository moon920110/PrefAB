import os
import logging
import argparse
import yaml

import torch

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from dataloader.dataset import PairDataset, TestDataset
from dataloader.again_reader import AgainReader
from executor.ranknet_trainer import RanknetTrainer
from executor.cardinal_trainer import CardinalTrainer
from utils.vis import *
from utils.utils import *

logger = get_logger(__name__)


def load_dataset(config):
    all_dataset, numeric_columns, bio_features_size = AgainReader(config).prepare_sequential_ranknet_dataset()
    train_size = int(len(all_dataset) * config['train']['train_ratio'])
    test_size = len(all_dataset) - train_size
    train_samples, test_samples = torch.utils.data.random_split(all_dataset, [train_size, test_size])

    train_dataset = PairDataset(train_samples, numeric_columns, bio_features_size, config)
    test_dataset = TestDataset(test_samples, numeric_columns, config)

    return train_dataset, test_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PrefAB prototype')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    accelerator = Accelerator(
        mixed_precision=config['train']['mixed_precision']
    )

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    config['train']['exp'] = create_new_filename(config['train']['log_dir'], config['train']['exp'])
    if accelerator.is_main_process:
        if not os.path.exists(config['train']['log_dir']):
            os.makedirs(config['train']['log_dir'], exist_ok=True)

        if not os.path.exists(os.path.join(config['train']['log_dir'], f"{config['train']['exp']}")):
            os.makedirs(os.path.join(config['train']['log_dir'], f"{config['train']['exp']}"), exist_ok=True)

        fh = logging.FileHandler(os.path.join(config['train']['log_dir'], f"{config['train']['exp']}", 'log.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.logger.addHandler(fh)

    set_seed(config['train']['seed'])

    train, test = load_dataset(config)
    if config['train']['mode'] == 'non_ordinal':
        trainer = CardinalTrainer(train, test, config=config, logger=logger, accelerator=accelerator)
    else:
        trainer = RanknetTrainer(train, test, config=config, logger=logger, accelerator=accelerator)
    trainer.train()
    logger.info("Training is done!")