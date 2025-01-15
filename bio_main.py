import os

import torch
import logging
import argparse
import yaml
import horovod

from dataloader.dataset import BioDataset
from executor.bio_trainer import BioTrainer
from utils.vis import *
from utils.utils import *
from dataloader.again_reader import AgainReader
from dataloader.bio_reader import BioReader


def train(config, dataset):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(config['train']['log_dir'], f"{config['train']['exp']}", 'log.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Training start {len(dataset)}")
    trainer = BioTrainer(dataset, config=config, logger=logger)
    trainer.train()
    logger.info("Training is done!")


if __name__ == '__main__':
    os.environ['AUDIODEV'] = 'null'
    parser = argparse.ArgumentParser(description='PrefAB prototype')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists(config['train']['log_dir']):
        os.makedirs(config['train']['log_dir'])
    config['train']['exp'] = create_new_filename(config['train']['log_dir'], config['train']['exp'])

    if not os.path.exists(os.path.join(config['train']['log_dir'], f"{config['train']['exp']}")):
        os.makedirs(os.path.join(config['train']['log_dir'], f"{config['train']['exp']}"))

    again = AgainReader(config)
    bio = BioReader(again, config)
    dataset = bio.bio
    bio_feature_size = bio.bio_size

    train_dataset = BioDataset(dataset, bio_feature_size, config)
    if config['train']['distributed']['multi_gpu']:
        horovod.run(train,
                    args=(config, train_dataset),
                    np=config['train']['distributed']['num_gpus'])
    else:
        train(config, train_dataset)
