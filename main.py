import os

import torch
import logging
import argparse
import yaml
import horovod

from dataloader.dataset import PairDataset, TestDataset
from executor.ranknet_trainer import RanknetTrainer
from executor.trainer import Trainer
from utils.vis import *
from utils.utils import *
from dataloader.again_reader import AgainReader


def train(config, dataset, testset):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(config['train']['log_dir'], f"{config['train']['exp']}", 'log.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if config['train']['mode'] == 'non_ordinal':
        trainer = Trainer(dataset, testset, config=config, logger=logger)
    else:
        trainer = RanknetTrainer(dataset, testset, config=config, logger=logger)
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

    dataset, numeric_columns, bio_features_size = AgainReader(config).prepare_sequential_ranknet_dataset()
    train_size = int(len(dataset) * config['train']['train_ratio'])
    test_size = len(dataset) - train_size
    train_samples, test_samples = torch.utils.data.random_split(dataset,
                                                                [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(
                                                                    config['train']['seed'])
                                                                )

    train_dataset = PairDataset(train_samples, numeric_columns, bio_features_size, config)
    test_dataset = TestDataset(test_samples, numeric_columns, config)
    if config['train']['distributed']['multi_gpu']:
        horovod.run(train,
                    args=(config, train_dataset, test_dataset),
                    np=config['train']['distributed']['num_gpus'])
    else:
        train(config, train_dataset, test_dataset)
