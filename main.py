import os

import logging
import argparse
import yaml
import horovod

from dataloader.dataset import PairDataset, TestDataset
from trainer.ranknet_trainer import RanknetTrainer
from trainer.trainer import Trainer
from utils.vis import *
from utils.utils import *


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

    dataset = PairDataset(config)
    t_dataset = TestDataset(dataset.dataset, dataset.numeric_columns, config)
    if config['train']['distributed']['multi_gpu']:
        horovod.run(train,
                    args=(config, dataset, t_dataset),
                    np=config['train']['distributed']['num_gpus'])
    else:
        train(config, dataset, t_dataset)
