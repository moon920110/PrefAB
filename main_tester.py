import os
import time
import json

import logging
import argparse
import yaml
import horovod

from dataloader.pair_loader import PairLoader
from trainer.ranknet_trainer import RanknetTrainer
from utils.vis import *
from utils.utils import *


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

    logger.info(f"Working at {time.strftime('%Y-%m-%d-%H-%M-%S')}")
    logger.info(json.dumps(config, indent=4, sort_keys=False))

    trainer = RanknetTrainer(dataset, config=config, logger=logger)
    trainer.train()


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

    dataset = PairLoader(config)
    if config['train']['distributed']['multi_gpu']:
        horovod.run(train,
                    args=(config, dataset),
                    np=config['train']['distributed']['num_gpus'])
    else:
        train(config, dataset)
