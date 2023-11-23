import os

import logging
import argparse
import yaml
import horovod

from trainer.ranknet_trainer import RanknetTrainer
from utils.vis import *


def train(config):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if not os.path.exists('log'):
        os.makedirs('log')
    fh = logging.FileHandler('log/log.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    trainer = RanknetTrainer(config=config, logger=logger)
    trainer.train()


if __name__ == '__main__':
    os.environ['AUDIODEV'] = 'null'
    parser = argparse.ArgumentParser(description='PrefAB prototype')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config['train']['distributed']['multi_gpu']:
        horovod.run(train,
                    args=(config,),
                    np=config['train']['distributed']['num_gpus'])
    else:
        train(config)
