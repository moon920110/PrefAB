import logging
import argparse
import yaml

from trainer.ranknet_trainer import RanknetTrainer
from utils.vis import *


if __name__ == '__main__':
    os.environ['AUDIODEV'] = 'null'
    parser = argparse.ArgumentParser(description='PrefAB prototype')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler('log.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    trainer = RanknetTrainer(config=config, logger=logger)
    trainer.train()
