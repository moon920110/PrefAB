import os
import logging
import argparse
import yaml

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from executor.ranknet_trainer import RanknetTrainer
from executor.trainer import Trainer
from utils.vis import *
from utils.utils import *

logger = get_logger(__name__)

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


    if config['train']['mode'] == 'non_ordinal':
        trainer = Trainer(config=config, logger=logger, accelerator=accelerator)
    else:
        trainer = RanknetTrainer(config=config, logger=logger, accelerator=accelerator)
    trainer.train()
    logger.info("Training is done!")