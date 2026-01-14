import os
import time
import json

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from network.prefab import Prefab
from executor.tester import RanknetTester
from dataloader.distributedWeightedSampler import DistributedWeightedSampler, WeightedSampler


class BaseTrainer:
    def __init__(self, train_dataset, test_dataset, config, logger, accelerator):
        self.config = config
        self.logger = logger
        self.accelerator = accelerator

        self.window_size = config['train']['window_size']
        self.mode = config['train']['mode']
        self.batch_size = config['train']['batch_size']

        train_size = int(len(train_dataset) * config['train']['train_ratio'])
        val_size = len(train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(train_dataset, [train_size, val_size])
        if val_size == 0:
            self.val_dataset = self.train_dataset

        self.meta_feature_size = train_dataset.get_meta_feature_size()
        self.bio_features_size = train_dataset.bio_features_size

        self.tester = RanknetTester(test_dataset, self.bio_features_size, config, logger)
        self.save_path = os.path.join(self.config['train']['save_dir'],
                                      f'{self.config["train"]["exp"]}_best.pth')

        self.model = Prefab(self.config, self.meta_feature_size, self.bio_features_size)
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None

        self.len_train_loader = None
        self.len_val_loader = None

    def _setup_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['train']['lr'])
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['train']['epoch'],
            eta_min=1e-6
        )

    def _prepare_dataloaders(self):
        is_distributed = self.accelerator.num_processes > 1

        if is_distributed:
            if self.config['train']['data_balancing']:
                train_sampler = DistributedWeightedSampler(
                    self.train_dataset,
                    num_replicas=self.accelerator.num_processes,
                    rank=self.accelerator.process_index
                )
            else:
                train_sampler = DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.accelerator.num_processes,
                    rank=self.accelerator.process_index,
                    shuffle=True
                )
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=False
            )
        else:
            train_sampler = WeightedSampler(self.train_dataset) if self.config['train']['data_balancing'] else None
            val_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.config['train']['num_workers'],
            shuffle=(train_sampler is None),
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=self.config['train']['num_workers'],
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )

    def _setup_writer(self):
        writer = None

        if self.accelerator.is_main_process:
            if not os.path.exists(self.config['train']['save_dir']):
                os.makedirs(self.config['train']['save_dir'])

            writer = SummaryWriter(
                log_dir=os.path.join(self.config['train']['log_dir'],
                                     f"{self.config['train']['exp']}"
                                     )
            )
            self.logger.info(f"Working at {time.strftime('%Y-%m-%d-%H-%M-%S')}")
            self.logger.info(json.dumps(self.config, indent=4, sort_keys=False))

        return writer

    def _set_epoch(self, epc):
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epc)
            self.logger.info(f'[Device {self.accelerator.process_index}] train sampler set epoch {epc}')
        iterator = tqdm(enumerate(self.train_loader), desc=f'Training Epoch {epc}',
                        disable=not self.accelerator.is_main_process, total=self.len_train_loader)
        return iterator

    def _save_model(self, acc, best_acc):
        if self.accelerator.is_main_process:
            if acc > best_acc:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                torch.save(unwrapped_model.state_dict(), self.save_path)
                self.logger.info(f'Best validation accuracy: {acc:.4f}, Saved to {self.save_path}')
                return acc
        return best_acc
