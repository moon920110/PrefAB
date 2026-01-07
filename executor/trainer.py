import os
import time
import json
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from dataloader.distributedWeightedSampler import DistributedWeightedSampler, WeightedSampler
from network.prefab import Prefab
from utils.utils import metric
from executor.tester import RanknetTester


class Trainer:
    def __init__(self, dataset, testset, config, logger, accelerator):
        self.config = config
        self.logger = logger
        self.accelerator = accelerator  # 객체 저장
        self.window_size = config['train']['window_size']
        self.mode = config['train']['mode']
        self.batch_size = config['train']['batch_size']

        train_size = int(len(dataset) * config['train']['train_ratio'])
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size]
        )

        self.meta_feature_size = dataset.get_meta_feature_size()
        self.bio_features_size = dataset.bio_features_size

        # Tester 초기화
        self.tester = RanknetTester(testset, self.bio_features_size, config, logger)
        self.save_path = os.path.join(self.config['train']['save_dir'],
                                      f'ranknet_{self.config["train"]["exp"]}_best.pth')

        self.model = Prefab(self.config, self.meta_feature_size, self.bio_features_size)

        self.ae_criterion = nn.L1Loss()
        self.main_criterion = nn.HuberLoss()
        self.aux_criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['train']['lr'])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.config['train']['schedule'], gamma=0.1)

        self._prepare_dataloaders()

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler
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
        else:
            train_sampler = WeightedSampler(self.train_dataset) if self.config['train']['data_balancing'] else None

        # [Val Sampler]
        if is_distributed:
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=False
            )
        else:
            val_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.config['train']['num_workers'],
            shuffle=(train_sampler is None),
            pin_memory=True,
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

    def train(self):
        if self.accelerator.is_main_process:
            if not os.path.exists(self.config['train']['save_dir']):
                os.makedirs(self.config['train']['save_dir'], exist_ok=True)

        writer = None
        if self.accelerator.is_main_process:
            writer = SummaryWriter(
                log_dir=os.path.join(self.config['train']['log_dir'], f"{self.config['train']['exp']}")
            )
            self.logger.info(f"Working at {time.strftime('%Y-%m-%d-%H-%M-%S')}")
            self.logger.info(json.dumps(self.config, indent=4, sort_keys=False))

        len_train_loader = len(self.train_loader)
        len_val_loader = len(self.val_loader)

        self.logger.info(f'build model on device: {self.accelerator.device}')

        best_acc = 0  # Regression이라 R2 score 기준

        for epc in range(self.config['train']['epoch']):
            losses = 0
            self.model.train()

            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epc)
                self.logger.info(f'[Device {self.accelerator.process_index}] train sampler set epoch {epc}')

            iterator = tqdm(enumerate(self.train_loader),
                            desc=f'Training Epoch {epc}',
                            disable=not self.accelerator.is_main_process,
                            total=len_train_loader)

            for i, (img1, feature1, bio, label, aux_label) in iterator:
                self.optimizer.zero_grad()

                o, a_o = self.model(img1, feature1, bio)
                o = o.squeeze()

                main_loss = self.main_criterion(o, label)
                aux_loss = self.aux_criterion(a_o, aux_label)

                loss = main_loss + aux_loss * self.config['train']['aux_loss_weight']

                # [Backpropagation]
                self.accelerator.backward(loss)
                self.optimizer.step()

                # Metrics (Regression R2)
                acc = metric(o.detach(), label.detach(), infer_type='regression')
                aux_acc, _ = metric(a_o.detach(), aux_label.detach(), infer_type='classification')

                if writer:
                    writer.add_scalar(f'train/main_loss', main_loss.item(), epc * len_train_loader + i)
                    writer.add_scalar(f'train/aux_loss', aux_loss.item(), epc * len_train_loader + i)
                    writer.add_scalar(f'train/r2_score', acc, epc * len_train_loader + i)
                    writer.add_scalar(f'train/aux_accuracy', aux_acc, epc * len_train_loader + i)
                    writer.add_scalar(f'train/loss', loss.item(), epc * len_train_loader + i)
                losses += loss.item()

            self.scheduler.step()

            log_msg = (
                f"[Device {self.accelerator.process_index}] "
                f"Epoch {epc} | Avg Loss: {losses / len_train_loader:.4f}"
            )
            self.logger.info(log_msg, main_process_only=False)

            self.model.eval()
            with torch.no_grad():
                accs = 0
                val_aux_accs = 0
                d1 = None

                iterator = tqdm(enumerate(self.val_loader),
                                desc=f'Evaluation',
                                disable=not self.accelerator.is_main_process,
                                total=len_val_loader)

                for i, (img1, feature1, bio, label, aux_label) in iterator:
                    o1, a_o, d1 = self.model(img1, feature1, bio)
                    o1 = o1.squeeze()

                    acc = metric(o1, label, infer_type='regression')
                    aux_acc, _ = metric(a_o, aux_label, infer_type='classification')

                    accs += acc
                    val_aux_accs += aux_acc

                    if writer:
                        writer.add_scalar(f'val/r2_score', acc, epc * len_val_loader + i)
                        writer.add_scalar(f'val/aux_accuracy', aux_acc, epc * len_val_loader + i)

                avg_acc = accs / len_val_loader
                avg_aux_acc = val_aux_accs / len_val_loader

                val_log_msg = (
                    f"[Device {self.accelerator.process_index}] "
                    f"Epoch {epc} Validation | R2 Score: {avg_acc:.4f} | Aux Acc: {avg_aux_acc:.4f}"
                )
                self.logger.info(val_log_msg, main_process_only=False)

            if self.accelerator.is_main_process:
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    torch.save(unwrapped_model.state_dict(), self.save_path)
                    self.logger.info(f"Saved best model to {self.save_path}")

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.tester.test(writer, self.save_path)

        if writer is not None:
            writer.close()