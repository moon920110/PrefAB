import os
import time
import json

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from dataloader.distributedWeightedSampler import DistributedWeightedSampler, WeightedSampler
from dataloader.dataset import PairDataset, TestDataset
from dataloader.again_reader import AgainReader
from network.prefab import Prefab
from network.loss import OrdinalCrossEntropyLoss
from executor.tester import RanknetTester
from utils.utils import metric


class RanknetTrainer:
    def __init__(self, config, logger, accelerator):
        self.config = config
        self.logger = logger
        self.accelerator = accelerator
        self.window_size = config['train']['window_size']
        self.mode = config['train']['mode']
        self.batch_size = config['train']['batch_size']

        all_dataset, numeric_columns, bio_features_size = AgainReader(config).prepare_sequential_ranknet_dataset()
        train_size = int(len(all_dataset) * config['train']['train_ratio'])
        test_size = len(all_dataset) - train_size
        train_samples, test_samples = torch.utils.data.random_split(all_dataset, [train_size, test_size])

        train_dataset = PairDataset(train_samples, numeric_columns, bio_features_size, config)
        test_dataset = TestDataset(test_samples, numeric_columns, config)

        train_size = int(len(train_dataset) * config['train']['train_ratio'])
        val_size = len(train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            train_dataset,
            [train_size, val_size]
        )
        if val_size == 0:
            self.val_dataset = self.train_dataset

        self.meta_feature_size = train_dataset.get_meta_feature_size()
        self.bio_features_size = train_dataset.bio_features_size
        self.tester = RanknetTester(test_dataset, self.bio_features_size, config, logger)
        self.save_path = os.path.join(self.config['train']['save_dir'],
                                      f'ranknet_{self.config["train"]["exp"]}_best.pth')

        self.model = Prefab(self.config, self.meta_feature_size, self.bio_features_size)
        if self.config['train']['fine_tune']:
            model_path = os.path.join(self.config['train']['save_dir'], self.config['experiment']['model'])
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.logger.info(f'Model loaded from {model_path} for fine-tuning')

        self.rank_criterion = OrdinalCrossEntropyLoss(self.config['train']['cutpoints'])
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

        len_train_loader = len(self.train_loader)
        len_val_loader = len(self.val_loader)
        best_acc = 0

        # (batch, pair, sequence, channel, height, width)
        for epc in range(self.config['train']['epoch']):
            losses = 0
            cm = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.model.train()
            l0_cnt, l1_cnt, l2_cnt = 0, 0, 0

            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epc)
                self.logger.info(f'[Device {self.accelerator.process_index}] train sampler set epoch {epc}')

            iterator = tqdm(enumerate(self.train_loader), desc=f'Training Epoch {epc}',
                            disable=not self.accelerator.is_main_process, total=len_train_loader)
            for i, (img1, feature1, img2, feature2, bio, label, aux_label) in iterator:
                ### Accelerator will automatically move the data to the device
                # img1 = img1.to(self.device)
                # feature1 = feature1.to(self.device)
                # img2 = img2.to(self.device)
                # feature2 = feature2.to(self.device)
                # label = label.to(self.device)
                # bio = bio.to(self.device)
                # aux_label = aux_label.to(self.device)

                l0_cnt += len(label[label == 0])
                l1_cnt += len(label[label == 1])
                l2_cnt += len(label[label == 2])

                img_cat = torch.cat([img1, img2], dim=0)
                feature_cat = torch.cat([feature1, feature2], dim=0)
                bio_cat = torch.cat([bio, bio], dim=0)

                self.optimizer.zero_grad()

                o_cat, a_o_cat = self.model(img_cat, feature_cat, bio_cat)
                # o2, a_o2, d2 = self.model(img2, feature2, bio)
                o1, o2 = torch.chunk(o_cat, 2, dim=0)
                a_o1, a_o2 = torch.chunk(a_o_cat, 2, dim=0)
                o = o2 - o1

                ranknet_loss = self.rank_criterion(o, label)
                aux_loss = torch.tensor(0, device=self.accelerator.device)
                if not self.config['train']['ablation']['aux']:
                    aux_loss = self.aux_criterion(a_o1, aux_label) + self.aux_criterion(a_o2, aux_label)

                loss = ranknet_loss + aux_loss * self.config['train']['aux_loss_weight']

                self.accelerator.backward(loss)
                self.optimizer.step()

                # acc, cm_tmp = metric(o, label, self.config['train']['cutpoints'])
                acc, cm_tmp = metric(o.detach(), label.detach(), self.rank_criterion.get_cutpoints())
                aux_acc1, _ = metric(a_o1, aux_label, infer_type='classification')
                aux_acc2, _ = metric(a_o2, aux_label, infer_type='classification')
                cm += cm_tmp

                if writer:
                    writer.add_scalar(f'train/ranknet_loss', ranknet_loss.item(), epc * len_train_loader + i)
                    writer.add_scalar(f'train/aux_loss', aux_loss.item(), epc * len_train_loader + i)
                    writer.add_scalar(f'train/accuracy', acc, epc * len_train_loader + i)
                    writer.add_scalar(f'train/aux_accuracy_1', aux_acc1, epc * len_train_loader + i)
                    writer.add_scalar(f'train/aux_accuracy_2', aux_acc2, epc * len_train_loader + i)
                    writer.add_scalar(f'train/loss', loss.item(), epc * len_train_loader + i)
                losses += loss.item()

            self.scheduler.step()

            log_msg = (f"[Device {self.accelerator.process_index}] Epoch {epc} | "
                       f"Avg Loss: {losses / len_train_loader:.4f} | "
                       f"L0: {l0_cnt} / L1: {l1_cnt} / L2: {l2_cnt} | "
                       f"Cutpoints: {self.config['train']['cutpoints']}")
            self.logger.info(log_msg, main_process_only=False)

            # write output image to tensorboard
            if writer:
                # if d1 is not None and d2 is not None:
                #     out_for_saving1 = d1.view(int(d1.shape[0] / self.window_size), self.window_size, *d1.shape[1:])[-1]
                #     out_for_saving2 = d2.view(int(d2.shape[0] / self.window_size), self.window_size, *d2.shape[1:])[-1]
                #     writer.add_images(f'train/epc_{epc}_output_1', out_for_saving1, epc, dataformats='NCHW')
                #     writer.add_images(f'train/epc_{epc}_output_2', out_for_saving2, epc, dataformats='NCHW')
                cm = pd.DataFrame(cm, index=['dec', 'same', 'inc'], columns=['dec', 'same', 'inc'])
                plt.figure(figsize=(10, 10))
                sns.heatmap(cm, annot=True, cmap='Blues')
                writer.add_figure(f'train/confusion_matrix epc_{epc}', plt.gcf())
                plt.close()

            self.model.eval()
            accs = 0
            cm = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

            with torch.no_grad():
                d1, d2 = None, None
                iterator = tqdm(
                    enumerate(self.val_loader),
                    desc=f'Evaluation',
                    disable=not self.accelerator.is_main_process,
                    total=len(self.val_loader)
                )
                for i, (img1, feature1, img2, feature2, bio, label, aux_label) in iterator:
                    # img1 = img1.to(self.device)
                    # feature1 = feature1.to(self.device)
                    # img2 = img2.to(self.device)
                    # feature2 = feature2.to(self.device)
                    # label = label.to(self.device)
                    # bio = bio.to(self.device)
                    # aux_label = aux_label.to(self.device)

                    o1, a_o1 = self.model(img1, feature1, bio)
                    o2, a_o2 = self.model(img2, feature2, bio)
                    o = o2 - o1

                    # acc, cm_tmp = metric(o, label, self.config['train']['cutpoints'])
                    acc, cm_tmp = metric(o, label, self.rank_criterion.get_cutpoints())
                    aux_acc1, _ = metric(a_o1, aux_label, infer_type='classification')
                    aux_acc2, _ = metric(a_o2, aux_label, infer_type='classification')

                    cm += cm_tmp
                    accs += acc
                    if writer:
                        writer.add_scalar(f'val/accuracy', acc, epc * len_val_loader + i)
                        writer.add_scalar(f'val/aux_accuracy_1', aux_acc1, epc * len_val_loader + i)
                        writer.add_scalar(f'val/aux_accuracy_2', aux_acc2, epc * len_val_loader + i)

                avg_acc = accs / len_val_loader
                cm_str = np.array_str(cm).replace('\n', '')
                val_log_msg = (f"[Device {self.accelerator.process_index}] Epoch {epc} | "
                               f"Avg Accuracy: {avg_acc:.4f} | "
                               f"Confusion Matrix [Dec/Same/Inc]: {cm_str}")
                self.logger.info(val_log_msg, main_process_only=False)

                if writer:
                    # if d1 is not None and d2 is not None:
                    #     out_for_saving1 = d1.view(int(d1.shape[0] / self.window_size), self.window_size, *d1.shape[1:])[-1]
                    #     out_for_saving2 = d2.view(int(d2.shape[0] / self.window_size), self.window_size, *d2.shape[1:])[-1]
                    #     writer.add_images(f'val/epc_{epc}_output_1', out_for_saving1, epc, dataformats='NCHW')
                    #     writer.add_images(f'val/epc_{epc}_output_2', out_for_saving2, epc, dataformats='NCHW')

                    cm = pd.DataFrame(cm, index=['dec', 'same', 'inc'], columns=['dec', 'same', 'inc'])
                    plt.figure(figsize=(10, 10))
                    sns.heatmap(cm, annot=True, cmap='Blues')
                    writer.add_figure(f'val/confusion_matrix epc_{epc}', plt.gcf())
                    plt.close()

            # model save if validation accuracy is the best
            if self.accelerator.is_main_process:
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    torch.save(unwrapped_model.state_dict(), self.save_path)
                    self.logger.info(f'Best validation accuracy: {best_acc:.4f}')

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.tester.test(writer, self.save_path)

        if writer:
            writer.close()
