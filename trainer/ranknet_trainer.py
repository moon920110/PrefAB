import os
import time

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import horovod.torch as hvd
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score

from dataloader.pair_loader import PairLoader
from dataloader.distributedWeightedSampler import DistributedWeightedSampler, WeightedSampler
from network.ranknet import RankNet
from network.focalLoss import FocalLoss


class RanknetTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        dataset = PairLoader(config, logger)
        train_size = int(len(dataset) * config['train']['train_ratio'])
        val_size = int(len(dataset) * config['train']['val_ratio'])
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size]
        )

        # torch.set_float32_matmul_precision('high')
        if config['train']['distributed']['multi_gpu']:
            hvd.init()
            if torch.cuda.is_available():
                torch.cuda.set_device(hvd.local_rank())
                torch.cuda.manual_seed(config['train']['distributed']['seed'])
                self.device = torch.device(f'cuda', hvd.local_rank())
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        if not os.path.exists(self.config['train']['save_dir']):
            os.makedirs(self.config['train']['save_dir'])

        train_sampler = None
        val_sampler = None
        rank = 0
        if self.config['train']['distributed']['multi_gpu']:
            rank = hvd.rank()
            train_sampler = DistributedWeightedSampler(self.train_dataset, num_replicas=hvd.size(), rank=rank)
            val_sampler = DistributedWeightedSampler(self.val_dataset, num_replicas=hvd.size(), rank=rank)
            writer = SummaryWriter(
                log_dir=os.path.join(self.config['train']['log_dir'],
                                     f"{self.config['train']['exp']}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                                     )
            ) if rank == 0 else None
        else:
            writer = SummaryWriter(
                log_dir=os.path.join(self.config['train']['log_dir'],
                                     f"{self.config['train']['exp']}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                                     )
            )
            train_sampler = WeightedSampler(self.train_dataset)
            val_sampler = WeightedSampler(self.val_dataset)

        # (batch, pair, sequence, channel, height, width)
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.config['train']['batch_size'],
                                  sampler=train_sampler,
                                  num_workers=self.config['train']['num_workers'],
                                  shuffle=(train_sampler is None),
                                  pin_memory=True)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.config['train']['batch_size'],
                                sampler=val_sampler,
                                num_workers=self.config['train']['num_workers'],
                                shuffle=(val_sampler is None),
                                pin_memory=True)

        div = 4 if self.config['train']['distributed']['multi_gpu'] else 1
        len_train_loader = len(train_loader) // div
        len_val_loader = len(val_loader) // div

        self.logger.info(f'build model gpu: {rank}')
        model = RankNet(self.config)
        model.to(self.device)
        ae_criterion = nn.L1Loss().to(self.device)
        rank_criterion = FocalLoss().to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['train']['lr'])
        # compiled_model = torch.compile(model)
        if self.config['train']['distributed']['multi_gpu']:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=hvd.Adasum, gradient_predivide_factor=1.0)
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=len_train_loader * self.config['train']['schedule'], gamma=0.1)

        best_acc = 0
        prev_best = 0
        for epc in range(self.config['train']['epoch']):
            losses = 0
            d1, d2 = None, None
            model.train()

            if self.config['train']['distributed']['multi_gpu']:
                train_sampler.set_epoch(epc)
                self.logger.info(f'[gpu {rank}]train sampler set epoch {epc}')

            for i, (img1, feature1, img2, feature2, label) in tqdm(enumerate(train_loader), desc=f'Training Epoch {epc}'):
                img1 = img1.to(self.device)
                feature1 = feature1.to(self.device)
                img2 = img2.to(self.device)
                feature2 = feature2.to(self.device)
                label = label.to(self.device)

                optimizer.zero_grad()
                o, d1, d2 = model(img1, feature1, img2, feature2)
                ranknet_loss = rank_criterion(o, label)
                ae_loss = ae_criterion(d1, img1.view(-1, *img1.shape[2:])) + ae_criterion(d2, img2.view(-1, *img2.shape[2:]))
                loss = ranknet_loss + ae_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                acc, _ = self._metric(o, label)

                if writer:
                    writer.add_scalar(f'train/ranknet_loss', ranknet_loss.item(), epc * len_train_loader + i)
                    writer.add_scalar(f'train/ae_loss', ae_loss.item(), epc * len_train_loader + i)
                    writer.add_scalar(f'train/accuracy', acc, epc * len_train_loader + i)
                losses += loss.item()

            if d1 is not None and d2 is not None:
                out_for_saving1 = d1.view(int(d1.shape[0]/4), 4, *d1.shape[1:])[-1]
                out_for_saving2 = d2.view(int(d2.shape[0]/4), 4, *d2.shape[1:])[-1]

            self.logger.info(f'[gpu:{rank}]epoch {epc} avg. loss {losses / len_train_loader:.4f}')

            # write output image to tensorboard
            if writer:
                writer.add_images(f'train/output_{rank} epc_{epc}', out_for_saving1, epc, dataformats='NCHW')
                writer.add_images(f'train/output_{rank} epc_{epc}', out_for_saving2, epc, dataformats='NCHW')

            model.eval()
            with torch.no_grad():
                accs = 0
                cm = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
                d1, d2 = None, None
                for i, (img1, feature1, img2, feature2, label) in tqdm(enumerate(val_loader), desc=f'Evaluation'):
                    img1 = img1.to(self.device)
                    feature1 = feature1.to(self.device)
                    img2 = img2.to(self.device)
                    feature2 = feature2.to(self.device)
                    label = label.to(self.device)

                    o, d1, d2 = model(img1, feature1, img2, feature2)

                    acc, cm_tmp = self._metric(o, label)
                    cm += cm_tmp
                    accs += acc
                    if writer:
                        writer.add_scalar(f'val/accuracy', acc, epc * len_val_loader + i)

                if writer:
                    if d1 is not None and d2 is not None:
                        out_for_saving1 = d1.view(int(d1.shape[0] / 4), 4, *d1.shape[1:])[-1]
                        out_for_saving2 = d2.view(int(d2.shape[0] / 4), 4, *d2.shape[1:])[-1]
                    writer.add_images(f'val/output_{rank} epc_{epc}', out_for_saving1, epc, dataformats='NCHW')
                    writer.add_images(f'val/output_{rank} epc_{epc}', out_for_saving2, epc, dataformats='NCHW')

                    self.logger.info(f'[gpu:{rank}]epoch {epc} avg. val acc {accs / len_val_loader:.4f}')
                    cm = pd.DataFrame(cm, index=['dec', 'same', 'inc'], columns=['dec', 'same', 'inc'])
                    plt.figure(figsize=(30, 30))
                    sns.heatmap(cm, annot=True, cmap='Blues')
                    writer.add_figure(f'val/confusion_matrix_{rank} epc_{epc}', plt.gcf())

            if accs / len_val_loader > best_acc:
                prev_best = best_acc
                best_acc = accs / len_val_loader

            # model save if validation accuracy is the best
            if rank == 0:
                if prev_best < best_acc:
                    hvd.save_checkpoint(
                        model,
                        os.path.join(self.config['train']['save_dir'],
                                     f'ranknet{self.config["train"]["exp"]}_{epc}.pth')
                    )
        writer.close()

    def _metric(self, y_pred, y_true):
        # set all elements less than 1/3 to 0, greater than 2/3 to 1, otherwise 0.5 in y_pred
        # y_pred = torch.where(y_pred < 1/3, torch.zeros_like(y_pred), y_pred)
        # y_pred = torch.where(y_pred > 2/3, torch.ones_like(y_pred), y_pred)
        # y_pred = torch.where((y_pred >= 1/3) & (y_pred <= 2/3), torch.ones_like(y_pred) * 0.5, y_pred)

        # acc = (y_pred == y_true).sum().item() / len(y_pred)
        # cm = self._confusion_matrix(y_true, y_pred)

        acc = accuracy_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy().argmax(axis=1))
        cm = confusion_matrix(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy().argmax(axis=1), labels=[0, 1, 2])

        return acc, cm

    def _confusion_matrix(self, y_true, y_pred):
        result = np.zeros((3, 3))
        label = [0, 1, 2]

        for i, v in enumerate(label):
            v_idx = (y_true == v)
            p_v = y_pred[v_idx]

            if len(p_v) == 0:
                continue

            for j, p in enumerate(label):
                result[j, i] = (p_v == p).sum()
        return result