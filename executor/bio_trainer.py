import os
import time
import json

import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import horovod.torch as hvd
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import confusion_matrix, accuracy_score

from dataloader.distributedWeightedSampler import DistributedWeightedSampler, WeightedSampler
from network.bio import BioNet


class BioTrainer:
    def __init__(self, dataset, config, logger):
        self.config = config
        self.logger = logger
        self.bio_feature_size = dataset.bio_feature_size
        self.batch_size = config['train']['batch_size']

        train_size = int(len(dataset) * self.config['train']['train_ratio'])
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset,
                                                            [train_size, val_size],
                                                            generator=torch.Generator().manual_seed(
                                                                self.config['train']['seed'])
                                                            )

        # torch.set_float32_matmul_precision('high')
        if config['train']['distributed']['multi_gpu']:
            hvd.init()
            if torch.cuda.is_available():
                torch.cuda.set_device(hvd.local_rank())
                torch.cuda.manual_seed(config['train']['seed'])
                self.device = torch.device(f'cuda', hvd.local_rank())
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        if not os.path.exists(self.config['train']['save_dir']):
            os.makedirs(self.config['train']['save_dir'])

        rank = 0
        if self.config['train']['distributed']['multi_gpu']:
            rank = hvd.rank()
            train_sampler = DistributedWeightedSampler(self.train_dataset, num_replicas=hvd.size(), rank=rank) if self.config['train']['data_balancing'] else DistributedSampler(self.train_dataset, num_replicas=hvd.size(), rank=rank)
            val_sampler = DistributedWeightedSampler(self.val_dataset, num_replicas=hvd.size(), rank=rank)  #  if self.config['train']['data_balancing'] else DistributedSampler(self.val_dataset, num_replicas=hvd.size(), rank=rank)

            if rank == 0:
                writer = SummaryWriter(
                    log_dir=os.path.join(self.config['train']['log_dir'],
                                         f"{self.config['train']['exp']}"
                                         )
                )
                self.logger.info(f"Working at {time.strftime('%Y-%m-%d-%H-%M-%S')}")
                self.logger.info(json.dumps(self.config, indent=4, sort_keys=False))
            else:
                writer = None
        else:
            writer = SummaryWriter(
                log_dir=os.path.join(self.config['train']['log_dir'],
                                     f"{self.config['train']['exp']}"
                                     )
            )
            train_sampler = WeightedSampler(self.train_dataset) if self.config['train']['data_balancing'] else None
            val_sampler = WeightedSampler(self.val_dataset) if self.config['train']['data_balancing'] else None

        # (batch, pair, sequence, channel, height, width)
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  sampler=train_sampler,
                                  num_workers=self.config['train']['num_workers'],
                                  shuffle=(train_sampler is None),
                                  pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                sampler=val_sampler,
                                num_workers=self.config['train']['num_workers'],
                                shuffle=(val_sampler is None),
                                pin_memory=True,
                                drop_last=True)

        train_div = self.config['train']['distributed']['num_gpus'] if self.config['train']['distributed']['multi_gpu'] and self.config['train']['data_balancing'] else 1
        eval_div = self.config['train']['distributed']['num_gpus'] if self.config['train']['distributed']['multi_gpu'] else 1
        len_train_loader = len(train_loader) // train_div if len(train_loader) > train_div else 1
        len_val_loader = len(val_loader) // eval_div if len(val_loader) > eval_div else 1

        self.logger.info(f'build model gpu: {rank}')
        model = BioNet(self.config, self.bio_feature_size)
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['train']['lr'])
        # compiled_model = torch.compile(model)
        if self.config['train']['distributed']['multi_gpu']:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=hvd.Adasum, gradient_predivide_factor=1.0)
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config['train']['schedule'], gamma=0.1)

        best_acc = 0

        for epc in range(self.config['train']['epoch']):
            losses = 0
            cm = np.zeros((4, 4))
            model.train()

            if self.config['train']['distributed']['multi_gpu']:
                train_sampler.set_epoch(epc)
                self.logger.info(f'[gpu {rank}]train sampler set epoch {epc}')

            for i, (bio, label) in tqdm(enumerate(train_loader), desc=f'Training Epoch {epc}'):
                bio = bio.to(self.device)
                label = label.to(self.device)

                optimizer.zero_grad()
                o = model(bio)
                o = o.squeeze()

                loss = criterion(o, label)

                loss.backward()
                optimizer.step()
                acc, cm_tmp = self._metric(o, label)
                cm += cm_tmp

                if writer:
                    writer.add_scalar(f'train/bio_loss', loss.item(), epc * len_train_loader + i)
                    writer.add_scalar(f'train/acc', acc, epc * len_train_loader + i)
                losses += loss.item()

            scheduler.step()
            self.logger.info(f'[gpu:{rank}]epoch {epc} avg. loss {losses / len_train_loader:.4f} ')

            if writer:
                cm = pd.DataFrame(cm, index=['c1', 'c2', 'c3', 'c4'], columns=['c1', 'c2', 'c3', 'c4'])
                plt.figure(figsize=(30, 30))
                sns.heatmap(cm, annot=True, cmap='Blues')
                writer.add_figure(f'train/confusion_matrix_{epc}', plt.gcf())

            model.eval()
            with torch.no_grad():
                accs = 0
                cm = np.zeros((4, 4))

                for i, (bio, label) in tqdm(enumerate(val_loader), desc=f'Evaluation'):
                    bio = bio.to(self.device)
                    label = label.to(self.device)

                    o1 = model(bio)
                    o1 = o1.squeeze()

                    acc, cm_tmp = self._metric(o1, label)
                    accs += acc
                    cm += cm_tmp
                    if writer:
                        writer.add_scalar(f'val/acc', acc, epc * len_val_loader + i)

                avg_acc = accs / len_val_loader
                if writer:
                    self.logger.info(f'[gpu:{rank}]epoch {epc} avg. val acc {avg_acc:.4f}')
                    cm = pd.DataFrame(cm, index=['c1', 'c2', 'c3', 'c4'], columns=['c1', 'c2', 'c3', 'c4'])
                    plt.figure(figsize=(30, 30))
                    sns.heatmap(cm, annot=True, cmap='Blues')
                    writer.add_figure(f'val/confusion_matrix_{epc}', plt.gcf())

            # model save if validation accuracy is the best
            if rank == 0:
                if avg_acc > best_acc and avg_acc > 0.70:
                    best_acc = avg_acc
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.config['train']['save_dir'],
                                     f'ranknet_{self.config["train"]["exp"]}_{epc}_{avg_acc*100:.2f}.pth')
                    )
        if writer is not None:
            writer.close()

    def _metric(self, y_pred, y_true):
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = y_pred.argmax(axis=1)
        y_true = y_true.cpu().detach().numpy()

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

        return acc, cm
