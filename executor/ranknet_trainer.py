import os
import time
import json

import dtw
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
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import confusion_matrix, accuracy_score

from dataloader.distributedWeightedSampler import DistributedWeightedSampler, WeightedSampler
from network.prefab import Prefab
from network.loss import FocalLoss, OrdinalCrossEntropyLoss


class RanknetTrainer:
    def __init__(self, dataset, testset, config, logger):
        self.config = config
        self.logger = logger
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
        self.test_dataset = testset

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
        model = Prefab(self.config, self.meta_feature_size, self.bio_features_size)
        model.to(self.device)
        ae_criterion = nn.L1Loss().to(self.device)
        rank_criterion = OrdinalCrossEntropyLoss(self.config['train']['cutpoints']).to(self.device)  # FocalLoss(alpha=self.config['train']['focal_alpha'], gamma=self.config['train']['focal_gamma']).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['train']['lr'])
        # compiled_model = torch.compile(model)
        if self.config['train']['distributed']['multi_gpu']:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=hvd.Adasum, gradient_predivide_factor=1.0)
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=len_train_loader * self.config['train']['schedule'], gamma=0.1)

        best_acc = 0

        for epc in range(self.config['train']['epoch']):
            losses = 0
            d1, d2 = None, None
            cm = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            model.train()
            l0_cnt = 0
            l1_cnt = 0
            l2_cnt = 0

            if self.config['train']['distributed']['multi_gpu']:
                train_sampler.set_epoch(epc)
                self.logger.info(f'[gpu {rank}]train sampler set epoch {epc}')

            for i, (img1, feature1, img2, feature2, bio, label) in tqdm(enumerate(train_loader), desc=f'Training Epoch {epc}'):
                img1 = img1.to(self.device)
                feature1 = feature1.to(self.device)
                img2 = img2.to(self.device)
                feature2 = feature2.to(self.device)
                label = label.to(self.device)
                bio = bio.to(self.device)

                l0_cnt += len(label[label == 0])
                l1_cnt += len(label[label == 1])
                l2_cnt += len(label[label == 2])

                optimizer.zero_grad()
                o1, d1 = model(img1, feature1, bio)
                o2, d2 = model(img2, feature2, bio)
                o = o2 - o1

                ranknet_loss = rank_criterion(o, label)
                if self.mode != 'feature':
                    ae_loss = ae_criterion(d1, img1.view(-1, *img1.shape[2:])) + ae_criterion(d2, img2.view(-1, *img2.shape[2:]))
                    # ae_loss = 0
                    loss = ranknet_loss + ae_loss * self.config['train']['ae_loss_weight']
                else:
                    loss = ranknet_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                acc, cm_tmp = self._metric(o, label, self.config['train']['cutpoints'])
                cm += cm_tmp

                if writer:
                    writer.add_scalar(f'train/ranknet_loss', ranknet_loss.item(), epc * len_train_loader + i)
                    if self.mode != 'feature':
                        writer.add_scalar(f'train/ae_loss', ae_loss.item(), epc * len_train_loader + i)
                    writer.add_scalar(f'train/accuracy', acc, epc * len_train_loader + i)
                    writer.add_scalar(f'train/loss', loss.item(), epc * len_train_loader + i)
                losses += loss.item()


            total_cnt = l0_cnt + l1_cnt + l2_cnt
            self.logger.info(f'[gpu:{rank}]epoch {epc} avg. loss {losses / len_train_loader:.4f} '
                             f'l0_cnt {l0_cnt} '
                             f'l1_cnt {l1_cnt} '
                             f'l2_cnt {l2_cnt}')

            # write output image to tensorboard
            if writer:
                if d1 is not None and d2 is not None:
                    out_for_saving1 = d1.view(int(d1.shape[0] / self.window_size), self.window_size, *d1.shape[1:])[-1]
                    out_for_saving2 = d2.view(int(d2.shape[0] / self.window_size), self.window_size, *d2.shape[1:])[-1]
                    writer.add_images(f'train/epc_{epc}_output_1', out_for_saving1, epc, dataformats='NCHW')
                    writer.add_images(f'train/epc_{epc}_output_2', out_for_saving2, epc, dataformats='NCHW')
                cm = pd.DataFrame(cm, index=['dec', 'same', 'inc'], columns=['dec', 'same', 'inc'])
                plt.figure(figsize=(30, 30))
                sns.heatmap(cm, annot=True, cmap='Blues')
                writer.add_figure(f'train/confusion_matrix_{rank} epc_{epc}', plt.gcf())

            model.eval()
            with torch.no_grad():
                accs = 0
                cm = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
                d1, d2 = None, None
                for i, (img1, feature1, img2, feature2, bio, label) in tqdm(enumerate(val_loader), desc=f'Evaluation'):
                    img1 = img1.to(self.device)
                    feature1 = feature1.to(self.device)
                    img2 = img2.to(self.device)
                    feature2 = feature2.to(self.device)
                    label = label.to(self.device)
                    bio = bio.to(self.device)

                    o1, d1 = model(img1, feature1, bio)
                    o2, d2 = model(img2, feature2, bio)
                    o = o2 - o1

                    acc, cm_tmp = self._metric(o, label, self.config['train']['cutpoints'])
                    cm += cm_tmp
                    accs += acc
                    if writer:
                        writer.add_scalar(f'val/accuracy', acc, epc * len_val_loader + i)

                avg_acc = accs / len_val_loader
                if writer:
                    if d1 is not None and d2 is not None:
                        out_for_saving1 = d1.view(int(d1.shape[0] / self.window_size), self.window_size, *d1.shape[1:])[-1]
                        out_for_saving2 = d2.view(int(d2.shape[0] / self.window_size), self.window_size, *d2.shape[1:])[-1]
                        writer.add_images(f'val/epc_{epc}_output_1', out_for_saving1, epc, dataformats='NCHW')
                        writer.add_images(f'val/epc_{epc}_output_2', out_for_saving2, epc, dataformats='NCHW')

                    self.logger.info(f'[gpu:{rank}]epoch {epc} avg. val acc {avg_acc:.4f}')
                    cm = pd.DataFrame(cm, index=['dec', 'same', 'inc'], columns=['dec', 'same', 'inc'])
                    plt.figure(figsize=(30, 30))
                    sns.heatmap(cm, annot=True, cmap='Blues')
                    writer.add_figure(f'val/confusion_matrix_{rank} epc_{epc}', plt.gcf())


            # model save if validation accuracy is the best
            if rank == 0:
                if avg_acc > best_acc and avg_acc > 0.70:
                    best_acc = avg_acc
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.config['train']['save_dir'],
                                     f'ranknet_{self.config["train"]["exp"]}_best.pth')
                    )

        self._test_per_player(model, 10, writer)
        if writer is not None:
            writer.close()

    def _metric(self, y_pred, y_true, cutpoints):
        _y_pred = y_pred.cpu().detach().numpy()
        _y_pred = np.where(_y_pred < cutpoints[0], 0, np.where(_y_pred < cutpoints[1], 1, 2))

        acc = accuracy_score(y_true.cpu().detach().numpy(), _y_pred)
        cm = confusion_matrix(y_true.cpu().detach().numpy(), _y_pred, labels=[0, 1, 2])

        return acc, cm

    def _test_per_player(self, model, size, writer):
        # load best model
        model.load_state_dict(
            torch.load(
                os.path.join(self.config['train']['save_dir'],
                             f'ranknet_{self.config["train"]["exp"]}_best.pth')
            )
        )
        model.to(self.device)
        model.eval()

        indices = self.test_dataset.sample_player_data(size)
        dtw_distances = []
        for i, idx in tqdm(enumerate(indices), desc='Reconstructing Graphs'):
            start_idx = self.test_dataset.player_idx[idx]
            end_idx = self.test_dataset.player_idx[idx + 1]

            outputs = []
            labels = []
            imgs = []
            features = []
            bios = []
            for data_idx in range(start_idx, end_idx):
                img, feature, bio, y = self.test_dataset[data_idx]
                imgs.append(img)
                features.append(feature)
                bios.append(bio)
                labels.append(y)

                if len(imgs) == self.batch_size or data_idx == end_idx - 1:
                    imgs = torch.stack(imgs).to(self.device)
                    features = torch.stack(features).to(self.device)
                    bios = torch.stack(bios).to(self.device)
                    o, _ = model(imgs, features, bios)
                    o = o.cpu().detach().numpy()
                    outputs.extend(o)

                    imgs = []
                    features = []
                    bios = []

            # normalize output to 0~1
            outputs = np.array(outputs).squeeze().squeeze()
            outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
            # dtw_distances.append(dtw.dtw(outputs, np.array(labels)).distance)

            for ii, (o, y) in enumerate(zip(outputs, labels)):
                if writer:
                    writer.add_scalars(f'test/player_{idx}',
                                       {'predict': o,
                                        'arousal': y,
                                       },
                                       ii)

        # dtw_distances = np.array(dtw_distances)
        # if writer:
        #     writer.add_scalar(f'test/dtw_mean', dtw_distances.mean(), epc)
        #     writer.add_scalar(f'test/dtw_std', dtw_distances.std(), epc)