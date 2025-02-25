import os
import time
import json

import torch
import numpy as np
import horovod.torch as hvd
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
        self.tester = RanknetTester(dataset, testset, config, logger)
        self.save_path = os.path.join(self.config['train']['save_dir'],
                                      f'ranknet_{self.config["train"]["exp"]}_best.pth')

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
        rank_criterion = nn.HuberLoss().to(self.device)  # FocalLoss(alpha=self.config['train']['focal_alpha'], gamma=self.config['train']['focal_gamma']).to(self.device)
        aux_criterion = nn.CrossEntropyLoss().to(self.device)

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
            model.train()

            if self.config['train']['distributed']['multi_gpu']:
                train_sampler.set_epoch(epc)
                self.logger.info(f'[gpu {rank}]train sampler set epoch {epc}')

            for i, (img1, feature1, bio, label, aux_label) in tqdm(enumerate(train_loader), desc=f'Training Epoch {epc}'):
                img1 = img1.to(self.device)
                feature1 = feature1.to(self.device)
                label = label.to(self.device)
                bio = bio.to(self.device)
                aux_label = aux_label.to(self.device)

                optimizer.zero_grad()
                o, a_o, d = model(img1, feature1, bio)
                o = o.squeeze()

                ranknet_loss = rank_criterion(o, label)
                aux_loss = aux_criterion(a_o, aux_label)
                if self.mode != 'feature':
                    ae_loss = ae_criterion(d, img1.view(-1, *img1.shape[2:]))
                    ae_loss = 0
                    loss = ranknet_loss + ae_loss * self.config['train']['ae_loss_weight'] + aux_loss * self.config['train']['aux_loss_weight']
                else:
                    loss = ranknet_loss + aux_loss * self.config['train']['aux_loss_weight']

                loss.backward()
                optimizer.step()
                acc = metric(o, label, infer_type='regression')
                aux_acc, _ = metric(a_o, aux_label, infer_type='classification')

                if writer:
                    writer.add_scalar(f'train/ranknet_loss', ranknet_loss.item(), epc * len_train_loader + i)
                    writer.add_scalar(f'train/aux_loss', aux_loss.item(), epc * len_train_loader + i)
                    # if self.mode != 'feature':
                    #     writer.add_scalar(f'train/ae_loss', ae_loss.item(), epc * len_train_loader + i)
                    writer.add_scalar(f'train/r2_score', acc, epc * len_train_loader + i)
                    writer.add_scalar(f'train/aux_accuracy', aux_acc, epc * len_train_loader + i)
                    writer.add_scalar(f'train/loss', loss.item(), epc * len_train_loader + i)
                losses += loss.item()

            scheduler.step()
            self.logger.info(f'[gpu:{rank}]epoch {epc} avg. loss {losses / len_train_loader:.4f} ')

            # write output image to tensorboard
            # if writer:
            #     if d is not None:
            #         out_for_saving = d.view(int(d.shape[0] / self.window_size), self.window_size, *d.shape[1:])[-1]
            #         writer.add_images(f'train/epc_{epc}_output', out_for_saving, epc, dataformats='NCHW')

            model.eval()
            with torch.no_grad():
                accs = 0
                d1, d2 = None, None
                for i, (img1, feature1, bio, label, aux_label) in tqdm(enumerate(val_loader), desc=f'Evaluation'):
                    img1 = img1.to(self.device)
                    feature1 = feature1.to(self.device)
                    label = label.to(self.device)
                    bio = bio.to(self.device)
                    aux_label = aux_label.to(self.device)

                    o1, a_o, d1 = model(img1, feature1, bio)
                    o1 = o1.squeeze()

                    acc = metric(o1, label, infer_type='regression')
                    aux_acc, _ = metric(a_o, aux_label, infer_type='classification')
                    accs += acc
                    if writer:
                        writer.add_scalar(f'val/r2_score', acc, epc * len_val_loader + i)
                        writer.add_scalar(f'val/aux_accuracy', aux_acc, epc * len_val_loader + i)

                avg_acc = accs / len_val_loader
                if writer:
                    # if d1 is not None:
                    #     out_for_saving1 = d1.view(int(d1.shape[0] / self.window_size), self.window_size, *d1.shape[1:])[-1]
                    #     writer.add_images(f'val/epc_{epc}_output_1', out_for_saving1, epc, dataformats='NCHW')

                    self.logger.info(f'[gpu:{rank}]epoch {epc} avg. val acc {avg_acc:.4f}')

            # model save if validation accuracy is the best
            if rank == 0:
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    torch.save(
                        model.state_dict(),
                        self.save_path
                    )
        if rank == 0:
            self.tester.test(writer, self.save_path)
        if writer is not None:
            writer.close()
