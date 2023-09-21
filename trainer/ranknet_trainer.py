import os
import time

import torch
import horovod.torch as hvd
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from dataloader.pair_loader import PairLoader
from network.ranknet import RankNet


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

        torch.set_float32_matmul_precision('high')
        if config['train']['distributed']['multi_gpu']:
            hvd.init()
            if torch.cuda.is_available():
                torch.cuda.set_device(hvd.local_rank())
                self.device = torch.device(f'cuda', hvd.local_rank())
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        train_sampler = None
        rank = 0
        if self.config['train']['distributed']['multi_gpu']:
            rank = hvd.rank()
            train_sampler = DistributedSampler(self.train_dataset, num_replicas=hvd.size(), rank=rank)
            writer = SummaryWriter(
                log_dir=os.path.join(self.config['train']['log_dir'],
                                     time.strftime('%Y-%m-%d-%H-%M-%S')
                                     )
            ) if rank == 0 else None
        else:
            writer = SummaryWriter(
                log_dir=os.path.join(self.config['train']['log_dir'],
                                     time.strftime('%Y-%m-%d-%H-%M-%S')
                                     )
            )

        # (batch, pair, sequence, channel, height, width)
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.config['train']['batch_size'],
                                  sampler=train_sampler,
                                  num_workers=self.config['train']['num_workers'],
                                  shuffle=(train_sampler is None),
                                  pin_memory=True)

        self.logger.debug(f'build model gpu: {rank}')
        model = RankNet(self.config)
        model.to(self.device)
        ae_criterion = nn.MSELoss().to(self.device)
        rank_criterion = nn.BCELoss().to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['train']['lr'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config['train']['schedule'], gamma=0.1)
        compiled_model = torch.compile(model)
        if self.config['train']['distributed']['multi_gpu']:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(compiled_model.state_dict(), root_rank=0)

        for epc in range(self.config['train']['epoch']):
            losses = 0
            if self.config['train']['distributed']['multi_gpu']:
                train_sampler.set_epoch(epc)
                self.logger.debug(f'[gpu {rank}]train sampler set epoch {epc}')

            cnt = 0
            out_for_saving1 = None
            out_for_saving2 = None
            for i, (img1, feature1, img2, feature2, label) in tqdm(enumerate(train_loader), desc=f'Training Epoch {epc}'):
                img1 = img1.to(self.device)
                feature1 = feature1.to(self.device)
                img2 = img2.to(self.device)
                feature2 = feature2.to(self.device)
                label = label.unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                o, d1, d2 = compiled_model(img1, feature1, img2, feature2)
                ranknet_loss = rank_criterion(o, label)
                ae_loss = ae_criterion(d1, img1.view(-1, *img1.shape[2:])) + ae_criterion(d2, img2.view(-1, *img2.shape[2:]))
                loss = ranknet_loss + ae_loss
                loss.backward()
                optimizer.step()

                if writer:
                    writer.add_scalar(f'ranknet_loss', ranknet_loss.item(), epc * len(train_loader) + i)
                    writer.add_scalar(f'ae_loss', ae_loss.item(), epc * len(train_loader) + i)
                losses += loss.item()
                cnt += 1
                out_for_saving1 = d1.view(int(d1.shape[0]/4), 4, *d1.shape[1:])[-1]
                out_for_saving2 = d2.view(int(d2.shape[0]/4), 4, *d2.shape[1:])[-1]

            self.logger.debug(f'[gpu:{rank}]epoch {epc} avg. loss {losses / cnt:.4f}')

            # write output image to tensorboard
            if writer:
                writer.add_images(f'output_{rank} epc_{epc}', out_for_saving1, epc, dataformats='NCHW')
                writer.add_images(f'output_{rank} epc_{epc}', out_for_saving2, epc, dataformats='NCHW')
            scheduler.step()

            # with torch.no_grad():
            #     model.eval()
            #     output = None
            #     for img_data, _, y in tqdm(val_loader, desc=f'Evaluation'):
            #         img_data = img_data[1].to(self.device)
            #
            #         _, output = model(img_data)
            #         loss = criterion(output, img_data)
            #         print(f'val loss {loss.item():.4f}')
            #     model.train()
        writer.close()
