import os

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm

from network.loss import OrdinalCrossEntropyLoss
from executor.trainer import BaseTrainer
from utils.utils import metric


class RanknetTrainer(BaseTrainer):
    def __init__(self, train_dataset, test_dataset, config, logger, accelerator):
        super().__init__(train_dataset, test_dataset, config, logger, accelerator)

        if self.config['train']['fine_tune']:
            model_path = os.path.join(self.config['train']['save_dir'], self.config['experiment']['model'])
            with open(model_path, "rb") as f:
                self.model.load_state_dict(torch.load(f, map_location='cpu'))
            self.logger.info(f'Model loaded from {model_path} for fine-tuning')

        self.rank_criterion = OrdinalCrossEntropyLoss(self.config['train']['cutpoints'])
        self.aux_criterion = nn.CrossEntropyLoss()

        self._setup_optimizer_and_scheduler()
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
        self.len_train_loader = len(self.train_loader)
        self.len_val_loader = len(self.val_loader)

    def train(self):
        writer = self._setup_writer()

        self.logger.info(f'build model on device: {self.accelerator.device}')
        best_acc = 0

        # (batch, pair, sequence, channel, height, width)
        for epc in range(self.config['train']['epoch']):
            losses = 0
            cm = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.model.train()
            l0_cnt, l1_cnt, l2_cnt = 0, 0, 0

            iterator = self._set_epoch(epc)
            for i, (img1, feature1, img2, feature2, bio, label, aux_label) in iterator:
                l0_cnt += len(label[label == 0])
                l1_cnt += len(label[label == 1])
                l2_cnt += len(label[label == 2])

                img_cat = torch.cat([img1, img2], dim=0)
                feature_cat = torch.cat([feature1, feature2], dim=0)
                bio_cat = torch.cat([bio, bio], dim=0)

                self.optimizer.zero_grad()

                o_cat, a_o_cat = self.model(img_cat, feature_cat, bio_cat)
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

                acc, cm_tmp = metric(o.detach(), label.detach(), self.rank_criterion.get_cutpoints())
                aux_acc1, _ = metric(a_o1, aux_label, infer_type='classification')
                aux_acc2, _ = metric(a_o2, aux_label, infer_type='classification')
                cm += cm_tmp

                if writer:
                    writer.add_scalar(f'train/ranknet_loss', ranknet_loss.item(), epc * self.len_train_loader + i)
                    writer.add_scalar(f'train/aux_loss', aux_loss.item(), epc * self.len_train_loader + i)
                    writer.add_scalar(f'train/accuracy', acc, epc * self.len_train_loader + i)
                    writer.add_scalar(f'train/aux_accuracy_1', aux_acc1, epc * self.len_train_loader + i)
                    writer.add_scalar(f'train/aux_accuracy_2', aux_acc2, epc * self.len_train_loader + i)
                    writer.add_scalar(f'train/loss', loss.item(), epc * self.len_train_loader + i)
                losses += loss.item()

            self.scheduler.step()

            log_msg = (f"[Device {self.accelerator.process_index}] Epoch {epc} | "
                       f"Avg Loss: {losses / self.len_train_loader:.4f} | "
                       f"L0: {l0_cnt} / L1: {l1_cnt} / L2: {l2_cnt} | "
                       f"Cutpoints: {self.config['train']['cutpoints']}")
            self.logger.info(log_msg, main_process_only=False)
            self._write_confusion_mat(writer, cm, epc, 'train')

            self.model.eval()
            accs = 0
            cm = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

            with torch.no_grad():
                iterator = tqdm(
                    enumerate(self.val_loader),
                    desc=f'Evaluation',
                    disable=not self.accelerator.is_main_process,
                    total=self.len_val_loader
                )
                for i, (img1, feature1, img2, feature2, bio, label, aux_label) in iterator:
                    img_cat = torch.cat([img1, img2], dim=0)
                    feature_cat = torch.cat([feature1, feature2], dim=0)
                    bio_cat = torch.cat([bio, bio], dim=0)
                    o_cat, a_o_cat = self.model(img_cat, feature_cat, bio_cat)
                    o1, o2 = torch.chunk(o_cat, 2, dim=0)
                    a_o1, a_o2 = torch.chunk(a_o_cat, 2, dim=0)
                    o = o2 - o1

                    acc, cm_tmp = metric(o, label, self.rank_criterion.get_cutpoints())
                    aux_acc1, _ = metric(a_o1, aux_label, infer_type='classification')
                    aux_acc2, _ = metric(a_o2, aux_label, infer_type='classification')

                    cm += cm_tmp
                    accs += acc
                    if writer:
                        writer.add_scalar(f'val/accuracy', acc, epc * self.len_val_loader + i)
                        writer.add_scalar(f'val/aux_accuracy_1', aux_acc1, epc * self.len_val_loader + i)
                        writer.add_scalar(f'val/aux_accuracy_2', aux_acc2, epc * self.len_val_loader + i)

                avg_acc = accs / self.len_val_loader
                cm_str = np.array_str(cm).replace('\n', '')
                val_log_msg = (f"[Device {self.accelerator.process_index}] Epoch {epc} | "
                               f"Avg Accuracy: {avg_acc:.4f} | "
                               f"Confusion Matrix [Dec/Same/Inc]: {cm_str}")
                self.logger.info(val_log_msg, main_process_only=False)
                self._write_confusion_mat(writer, cm, epc, 'val')

            best_acc = self._save_model(avg_acc, best_acc)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.tester.test(writer, self.save_path)

        if writer:
            writer.close()

    def _write_confusion_mat(self, writer, cm, epc, phase):
        if writer:
            cm = pd.DataFrame(cm, index=['dec', 'same', 'inc'], columns=['dec', 'same', 'inc'])
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, cmap='Blues')
            writer.add_figure(f'{phase}/confusion_matrix epc_{epc}', plt.gcf())
            plt.close()
