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
            self.scheduler
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler
        )
        self.len_train_loader = len(self.train_loader)
        self.len_val_loader = len(self.val_loader)

    def train(self):
        device = self.accelerator.device
        writer = self._setup_writer()

        self.logger.info(f'build model on device: {self.accelerator.device}')
        best_acc = 0

        # (batch, pair, sequence, channel, height, width)
        for epc in range(self.config['train']['epoch']):
            losses = 0
            self.model.train()
            l0_cnt, l1_cnt, l2_cnt = 0, 0, 0
            train_outputs = []
            train_labels = []
            train_aux_labels = []
            train_aux_preds1 = []
            train_aux_preds2 = []

            iterator = self._set_epoch(epc)
            for i, (img1, feature1, img2, feature2, bio, label, aux_label) in iterator:
                img1, feature1, img2, feature2, bio, label, aux_label = (
                    img1.to(device), feature1.to(device), img2.to(device), feature2.to(device),
                    bio.to(device), label.to(device), aux_label.to(device))
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

                train_outputs.append(o.detach().cpu())
                train_labels.append(label.detach().cpu())
                train_aux_labels.append(aux_label.detach().cpu())
                train_aux_preds1.append(a_o1.detach().cpu())
                train_aux_preds2.append(a_o2.detach().cpu())

                if writer:
                    writer.add_scalar(f'Train/ranknet_loss', ranknet_loss.item(), epc * self.len_train_loader + i)
                    writer.add_scalar(f'Train/aux_loss', aux_loss.item(), epc * self.len_train_loader + i)
                    writer.add_scalar(f'Train/loss', loss.item(), epc * self.len_train_loader + i)
                losses += loss.item()

            self.scheduler.step()

            self._write_metrics(writer, train_outputs, train_labels, train_aux_labels,
                                train_aux_preds1, train_aux_preds2, epc, 'Train',
                                losses, l0_cnt, l1_cnt, l2_cnt)

            self.model.eval()

            val_outputs = []
            val_labels = []
            val_aux_labels = []
            val_aux_preds1 = []
            val_aux_preds2 = []

            with torch.no_grad():
                iterator = tqdm(
                    enumerate(self.val_loader),
                    desc=f'Evaluation',
                    disable=not self.accelerator.is_main_process,
                    total=self.len_val_loader
                )
                for i, (img1, feature1, img2, feature2, bio, label, aux_label) in iterator:
                    img1, feature1, img2, feature2, bio, label, aux_label = (
                        img1.to(device), feature1.to(device), img2.to(device), feature2.to(device),
                        bio.to(device), label.to(device), aux_label.to(device))
                    img_cat = torch.cat([img1, img2], dim=0)
                    feature_cat = torch.cat([feature1, feature2], dim=0)
                    bio_cat = torch.cat([bio, bio], dim=0)
                    o_cat, a_o_cat = self.model(img_cat, feature_cat, bio_cat)
                    o1, o2 = torch.chunk(o_cat, 2, dim=0)
                    a_o1, a_o2 = torch.chunk(a_o_cat, 2, dim=0)
                    o = o2 - o1

                    val_outputs.append(o.cpu())
                    val_labels.append(label.cpu())
                    val_aux_labels.append(aux_label.cpu())
                    val_aux_preds1.append(a_o1.cpu())
                    val_aux_preds2.append(a_o2.cpu())


                acc = self._write_metrics(
                    writer, val_outputs, val_labels, val_aux_labels,
                    val_aux_preds1, val_aux_preds2, epc, 'Val')

            best_acc = self._save_model(acc, best_acc)

        self.accelerator.wait_for_everyone()
        # if self.accelerator.is_main_process:
        torch.cuda.empty_cache()
        self.tester.test(writer, self.save_path, self.accelerator)

        if writer:
            writer.close()

    def _write_metrics(self, writer, output, label, aux_labels, aux_preds1, aux_preds2, epc, phase,
                       loss=0, l0_cnt=0, l1_cnt=0, l2_cnt=0):
        outputs_tc = torch.cat(output)
        labels_tc = torch.cat(label)
        aux_labels_tc = torch.cat(aux_labels)
        aux_preds1_tc = torch.cat(aux_preds1)
        aux_preds2_tc = torch.cat(aux_preds2)

        acc, cm = metric(outputs_tc, labels_tc, self.rank_criterion.get_cutpoints())
        tau, p_val = metric(outputs_tc, labels_tc, infer_type='kendal_tau')
        aux_acc1, _ = metric(aux_preds1_tc, aux_labels_tc, infer_type='classification')
        aux_acc2, _ = metric(aux_preds2_tc, aux_labels_tc, infer_type='classification')

        cm_str = np.array_str(cm).replace('\n', '')

        if phase == "Val":
            log_msg = (f"[Device {self.accelerator.process_index}] Epoch {epc} | "
                           f"Avg Accuracy (F1): {acc:.4f} | "
                           f"Confusion Matrix [Dec/Same/Inc]: {cm_str}")
        else:
            log_msg = (f"[Device {self.accelerator.process_index}] Epoch {epc} | "
                       f"Avg Loss: {loss / self.len_train_loader:.4f} | "
                       f"L0: {l0_cnt} / L1: {l1_cnt} / L2: {l2_cnt} | "
                       f"Cutpoints: {self.config['train']['cutpoints']}"
                       f"Confusion Matrix [Dec/Same/Inc]: {cm_str}")
        self.logger.info(log_msg, main_process_only=True)

        if writer:
            writer.add_scalar(f'{phase}/accuracy (F1)', acc, epc)
            writer.add_scalar(f'{phase}/kendal tau', tau, epc)
            writer.add_scalar(f'{phase}/kendal tau_p', p_val, epc)
            writer.add_scalar(f'{phase}/aux_accuracy_1', aux_acc1, epc)
            writer.add_scalar(f'{phase}/aux_accuracy_2', aux_acc2, epc)

            cm = pd.DataFrame(cm, index=['dec', 'same', 'inc'], columns=['dec', 'same', 'inc'])
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, cmap='Blues')
            writer.add_figure(f'{phase}/confusion_matrix epc_{epc}', plt.gcf())
            plt.close()

        return acc
