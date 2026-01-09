import torch
import torch.nn as nn
from tqdm import tqdm

from utils.utils import metric
from executor.trainer import BaseTrainer


class CardinalTrainer(BaseTrainer):
    def __init__(self, train_dataset, test_dataset, config, logger, accelerator):
        super().__init__(train_dataset, test_dataset, config, logger, accelerator)

        self.main_criterion = nn.HuberLoss()
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
        best_acc = 0 # R2 score

        for epc in range(self.config['train']['epoch']):
            losses = 0
            self.model.train()

            iterator = self._set_epoch(epc)
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
                acc = metric(o, label, infer_type='regression')
                aux_acc, _ = metric(a_o, aux_label, infer_type='classification')

                if writer:
                    writer.add_scalar(f'train/main_loss', main_loss.item(), epc * self.len_train_loader + i)
                    writer.add_scalar(f'train/aux_loss', aux_loss.item(), epc * self.len_train_loader + i)
                    writer.add_scalar(f'train/r2_score', acc, epc * self.len_train_loader + i)
                    writer.add_scalar(f'train/aux_accuracy', aux_acc, epc * self.len_train_loader + i)
                    writer.add_scalar(f'train/loss', loss.item(), epc * self.len_train_loader + i)
                losses += loss.item()

            self.scheduler.step()

            log_msg = (
                f"[Device {self.accelerator.process_index}] "
                f"Epoch {epc} | Avg Loss: {losses / self.len_train_loader:.4f}"
            )
            self.logger.info(log_msg, main_process_only=False)

            self.model.eval()
            with torch.no_grad():
                accs = 0
                val_aux_accs = 0

                iterator = tqdm(enumerate(self.val_loader),
                                desc=f'Evaluation',
                                disable=not self.accelerator.is_main_process,
                                total=self.len_val_loader)

                for i, (img1, feature1, bio, label, aux_label) in iterator:
                    o1, a_o, d1 = self.model(img1, feature1, bio)
                    o1 = o1.squeeze()

                    acc = metric(o1, label, infer_type='regression')
                    aux_acc, _ = metric(a_o, aux_label, infer_type='classification')

                    accs += acc
                    val_aux_accs += aux_acc

                    if writer:
                        writer.add_scalar(f'val/r2_score', acc, epc * self.len_val_loader + i)
                        writer.add_scalar(f'val/aux_accuracy', aux_acc, epc * self.len_val_loader + i)

                avg_acc = accs / self.len_val_loader
                avg_aux_acc = val_aux_accs / self.len_val_loader

                val_log_msg = (
                    f"[Device {self.accelerator.process_index}] "
                    f"Epoch {epc} Validation | R2 Score: {avg_acc:.4f} | Aux Acc: {avg_aux_acc:.4f}"
                )
                self.logger.info(val_log_msg, main_process_only=False)

            best_acc = self._save_model(avg_acc, best_acc)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.tester.test(writer, self.save_path)

        if writer is not None:
            writer.close()