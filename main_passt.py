import os
import sys
from pathlib import Path
import numpy as np
from sklearn import metrics
import pickle

import torch
from torch.nn import functional as F
from einops import repeat, rearrange
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from helpers.helpers import mixstyle, mixup
from helpers.ramp import exp_warmup_linear_down, cosine_cycle
from helpers.workersinit import worker_init_fn
from models.passt import get_model, PaSST
from models.preprocess import AugmentMelSTFT
from datasets.dataset import get_training_set, get_test_set

# RUN_NAME = input('Please enter name of run: ')
RUN_NAME = 'passt_dir_mixstyle'

def get_scheduler_lambda(warm_up_len=3, ramp_down_start=3, ramp_down_len=10, last_lr_value=0.01, nr_of_epochs=25,
                         schedule_mode="exp_lin"):
    if schedule_mode == "exp_lin":
        return exp_warmup_linear_down(warm_up_len, ramp_down_len, ramp_down_start, last_lr_value)
    if schedule_mode == "cos_cyc":
        return cosine_cycle(warm_up_len, ramp_down_start, last_lr_value)
    if schedule_mode == "exp_down":
        return exp_rampdown(ramp_down_len, nr_of_epochs)
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown for a lambda funtion.")


def get_lr_scheduler(optimizer, schedule_mode="exp_lin"):
    if schedule_mode in {"exp_lin", "cos_cyc", "exp_down"}:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            get_scheduler_lambda(schedule_mode=schedule_mode)
        )

    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")


def get_optimizer(params, lr=0.00001, adamw=True, weight_decay=0.001):
    if adamw:
        print(f"\nUsing adamw weight_decay={weight_decay}!\n")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr)


class M(pl.LightningModule):
    def __init__(self, ):
        super(M, self).__init__()

        self.mel = AugmentMelSTFT(
            n_mels=128,
            sr=32000,
            win_length=800,
            hopsize=320,
            n_fft=1024,
            freqm=48,
            timem=20,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=1,
            fmax_aug_range=1000
        )

        self.net = get_model(arch="passt_s_swa_p16_128_ap476",
                             n_classes=10,
                             input_fdim=128,
                             s_patchout_t=0,
                             s_patchout_f=6)

        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        self.mixup_alpha = 0.0
        self.mixstyle_p = 0.4
        self.mixstyle_alpha = 0.4
        self.calc_device_info = True
        self.epoch = 0

    def forward(self, x):
        return self.net(x)

    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, files, y = batch
        # x, files, y, device_indices, cities, indices = batch

        if self.mel:
            x = self.mel_forward(x)

        batch_size = len(y)

        if self.mixstyle_p > 0:
            x = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha)

            y_hat, embed = self.forward(x)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        elif self.mixup_alpha:
            rn_indices, lam = my_mixup(batch_size, self.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(batch_size, 1, 1, 1) + x[rn_indices] * (1. - lam.reshape(batch_size, 1, 1, 1))

            y_hat, embed = self.forward(x)

            samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(batch_size) +
                            F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(batch_size)))
        else:
            y_hat, embed = self.forward(x)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        loss = samples_loss.mean()
        samples_loss = samples_loss.detach()

        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred = (preds == y).sum()
        results = {"loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}

        if self.calc_device_info:
            devices = [d.rsplit("-", 1)[1][:-4] for d in files]

            for d in self.device_ids:
                results["devloss." + d] = torch.as_tensor(0., device=self.device)
                results["devcnt." + d] = torch.as_tensor(0., device=self.device)

            for i, d in enumerate(devices):
                results["devloss." + d] = results["devloss." + d] + samples_loss[i]
                results["devcnt." + d] = results["devcnt." + d] + 1.

        return results

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        train_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'train.loss': avg_loss, 'train_acc': train_acc, 'step': self.current_epoch}

        if self.calc_device_info:
            for d in self.device_ids:
                dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
                dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
                logs["tloss." + d] = dev_loss / dev_cnt
                logs["tcnt." + d] = dev_cnt

        self.log_dict(logs)

        print(f"Training Loss: {avg_loss}")
        print(f"Training Accuracy: {train_acc}")


    def validation_step(self, batch, batch_idx):
        x, files, y = batch

        if self.mel:
            x = self.mel_forward(x)

        y_hat, embed = self.forward(x)

        samples_loss = F.cross_entropy(y_hat, y, reduction="none")
        loss = samples_loss.mean()

        self.log("validation.loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == y)
        n_correct_pred = n_correct_pred_per_sample.sum()
        results = {"val_loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}

        if self.calc_device_info:
            devices = [d.rsplit("-", 1)[1][:-4] for d in files]
            for d in self.device_ids:
                results["devloss." + d] = torch.as_tensor(0., device=self.device)
                results["devcnt." + d] = torch.as_tensor(0., device=self.device)
                results["devn_correct." + d] = torch.as_tensor(0., device=self.device)

            for i, d in enumerate(devices):
                results["devloss." + d] = results["devloss." + d] + samples_loss[i]
                results["devn_correct." + d] = results["devn_correct." + d] + n_correct_pred_per_sample[i]
                results["devcnt." + d] = results["devcnt." + d] + 1
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        val_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'val.loss': avg_loss, 'val_acc': val_acc, 'step': self.current_epoch}

        if self.calc_device_info:
            for d in self.device_ids:
                dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
                dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
                dev_corrct = torch.stack([x["devn_correct." + d] for x in outputs]).sum()
                logs["vloss." + d] = dev_loss / dev_cnt
                logs["vacc." + d] = dev_corrct / dev_cnt
                logs["vcnt." + d] = dev_cnt
                # device groups
                logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
                logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
                logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

            for d in set(self.device_groups.values()):
                logs["acc." + d] = logs["acc." + d] / logs["count." + d]
                logs["lloss.False" + d] = logs["lloss." + d] / logs["count." + d]

        self.log_dict(logs)

        if self.epoch > 0:
            print()
            print(f"Validation Loss: {avg_loss}")
            print(f"Validation Accuracy: {val_acc}")

        self.epoch += 1


    # the test functionality is exclusively used to store predictions on all samples of the development set
    def test_step(self, batch, batch_idx):
        (x, files, y), indices = batch
        # x, files, y, device_indices, cities, indices = batch
        if self.stored_predictions is not None:
            y_hat = self.stored_predictions[indices].to(y.device)
        else:
            if self.mel:
                x = self.mel_forward(x)

            y_hat, embed = self.forward(x)

        samples_loss = F.cross_entropy(y_hat, y, reduction="none")
        loss = samples_loss.mean()

        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == y)
        n_correct_pred = n_correct_pred_per_sample.sum()
        results = {"val_loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y),
                   "logits": y_hat, "sample_indices": indices}
        return results

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'val.loss': avg_loss, 'val_acc': val_acc}
        self.log_dict(logs)

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, f = batch
        if self.mel:
            x = self.mel_forward(x)
        y_hat, _ = self.forward(x)
        return f, y_hat

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = get_optimizer(self.parameters())

        return {
            'optimizer': optimizer,
            'lr_scheduler': get_lr_scheduler(optimizer)
        }

    def configure_callbacks(self):
        return get_extra_checkpoint_callback(save_last_n=1)


def get_extra_checkpoint_callback(save_last_n=None, checkpoint_dir_path='checkpoints/passt/val_acc'):
    if save_last_n is None:
        return []

    return [ModelCheckpoint(monitor="val_acc", save_top_k=1, mode='max',
                            dirpath=checkpoint_dir_path, filename=RUN_NAME)]


def main():
    trainer = pl.Trainer(max_epochs=25, gpus=1, weights_summary='full', benchmark=True, precision=16)

    train_set = get_training_set(apply_dir=True, prob_dir=0.6, sr=32000, identifier='resample32000',
                                 cache_root_path='/share/rk6/shared/kofta_cached_datasets/')

    train_loader = DataLoader(dataset=train_set, batch_size=80, num_workers=16, shuffle=True,
                              worker_init_fn=worker_init_fn)

    val_set = get_test_set(sr=32000, identifier='resample32000',
                           cache_root_path='/share/rk6/shared/kofta_cached_datasets/')

    val_loader = DataLoader(dataset=val_set, batch_size=20, num_workers=16,
                            worker_init_fn=worker_init_fn)

    modul = M()

    trainer.fit(
        modul,
        train_dataloader=train_loader,
        val_dataloaders=val_loader
    )


main()