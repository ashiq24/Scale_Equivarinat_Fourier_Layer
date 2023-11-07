"""Implements the core support for a classifier class."""

import logging
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from timm.utils import accuracy


class AbstractBaseClassifierModel(pl.LightningModule):
    """Base classifier model for classification.

        Reusable code for clasifier models. To make new classifiers, just implement
        the initializer and the forward method
    """

    def __init__(self, optimizer=None, optimizer_kwargs={},
                 scheduler=None, scheduler_kwargs={},
                 param_scheduler=None, param_scheduler_kwargs={},
                 warmup_epochs=0):
        super().__init__()
        self.optimizer_fn = optimizer if optimizer is not None else torch.optim.AdamW
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_fn = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.param_scheduler_fn = param_scheduler
        self.param_scheduler_kwargs = param_scheduler_kwargs
        self.warmup_epochs = warmup_epochs
        self.criterion = nn.CrossEntropyLoss()

    def _forward_step(self, batch, batch_idx, stage='train', sync_dist=False):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = accuracy(logits, y)[0]/100
        self.log('%s_loss' % stage, loss, on_step=True,
                 on_epoch=True, logger=True, sync_dist=sync_dist, prog_bar=stage == 'val')
        self.log('%s_acc' % stage, acc, on_step=True,
                 on_epoch=True, logger=True, sync_dist=sync_dist, prog_bar=stage == 'val')
        return logits, loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        _, loss = self._forward_step(
            batch, batch_idx, stage='train', sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        _, loss = self._forward_step(
            batch, batch_idx, stage='val', sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        _, loss = self._forward_step(
            batch, batch_idx, stage='test', sync_dist=True)
        return loss

    def optimizer_step(self, epoch, batch_idx,
                       optimizer, optimizer_closure):
        """Called when the train epoch begins."""
        # Lr warmup
        if self.current_epoch < self.warmup_epochs:
            it_curr = self.trainer.num_training_batches*self.current_epoch+1+batch_idx
            it_max = self.trainer.num_training_batches*self.warmup_epochs
            lr_scale = float(it_curr) / it_max
            for pg in self.trainer.optimizers[0].param_groups:
                pg['lr'] = lr_scale * self.learning_rate

        # Update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        scheduler.step()

    def configure_optimizers(self):
        logging.info(
            f'Configuring optimizer: {self.optimizer_fn} with {self.optimizer_kwargs}')

        _lr = 0 if self.warmup_epochs != 0 else self.learning_rate
        weight_group = self.get_weight_group()
        optimizer = self.optimizer_fn(weight_group,
                                      lr=_lr, weight_decay=self.weight_decay,
                                      **self.optimizer_kwargs)
        if self.scheduler_fn is None:
            return optimizer

        if isinstance(self.scheduler_fn, list):
            schedulers = [sch_fn(optimizer, **sch_kwargs) for sch_fn,
                          sch_kwargs in zip(self.scheduler_fn, self.scheduler_kwargs)]
        else:
            schedulers = [
                {'scheduler': self.scheduler_fn(optimizer, **self.scheduler_kwargs),
                 'frequency': 1,
                 'name': 'main_lr_scheduler',
                 'interval': 'epoch'
                 }]

        return [optimizer], schedulers
