

"""Computer vision example on Transfer Learning.

This computer vision example illustrates how one could fine-tune a pre-trained
network (by default, a ResNet50 is used) using pytorch-lightning. For the sake
of this example, the 'cats and dogs dataset' (~60MB, see `DATA_URL` below) and
the proposed network (denoted by `TransferLearningModel`, see below) is
trained for 15 epochs. The training consists in three stages. From epoch 0 to
4, the feature extractor (the pre-trained network) is frozen except maybe for
the BatchNorm layers (depending on whether `train_bn = True`). The BatchNorm
layers (if `train_bn = True`) and the parameters of the classifier are trained
as a single parameters group with lr = 1e-2. From epoch 5 to 9, the last two
layer groups of the pre-trained network are unfrozen and added to the
optimizer as a new parameter group with lr = 1e-4 (while lr = 1e-3 for the
first parameter group in the optimizer). Eventually, from epoch 10, all the
remaining layer groups of the pre-trained network are unfrozen and added to
the optimizer as a third parameter group. From epoch 10, the parameters of the
pre-trained network are trained with lr = 1e-5 while those of the classifier
are trained with lr = 1e-4.
Note:
    See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import argparse
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Generator, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)

#  --- Utility functions ---


def _make_trainable(module: torch.nn.Module) -> None:
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: torch.nn.Module,
                      train_bn: bool = True) -> None:
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module: torch.nn.Module,
           n: Optional[int] = None,
           train_bn: bool = True) -> None:
    """Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)



def filter_params(module: torch.nn.Module,
                  train_bn: bool = True) -> Generator:
    """Yields the trainable parameters of a given module.
    Args:
        module: A given module
        train_bn: If True, leave the BatchNorm layers in training mode
    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, train_bn=train_bn):
                yield param


def _unfreeze_and_add_param_group(module: torch.nn.Module,
                                  optimizer: Optimizer,
                                  lr: Optional[float] = None,
                                  train_bn: bool = True):
    """Unfreezes a module and adds its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    optimizer.add_param_group(
        {'params': filter_params(module=module, train_bn=train_bn),
         'lr': params_lr / 10.,
         })


#  --- Pytorch-lightning module ---


class TransferLearningModel(pl.LightningModule):
    """Transfer Learning with pre-trained ResNet50.
    Args:
        hparams: Model hyperparameters
        dl_path: Path where the data will be downloaded
    """
    def __init__(self,
                 hparams: argparse.Namespace,
                 dl_path: Union[str, Path]) -> None:
        super().__init__()
        self.hparams = hparams
        self.dl_path = dl_path
        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        model_func = getattr(models, self.hparams.backbone)
        backbone = model_func(pretrained=True)

        _layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*_layers)
        freeze(module=self.feature_extractor, train_bn=self.hparams.train_bn)

        # 2. Classifier:
        _fc_layers = [torch.nn.Linear(2048, 256),
                      torch.nn.Linear(256, 32),
                      torch.nn.Linear(32, 1)]
        self.fc = torch.nn.Sequential(*_fc_layers)

        # 3. Loss:
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x):
        """Forward pass. Returns logits."""

        # 1. Feature extraction:
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)

        # 2. Classifier (returns logits):
        x = self.fc(x)

        return x

    def loss(self, labels, logits):
        return self.loss_func(input=logits, target=labels)

    def train(self, mode=True):
        super().train(mode=mode)

        epoch = self.current_epoch
        if epoch < self.hparams.milestones[0] and mode:
            # feature extractor is frozen (except for BatchNorm layers)
            freeze(module=self.feature_extractor,
                   train_bn=self.hparams.train_bn)

        elif self.hparams.milestones[0] <= epoch < self.hparams.milestones[1] and mode:
            # Unfreeze last two layers of the feature extractor
            freeze(module=self.feature_extractor,
                   n=-2,
                   train_bn=self.hparams.train_bn)

    def on_epoch_start(self):
        """Use `on_epoch_start` to unfreeze layers progressively."""
        optimizer = self.trainer.optimizers[0]
        if self.current_epoch == self.hparams.milestones[0]:
            _unfreeze_and_add_param_group(module=self.feature_extractor[-2:],
                                          optimizer=optimizer,
                                          train_bn=self.hparams.train_bn)

        elif self.current_epoch == self.hparams.milestones[1]:
            _unfreeze_and_add_param_group(module=self.feature_extractor[:-2],
                                          optimizer=optimizer,
                                          train_bn=self.hparams.train_bn)

    def training_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)
        y_bin = torch.ge(y_logits, 0)

        # 2. Compute loss & accuracy:
        train_loss = self.loss(y_true, y_logits)
        num_correct = torch.eq(y_bin.view(-1), y_true.view(-1)).sum()

        # 3. Outputs:
        tqdm_dict = {'train_loss': train_loss}
        output = OrderedDict({'loss': train_loss,
                              'num_correct': num_correct,
                              'log': tqdm_dict,
                              'progress_bar': tqdm_dict})

        return output

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        train_loss_mean = torch.stack([output['loss']
                                       for output in outputs]).mean()
        train_acc_mean = torch.stack([output['num_correct']
                                      for output in outputs]).sum().float()
        train_acc_mean /= (len(outputs) * self.hparams.batch_size)
        return {'log': {'train_loss': train_loss_mean,
                        'train_acc': train_acc_mean,
                        'step': self.current_epoch}}

    def validation_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)
        y_bin = torch.ge(y_logits, 0)

        # 2. Compute loss & accuracy:
        val_loss = self.loss(y_true, y_logits)
        num_correct = torch.eq(y_bin.view(-1), y_true.view(-1)).sum()

        return {'val_loss': val_loss,
                'num_correct': num_correct}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()
        val_acc_mean = torch.stack([output['num_correct']
                                    for output in outputs]).sum().float()
        val_acc_mean /= (len(outputs) * self.hparams.batch_size)
        return {'log': {'val_loss': val_loss_mean,
                        'val_acc': val_acc_mean,
                        'step': self.current_epoch}}

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      self.parameters()),
                               lr=self.hparams.lr)

        scheduler = MultiStepLR(optimizer,
                                milestones=self.hparams.milestones,
                                gamma=self.hparams.lr_scheduler_gamma)

        return [optimizer], [scheduler]
