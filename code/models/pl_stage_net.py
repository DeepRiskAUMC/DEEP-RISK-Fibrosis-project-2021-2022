import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import matplotlib.pyplot as plt
import io
import numpy as np
import itertools
import torchvision
import matplotlib.cm as cm
import math
from torchvision.transforms import transforms
import cv2
from argparse import Namespace

from models import get_model

DEBUG = False

class MyDice(torchmetrics.Metric):
    def __init__(self, threshold=0.5, dist_sync_on_step=False, smoothing=0):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold
        self.smoothing = smoothing
        self.add_state('sum_dice_scores', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, f"shapes {preds.shape} and {target.shape} don't match"
        preds = preds > self.threshold
        target = target > 0.5
        # calculate dice separately for each image in batch
        intersection = (preds * target).sum(dim=(1, 2, 3))
        dice = (2 * intersection +  self.smoothing) / (preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smoothing)

        self.sum_dice_scores += dice.sum()
        self.total += dice.numel()
        #print(f"{self.threshold=},{self.sum_dice_scores=}.{self.total=} , {preds.sum()=}")

    def compute(self):
        return self.sum_dice_scores / self.total



class SingleStageBase(pl.LightningModule):
    """Model superclass that handles:
            - loss
            - optimizers
            - schedulers
            - train, validation and test steps
            - logging
            - forward
            - cam processing
        Such that subclass models only need:
            - forward_features (i.e. all layers except final global pooling)

        Args:
            lr (float): learning rate. Default: 1-e3.
            weight_decay (float): AdamW weight decay. Default: 0., i.e. normal Adam.
            schedule (str): Learning rate scheduler to use. Default None. Options: None, step, cosine
            max_iters (int): Maximal number of iterations, which is necessary for cosine lr scheduling. Default: -1 / needs to be set for cosine schedule.
            warmup_iters (int): Number of iterations to use for linear learning rate warmup. Default: 0 / No warmup.

    """
    def __init__(self, lr=1e-3, weight_decay=0,
                schedule=None, max_iters=-1, warmup_iters=0,
                num_classes=2, heavy_log=10,
                mask_loss_bce=1.0, pretrain=5,
                backbone="drnd", pre_weights_path=None,
                myo_mask_dilation=0, cheat_dice=0,
                focal_lambda=0.01, sg_psi=0.3,
                aspp=False,
                **kwargs):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.schedule = schedule
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        assert num_classes > 1
        self.num_classes = num_classes
        self.save_hyperparameters()
        self.heavy_log = heavy_log
        self.cheat_dice = cheat_dice
        self.myo_mask_dilation = myo_mask_dilation
        self.MASK_LOSS_BCE = mask_loss_bce
        self.PRETRAIN = pretrain
        self.focal_lambda = focal_lambda
        self.sg_psi = sg_psi
        self.aspp = aspp

        # model
        """ enc_cfg = {"MODEL" : "ae",
                   "BACKBONE" : model,
                   "PRE_WEIGHTS_PATH" : None,
                   "MASK_LOSS_BCE" : self.MASK_LOSS_BCE,
                   "FOCAL_P" : 3,
                   "FOCAL_LAMBDA" : 0.01,
                   "PAMR_KERNEL" : [1, 2, 4, 8, 12, 24],
                   "PAMR_ITER" : 10,
                   "SG_PSI" : 0.3}"""
        enc_cfg = Namespace()
        enc_cfg.MODEL = "ae"
        enc_cfg.BACKBONE = backbone
        enc_cfg.PRE_WEIGHTS_PATH = pre_weights_path
        enc_cfg.MASK_LOSS_BCE = self.MASK_LOSS_BCE
        enc_cfg.FOCAL_P = 3
        enc_cfg.FOCAL_LAMBDA = self.focal_lambda
        enc_cfg.PAMR_KERNEL = [1, 2, 4, 8, 12, 24]
        enc_cfg.PAMR_ITER = 10
        enc_cfg.SG_PSI = self.sg_psi

        self.enc = get_model(enc_cfg, num_classes=self.num_classes, **kwargs)



        self.output2pred = nn.Sigmoid()

        def _make_metric_dict(metric, splits=["Train", "Val", "Test"], **kwargs):
            return nn.ModuleDict({split : metric(**kwargs) for split in splits })

        #self.accuracy = nn.ModuleDict({
        #    "Train" : torchmetrics.Accuracy(threshold=0.5),
        #    "Val" : torchmetrics.Accuracy(threshold=0.5),
        #    "Test" : torchmetrics.Accuracy(threshold=0.5)
        #})

        #self.confusion = nn.ModuleDict({
        #    "Train" : torchmetrics.ConfusionMatrix(threshold=0.5),
        #    "Val" : torchmetrics.Accuracy(threshold=0.5),
        #    "Test" : torchmetrics.Accuracy(threshold=0.5)
        #})
        self.accuracy = _make_metric_dict(torchmetrics.Accuracy, threshold=0.5)
        self.confusion = _make_metric_dict(torchmetrics.ConfusionMatrix, num_classes=self.num_classes, threshold=0.5)
        self.auroc = _make_metric_dict(torchmetrics.AUROC)
        self.pr_curve = _make_metric_dict(torchmetrics.PrecisionRecallCurve)
        self.roc = _make_metric_dict(torchmetrics.ROC)
        self.auc = _make_metric_dict(torchmetrics.AUC)
        self.dice = _make_metric_dict(MyDice, threshold=0.5)
        self.dice2 = _make_metric_dict(MyDice, threshold=0.8)
        self.dice3 = _make_metric_dict(MyDice, threshold=0.2)
        self.dice_pseudo_gt = _make_metric_dict(MyDice, threshold=0.5)

        self.example_batch = {"Train" : None, "Val" : None , "Test": None}



    #def forward(self, *args, **kwargs):
    #    raise NotImplementedError




    def configure_optimizers(self):

        # TODO: optimizer using different LR, not necessary because training from scratch?
        #enc_params = self.enc.parameter_groups(self.lr, self.weight_decay)
        enc_params = self.enc.parameters()

        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(enc_params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(enc_params, lr=self.lr)

        if self.warmup_iters == 0 and self.schedule == None:
            return optimizer

        schedulers_list = []
        if self.warmup_iters != 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=self.warmup_iters)
            schedulers_list.append(warmup_scheduler)

        if self.schedule == 'step':
            schedulers_list.append(torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1))
        elif self.schedule == 'cosine':
            schedulers_list.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_iters - self.warmup_iters))

        if self.warmup_iters != 0 and self.schedule != None:
            milestones = [self.warmup_iters]
        else:
            milestones = []

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers_list, milestones=milestones)
        scheduler.optimizer = optimizer # dirty fix for bug in SequentialLR/ Lightning interaction

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }





    def multilabel_soft_margin_loss(self, outputs, labels):
        #TODO:  is it necessary to set zero values to -1?
        #labels[labels == 0] = -1
        return F.multilabel_soft_margin_loss(outputs, labels)


    def _myo_seg_to_mask(self, x, dilation=0, size="cam"):
        assert size in ["cam", "image"]
        if dilation != 0:
            x = F.max_pool2d(x, kernel_size=dilation, stride=1, padding=dilation//2)

        if size == "cam":
            x = F.interpolate(x, scale_factor=1/self.downsampling,mode='area')
        elif size == "image":
            pass
        else:
            raise NotImplementedError

        x = (x > 0.5).float()
        return x


    def _update_dice(self, torchmetric_dice, mask, fibrosis_seg_label, myo_seg, cheat=False):
        # select only masks that have a non empty fibrosis label
        labeled =  (fibrosis_seg_label.amin(dim=(1, 2, 3), keepdim=False) != -1) * (fibrosis_seg_label.sum(dim=(1, 2, 3), keepdim=False) != 0)
        if labeled.sum() == 0:
            return
        else:
            mask = mask[labeled]
            fibrosis_seg_label = fibrosis_seg_label[labeled]

            if cheat == True:
                myo_mask = self.myo_seg_to_mask(myo_seg, dilation=self.myo_mask_dilation, size="image")
                mask = fibrosis_seg_label * myo_mask
            else:
                mask = mask
            torchmetric_dice(mask, fibrosis_seg_label)

        return




    def _step(self, batch, split, logging=True):
        assert split in ["Train", "Val", "Test"]
        if split == "Train":
            self.enc.train()
        else:
            self.enc.eval()

        epoch = self.current_epoch
        image = batch['img']
        gt_labels = batch['label']
        myo_seg = batch['myo_seg']
        fibrosis_seg_label = batch['fibrosis_seg_label']


        PRETRAIN = epoch < (11 if DEBUG else self.PRETRAIN)

        # denorm images TODO: is this necessary
        #image_raw = self.denorm(image.clone())
        image_raw = image.clone()

        # classification
        gt_labels = gt_labels.unsqueeze(-1)
        cls_out, cls_fg, masks, mask_logits, pseudo_gt, loss_mask = self.enc(image, image_raw, gt_labels)
        #print(f"{cls_out=}")
        # classification loss
        loss_cls = self.multilabel_soft_margin_loss(cls_out, gt_labels).mean()
        #print(f"{loss_cls=}")

        # keep track of losses for logging
        batch_dict = {"loss_cls": loss_cls.item(),
                      "loss_fg": cls_fg.mean().item()}

        loss = loss_cls.clone()
        if "dec" in masks:
            loss_mask = loss_mask.mean()

            if not PRETRAIN:
                loss += self.MASK_LOSS_BCE * loss_mask

            assert not "pseudo" in masks
            masks["pseudo"] = pseudo_gt
            batch_dict["loss_mask"] = loss_mask.item()

        batch_dict["loss"] = loss #loss.item()

        for mask_key, mask_val in masks.items():
            masks[mask_key] = masks[mask_key].detach()

        mask_logits = mask_logits.detach()

        cls_out = cls_out.detach()
        cls_pred = self.output2pred(cls_out)
        #print(f"{cls_pred=}")
        #print(f"{gt_labels=}")

        # logging
        if logging == True:
            self.accuracy[split](cls_pred, gt_labels)
            self._update_dice(self.dice[split], masks["cam"][:, 1:], fibrosis_seg_label, myo_seg, cheat=self.cheat_dice)
            self._update_dice(self.dice2[split], masks["cam"][:, 1:], fibrosis_seg_label, myo_seg, cheat=self.cheat_dice)
            self._update_dice(self.dice3[split], masks["cam"][:, 1:], fibrosis_seg_label, myo_seg, cheat=self.cheat_dice)
            self._update_dice(self.dice_pseudo_gt[split], masks["pseudo"][:, 1:], fibrosis_seg_label, myo_seg, cheat=self.cheat_dice)


            if (self.current_epoch + 1) % self.heavy_log == 0:
                self.auroc[split](cls_pred, gt_labels)
                self.confusion[split](cls_pred, gt_labels)
                self.pr_curve[split](cls_pred, gt_labels)
                self.roc[split](cls_pred, gt_labels)

                if self.example_batch[split] == None:
                    self.example_batch[split] = batch

        # make sure to cut the return values from graph
        return batch_dict, cls_out.detach(), masks, mask_logits


    def log_cams(self, split, mask_key):
        assert mask_key in ["cam", "dec", "pseudo"]
        example_batch = self.example_batch[split]
        # reference image grid with labels
        example_img_batch = example_batch['img']
        example_label_batch = example_batch['label']
        example_myo_seg_batch = example_batch['myo_seg']
        example_fibrosis_seg_batch = example_batch['fibrosis_seg_label']
        example_img_path = example_batch['img_path']

        batch_dict, cls_out, masks, mask_logits = self._step(example_batch, split, logging=False)
        example_prediction_batch = torch.squeeze(self.output2pred(cls_out)).cpu().detach()

        cls_pred = self.output2pred(cls_out)

        grid_imgs = make_grid_with_labels(example_img_batch, example_label_batch, example_prediction_batch, nrow=4, normalize=True).detach().cpu()
        grid_myo = torchvision.utils.make_grid(example_myo_seg_batch, nrow=4, padding=2).cpu().detach()
        grid_myo_masked = np.ma.masked_where(grid_myo < 0.5, grid_myo)
        grid_fibrosis = torchvision.utils.make_grid(example_fibrosis_seg_batch, nrow=4, padding=2).cpu().detach()
        grid_fibrosis_masked = np.ma.masked_where(grid_fibrosis <= 0, grid_fibrosis)

        grid_cams = torchvision.utils.make_grid(masks[mask_key][:,1:], nrow=4, padding=2).cpu().detach()

        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        fig.colorbar(cm.ScalarMappable(cmap="coolwarm"), ax=axs.ravel().tolist())

        axs[0].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[1].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[1].imshow(grid_myo_masked[0,:,:], cmap="Blues", alpha=0.5, vmin=0, vmax=1)
        axs[1].imshow(grid_fibrosis_masked[0,:,:], cmap="Reds", alpha=0.8, vmin=0, vmax=1)
        axs[2].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[2].imshow(grid_cams[0,:,:], cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)

        axs[0].set_axis_off()
        axs[1].set_axis_off()
        axs[2].set_axis_off()
        self.logger.experiment.add_figure(f"{mask_key}/{split}", fig, self.current_epoch)
        # for training set, show a different sample each time
        if split == "Train" and mask_key == "cam":
            self.example_batch[split] = None
        return




    def training_step(self, train_batch, batch_idx):
        batch_dict, _, _, _ = self._step(train_batch, "Train")
        return batch_dict


    def validation_step(self, val_batch, batch_idx):
        batch_dict, _, _, _ = self._step(val_batch, "Val")
        return batch_dict


    def test_step(self, test_batch, batch_idx):
        batch_dict, _, _, _ = self._step(test_batch, "Test")
        return batch_dict


    def _epoch_end(self, outputs, split):
        assert split in ["Train", "Val", "Test"]
        # log losses
        for key in outputs[0]:
            if "loss" in key:
                epoch_loss = torch.nanmean(torch.Tensor([x[key] for x in outputs]))
                self.logger.experiment.add_scalar(f"{key}/{split}", epoch_loss, self.current_epoch)

        # log accuracy
        epoch_accuracy = self.accuracy[split].compute()
        self.accuracy[split].reset()
        self.logger.experiment.add_scalar(f"Accuracy/{split}", epoch_accuracy, self.current_epoch)
        # log dice
        for name, metric in zip(["Dice_t=0.5", "Dice_t=0.8", "Dice_t=0.2", "Dice_pseudo"], [self.dice[split], self.dice2[split], self.dice3[split], self.dice_pseudo_gt[split]]):
            epoch_dice = metric.compute()
            metric.reset()
            self.logger.experiment.add_scalar(f"{name}/{split}", epoch_dice, self.current_epoch)
        # heavy log
        if (self.current_epoch + 1) % self.heavy_log == 0:
            epoch_auroc = self.auroc[split].compute()
            self.auroc[split].reset()
            self.logger.experiment.add_scalar(f"AUROC/{split}", epoch_auroc, self.current_epoch)

            confusion_matrix = self.confusion[split].compute()
            self.confusion[split].reset()
            confusion_figure = plot_confusion_matrix(confusion_matrix, range(confusion_matrix.shape[0]))
            self.logger.experiment.add_figure(f"Confusion matrix/{split}", confusion_figure, self.current_epoch)

            precision, recall, pr_thresholds = self.pr_curve[split].compute()
            self.pr_curve[split].reset()
            pr_curve_figure = plot_pr_curve(precision, recall, pr_thresholds, self.auc[split])
            self.auc[split].reset()
            self.logger.experiment.add_figure(f"PR-curve/{split}", pr_curve_figure, self.current_epoch)

            fpr, tpr, roc_thresholds = self.roc[split].compute()
            self.roc[split].reset()
            roc_figure = plot_roc(fpr, tpr, roc_thresholds, self.auc[split])
            self.auc[split].reset()
            self.logger.experiment.add_figure(f"ROC/{split}", roc_figure, self.current_epoch)

            self.log_cams(split, "pseudo")
            self.log_cams(split, "dec")
            self.log_cams(split, "cam")

        return


    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "Train")


    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "Val")


    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "Test")


### PLOTTING FUNCTIONS

def plot_confusion_matrix(cm, class_names):
  """
  stolen from www.tensorflow.org/tensorboard/image_summaries

  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  cm = cm.detach().cpu().numpy()
  figure = plt.figure(figsize=(7, 7))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def plot_pr_curve(precision, recall, thresholds, auc):
    if not isinstance(precision, list):
        precision, recall = [precision], [recall]
    figure = plt.figure(figsize=(7, 7))
    plt.title("Precision-recall curve")
    plt.tight_layout()
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    for precision_class, recall_class in zip(precision, recall):
        auc(recall_class, precision_class)
        auc_class = auc.compute()
        plt.plot(recall_class.detach().cpu(), precision_class.detach().cpu(), label=f"AUC:{auc_class:.3f}")
    plt.legend()
    return figure

def plot_roc(fpr, tpr, thresholds, auc):
    if not isinstance(fpr, list):
        fpr, tpr = [fpr], [tpr]
    figure = plt.figure(figsize=(7, 7))
    plt.title("Receiver Operating Characteristic")
    plt.tight_layout()
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.plot([0, 1], [0, 1], linestyle='--')
    for fpr_class, tpr_class in zip(fpr, tpr):
        auc(fpr_class, tpr_class)
        auc_class = auc.compute()
        plt.plot(fpr_class.detach().cpu(), tpr_class.detach().cpu(), label=f"AUC:{auc_class:.3f}")
    plt.legend()
    return figure

irange = range

def make_grid_with_labels(tensor, labels, predictions, nrow=8, limit=20, padding=2,
                          normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        labels (Tensor):  labels tensor of shape (B)
        predicitons (Tensor):  predictions tensor of shape (B)
        limit ( int, optional): Limits number of images and labels to make grid of
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    # Opencv configs
    if limit is not None:
        tensor = tensor[:limit, ::]
        labels = labels[:limit]
        predictions = predictions[:limit]


    font = 1
    fontScale = 1
    color = (255, 0, 0)
    thickness = 1

    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            working_tensor = tensor[k]
            if labels is not None:
                org = (0, int(tensor[k].shape[1] * 0.1))
                working_image = cv2.UMat(
                    np.asarray(np.transpose(working_tensor.cpu().numpy(), (1, 2, 0)) * 255).astype('uint8'))
                image = cv2.putText(working_image, f"label: {str(labels[k].item())}      pred: {predictions[k].item():.2f}",
                                    org, font, fontScale, color, thickness, cv2.LINE_AA)
                working_tensor = transforms.ToTensor()(image.get())
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(working_tensor)
            k = k + 1
    return grid


if __name__ == '__main__':
    print("nothing here")
