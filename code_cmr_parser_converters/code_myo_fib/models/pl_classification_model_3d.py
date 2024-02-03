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

from .drn_3d import DRN3D

class MyDice(torchmetrics.Metric):
    def __init__(self, threshold=0.5, dist_sync_on_step=False, smoothing=0, name='Dice', dim="3D"):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = name
        self.threshold = threshold
        self.smoothing = smoothing
        assert dim in ["2D", "3D"]
        self.dim = dim
        self.add_state('sum_dice_scores', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, weak_label: torch.Tensor):
        assert preds.shape == target.shape, f"shapes {preds.shape} and {target.shape} don't match"
        # remove padded slices
        non_padded = ~weak_label.isnan()
        B, C, D, H, W = preds.shape
        if self.dim == "2D":
            preds = preds.view(B*D, -1)[non_padded.view(-1)]
            target = target.view(B*D, -1)[non_padded.view(-1)]
        elif self.dim == "3D":
            preds = preds.view(B*D, -1)[non_padded.view(-1)]
            target = target.view(B*D, -1)[non_padded.view(-1)]

            preds = preds.view(B, -1)
            target = target.view(B, -1)

        # remove slices/mris without valid target label
        labeled =  (target.amin(dim=(1), keepdim=False) > -0.01)
        if labeled.sum() == 0:
            return
        preds = preds[labeled]
        target = target[labeled]

        preds = preds > self.threshold
        target = target > 0.0

        # without smoothing, dice is undefined when both pred and target empty. only evaluate fibrotic patients
        if self.smoothing == 0:
            non_empty =  (target.amax(dim=(1), keepdim=False) > 0.01) # * (preds.amax(dim=(1), keepdim=False) > 0.01)
            if non_empty.sum() == 0:
                return
            preds = preds[non_empty]
            target = target[non_empty]

        # calculate dice separately for each image in batch
        intersection = (preds * target).sum(dim=(1))
        dice = (2 * intersection +  self.smoothing) / (preds.sum(dim=(1)) + target.sum(dim=(1)) + self.smoothing)

        self.sum_dice_scores += dice.sum()
        self.total += dice.numel()

    def compute(self):
        return self.sum_dice_scores / self.total



class ClassificationModel3D(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=0, label_smoothing=0,
                schedule=None, max_iters=-1, warmup_iters=0,
                num_classes=1, in_chans=1,
                heavy_log=10, pooling="avg", feature_dropout=0.0,
                cam_dropout=0.0,
                myo_mask_pooling=False, myo_mask_threshold=0.3,
                downsampling=8,
                myo_mask_dilation=0, cheat_dice=False,
                myo_mask_prob=1, model='drnd',
                classification_level='2D', pos_weight=0.5,
                myo_input=False,
                **kwargs):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.schedule = schedule
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.num_classes = num_classes
        self.save_hyperparameters()
        self.heavy_log = heavy_log
        self.pooling = pooling
        self.myo_mask_pooling = myo_mask_pooling
        self.myo_mask_threshold = myo_mask_threshold
        self.downsampling = downsampling
        self.cheat_dice = cheat_dice
        self.myo_mask_prob = myo_mask_prob
        self.myo_mask_dilation = myo_mask_dilation
        assert classification_level in ["2D", "3D"]
        self.classification_level = classification_level
        self.pos_weight = pos_weight
        self.myo_input = myo_input
        # load backbone
        if model == 'drnd':
            self.model = DRN3D(num_classes=self.num_classes,
                                in_chans=in_chans, downsampling=self.downsampling,
                                **kwargs)
        else:
            raise NotImplementedError

        # dropout
        self.feature_dropout_layer = nn.Dropout3d(p=feature_dropout)
        self.cam_dropout_layer = nn.Dropout(p=cam_dropout)
        # 1x1 class convolution
        self.classifier = nn.Conv3d(self.model.fan_out(), self.num_classes, kernel_size=1, bias=True)

        # global pooling
        assert pooling in ["avg", "max"]
        if classification_level == "3D":
            if self.myo_mask_pooling == False:
                if pooling == "avg":
                    self.final_pool =  nn.AdaptiveAvgPool3d((1, 1, 1))
                elif pooling == "max":
                    self.final_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
            else:
                if pooling == "avg":
                    self.final_pool = MaskedAdaptiveAvgPool3d((1, 1, 1))
                elif pooling == "max":
                    self.final_pool = MaskedAdaptiveMaxPool3d((1, 1, 1))
        elif classification_level == "2D":
            if self.myo_mask_pooling == False:
                if pooling == "avg":
                    self.final_pool =  nn.AdaptiveAvgPool3d((None, 1, 1))
                elif pooling == "max":
                    self.final_pool = nn.AdaptiveMaxPool3d((None, 1, 1))
            else:
                if pooling == "avg":
                    self.final_pool = MaskedAdaptiveAvgPool3d((None, 1, 1))
                elif pooling == "max":
                    self.final_pool = MaskedAdaptiveMaxPool3d((None, 1, 1))



        # metrics
        # setup metrics & logging parameters
        def _make_metric_dict(metric, splits=["Train", "Val", "Test"], **kwargs):
            return nn.ModuleDict({split : metric(**kwargs) for split in splits })

        self.accuracy = _make_metric_dict(torchmetrics.Accuracy, threshold=0.5)
        self.confusion = _make_metric_dict(torchmetrics.ConfusionMatrix, num_classes=max(2, self.num_classes), threshold=0.5)
        self.auroc = _make_metric_dict(torchmetrics.AUROC)
        self.pr_curve = _make_metric_dict(torchmetrics.PrecisionRecallCurve)
        self.roc = _make_metric_dict(torchmetrics.ROC)
        self.auc = _make_metric_dict(torchmetrics.AUC)

        # dice
        self.dice_t1_2D = _make_metric_dict(MyDice, threshold=0.1, smoothing=1, dim="2D", name="Dice_t.1_2D")
        self.dice_t3_2D = _make_metric_dict(MyDice, threshold=0.3, smoothing=1, dim="2D", name="Dice_t.3_2D")
        self.dice_t5_2D = _make_metric_dict(MyDice, threshold=0.5, smoothing=1, dim="2D", name="Dice_t.5_2D")
        self.dice_t1_3D = _make_metric_dict(MyDice, threshold=0.1, smoothing=1, dim="3D", name="Dice_t.1_3D")
        self.dice_t3_3D = _make_metric_dict(MyDice, threshold=0.3, smoothing=1, dim="3D", name="Dice_t.3_3D")
        self.dice_t5_3D = _make_metric_dict(MyDice, threshold=0.5, smoothing=1, dim="3D", name="Dice_t.5_3D")


        self.dices = [self.dice_t1_2D, self.dice_t3_2D, self.dice_t5_2D,
                     self.dice_t1_3D, self.dice_t3_3D, self.dice_t5_3D]
        #self.dices = [self.dice_t1_3D]


        self.example_batch = {"Train" : None, "Val" : None , "Test": None}
        self.output2pred = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=-1)



    def forward_features(self, x):
        return self.model.forward_features(x)


    def forward(self, x, myo_seg=None):
        if myo_seg != None and self.myo_input == True:
            x = torch.cat((x, myo_seg), dim=1)
        # forward features (class activation map)
        x = self.forward_features(x)
        x = self.feature_dropout_layer(x)
        x = self.classifier(x)
        x = self.cam_dropout_layer(x)
        if myo_seg == None or self.myo_mask_pooling == False:
            x = self.final_pool(x)
        else:
            myo_mask = self.myo_seg_to_mask(myo_seg, dilation=self.myo_mask_dilation)
            x = self.final_pool(x, myo_mask)
        x = torch.flatten(x, start_dim=1)
        return x

    def myo_seg_to_mask(self, x, dilation=0, size="cam"):
        assert size in ["cam", "image"]

        if dilation != 0:
            x = F.max_pool3d(x, kernel_size=(1, dilation, dilation), stride=1, padding=(0, dilation//2, dilation//2))
        if size == "cam":
            x = F.interpolate(x, scale_factor=(1, 1/self.downsampling, 1/self.downsampling), mode='area')
        elif size == "image":
            pass
        else:
            raise NotImplementedError

        x = (x > self.myo_mask_threshold).float()

        # mask only in training mode, or when in eval mode and model is trained using mask_prob 1
        myo_mask_prob = self.myo_mask_prob if (self.training or self.myo_mask_prob == 1) else 0
        no_mask = torch.rand(x.shape[0]) >  myo_mask_prob
        x[no_mask, ...] = 1.0

        return x


    def make_cam(self, x, select_values="pos_only", upsampling="no", myo_seg=None, cheat_gt=None,
                    fullsize_mask=False):
        if select_values not in ["pos_only", "all", "sigmoid", "logit"]:
            raise NotImplementedError
        if upsampling not  in ["no", "trilinear"]:
            raise NotImplementedError

        if myo_seg != None and self.myo_input == True:
            x = torch.cat((x, myo_seg), dim=1)
        x = self.forward_features(x)
        x = self.classifier(x)

        if cheat_gt != None:
            x = self.myo_seg_to_mask(cheat_gt)

        if myo_seg != None and self.myo_mask_pooling == True:
            myo_mask = self.myo_seg_to_mask(myo_seg, dilation=self.myo_mask_dilation)
            assert myo_mask.shape == x.shape, f"shapes {myo_mask.shape} and {x.shape} don't match"
            x = x * myo_mask

        eps = 1e-5
        if select_values == "pos_only":
            # remove negative and scale each class in [0,1]
            x = F.relu(x)
            x = x / (torch.amax(x, (-3, -2, -1), keepdims=True) + eps)
        elif select_values == "all":
            # scale all values in [0, 1]
            x = x - torch.amin(x, (-3, -2, -1), keepdims=True)
            x  = x / (torch.amax(x, (-3, -2, -1), keepdims=True) + eps)
        elif select_values == "sigmoid":
            x = torch.sigmoid(x)
        elif select_values == "logit":
            pass

        if upsampling == "trilinear":
            scale_factor = (1, self.downsampling, self.downsampling)
            x = F.interpolate(x, mode='trilinear', align_corners=True, scale_factor=scale_factor)
        elif upsampling == "no":
            pass

        if fullsize_mask == True:
            myo_mask = self.myo_seg_to_mask(myo_seg, dilation=self.myo_mask_dilation, size='image')
            x = x * myo_mask

        return x



    def configure_optimizers(self):
        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

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



    def cross_entropy_loss(self, outputs, labels):
        raise NotImplementedError # "NaN labels are not handled yet."
        # ignore padded slices with label NaN
        #outputs = outputs.view(-1)[~labels.isnan().view(-1)]
        #labels = labels[~labels.isnan()].view(-1)
        #return F.cross_entropy(outputs, labels, label_smoothing=self.label_smoothing, reduction='none').nanmean()



    def bce_with_logits_loss(self, outputs, labels):
        # bce requires float labels
        labels = labels.float()
        if self.label_smoothing != 0:
            with torch.no_grad():
                labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # ignore padded slices with label NaN
        outputs = outputs.view(-1)[~labels.isnan().view(-1)]
        labels = labels[~labels.isnan()].view(-1)
        loss = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=torch.ones_like(labels)*self.pos_weight)
        return loss

    def _step(self, batch, split, logging=True):
        assert split in ["Train", "Val", "Test"]

        image = batch['img']
        labels = batch['label']
        myo_seg = batch['myo_seg']
        fibrosis_seg_label = batch['fibrosis_seg_label']
        outputs = self.forward(image, myo_seg)


        if self.classification_level == "3D":
            # reduce slice-level supervision to mri-level supervision
            labels = labels.nan_to_num().amax(dim=-1, keepdim=True)

        if self.num_classes == 1:
            loss = self.bce_with_logits_loss(outputs, labels)
        else:
            loss = self.cross_entropy_loss(outputs, labels)

        batch_dict= {
            "loss" : loss
        }

        if logging == True:
            # filter out padded slices
            preds = self.output2pred(outputs).view(-1)[~labels.isnan().view(-1)]
            labels = labels[~labels.isnan()].view(-1).int()

            self.accuracy[split](preds, labels)

            if (self.current_epoch + 1) % self.heavy_log == 0:
                # ADD HEAVY LOGGING
                self.auroc[split](preds, labels)
                self.confusion[split](preds, labels)
                self.pr_curve[split](preds, labels)
                self.roc[split](preds, labels)

                # log dice
                if self.cheat_dice == True:
                    cheat_gt = fibrosis_seg_label
                else:
                    cheat_gt = None

                cam = self.make_cam(image, myo_seg=myo_seg, select_values="pos_only", upsampling="trilinear", cheat_gt=cheat_gt)
                for metric in self.dices:
                    metric[split](cam, fibrosis_seg_label, batch['label'])

                if self.example_batch[split] == None:
                    self.example_batch[split] = batch

        return batch_dict


    def training_step(self, batch, batch_idx):
        batch_dict = self._step(batch, "Train")
        return batch_dict

    def validation_step(self, batch, batch_idx):
        batch_dict = self._step(batch, "Val")
        return batch_dict

    def test_step(self, batch, batch_idx):
        batch_dict = self._step(batch, "Test")
        return batch_dict



    def log_confusion_matrix(self, split):
        class_names = ["Normal", "Fibrotic"]

        cm = self.confusion[split].compute().detach().cpu().numpy()
        self.confusion[split].reset()

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
        self.logger.experiment.add_figure(f"Confusion/{split}", figure, self.current_epoch)



    def log_pr_curve(self, split):
        precision, recall, thresholds = self.pr_curve[split].compute()
        self.pr_curve[split].reset()

        if not isinstance(precision, list):
            precision, recall = [precision], [recall]

        figure = plt.figure(figsize=(7, 7))
        plt.title("Precision-recall curve")
        plt.tight_layout()
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        for precision_class, recall_class in zip(precision, recall):
            self.auc[split](recall_class, precision_class)
            auc_class = self.auc[split].compute()
            self.auc[split].reset()

            plt.plot(recall_class.detach().cpu(), precision_class.detach().cpu(), label=f"AUC:{auc_class:.3f}")
        plt.legend()

        self.logger.experiment.add_figure(f"PR-curve/{split}", figure, self.current_epoch)



    def log_roc(self, split):
        fpr, tpr, thresholds = self.roc[split].compute()
        self.roc[split].reset()

        if not isinstance(fpr, list):
            fpr, tpr = [fpr], [tpr]

        figure = plt.figure(figsize=(7, 7))
        plt.title("Receiver Operating Characteristic")
        plt.tight_layout()
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.plot([0, 1], [0, 1], linestyle='--')
        for fpr_class, tpr_class in zip(fpr, tpr):
            self.auc[split](fpr_class, tpr_class)
            auc_class = self.auc[split].compute()
            self.auc[split].reset()
            plt.plot(fpr_class.detach().cpu(), tpr_class.detach().cpu(), label=f"AUC:{auc_class:.3f}")
        plt.legend()

        self.logger.experiment.add_figure(f"ROC/{split}", figure, self.current_epoch)




    def log_cams(self, example_batch, split_name, upsampling="trilinear", select_values="all"):
        img = example_batch['img']
        label = example_batch['label']
        myo_seg = example_batch['myo_seg']
        fibrosis_seg = example_batch['fibrosis_seg_label']
        img_path = example_batch['img_path']

        # reformat depth and batch to one dim, remove padded slices (NaN label)
        B, C, D, H, W = img.shape
        labeled = ~label.isnan().view(-1)
        img = img.view(B*D, C, H, W)[labeled]
        myo_seg = myo_seg.view(B*D, C, H, W)[labeled]
        fibrosis_seg = fibrosis_seg.view(B*D, C, H, W)[labeled]

        if self.classification_level == "2D":
            preds = self.output2pred(self.forward(example_batch['img'], example_batch['myo_seg']))
            preds = preds.view(-1)[labeled]
            label = label.view(-1)[labeled]
        elif self.classification_level == "3D":
            preds = self.output2pred(self.forward(example_batch['img'], example_batch['myo_seg']))
            preds = (preds * label).view(-1)[labeled]
            label = (torch.ones_like(label) * label.nan_to_num().amax(dim=-1, keepdim=True)).view(-1)[labeled]

        limit = 20

        grid_imgs = make_grid_with_labels(img[:limit], label[:limit], preds[:limit], nrow=4, normalize=True).detach().cpu()
        grid_myo = torchvision.utils.make_grid(myo_seg[:limit], nrow=4, padding=2).cpu().detach()
        grid_myo_masked = np.ma.masked_where(grid_myo < self.myo_mask_threshold, grid_myo)
        grid_fibrosis = torchvision.utils.make_grid(fibrosis_seg[:limit], nrow=4, padding=2).cpu().detach()
        grid_fibrosis_masked = np.ma.masked_where(grid_fibrosis <= 0, grid_fibrosis)

        cam = self.make_cam(example_batch['img'], select_values=select_values, upsampling=upsampling, myo_seg=example_batch['myo_seg']).cpu().detach()
        cam = cam.view(B*D, C, H, W)[labeled]
        grid_cams = torchvision.utils.make_grid(cam[:limit], nrow=4, padding=2).cpu().detach()

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
        self.logger.experiment.add_figure(f"CAM_{select_values=}_{upsampling=}/{split_name}", fig, self.current_epoch)
        #plt.show()
        return




    def _epoch_end(self, outputs, split):
        assert split in ["Train", "Val", "Test"]
        epoch_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f"Loss/{split}", epoch_loss, self.current_epoch)

        epoch_accuracy = self.accuracy[split].compute()
        self.accuracy[split].reset()
        self.logger.experiment.add_scalar(f"Accuracy/{split}", epoch_accuracy, self.current_epoch)

        if (self.current_epoch + 1) % self.heavy_log == 0:
            epoch_auroc = self.auroc[split].compute()
            self.auroc[split].reset()
            self.logger.experiment.add_scalar(f"AUROC/{split}", epoch_auroc, self.current_epoch)

            self.log_confusion_matrix(split)
            self.log_pr_curve(split)
            self.log_roc(split)

            self.log_cams(self.example_batch[split], split, upsampling="trilinear", select_values="pos_only")
            self.log_cams(self.example_batch[split], split, upsampling="trilinear", select_values="all")
            self.log_cams(self.example_batch[split], split, upsampling="trilinear", select_values="sigmoid")
            self.example_batch[split] = None

            # log dice
            for metric in self.dices:
                epoch_m = metric[split].compute()
                metric[split].reset()
                self.logger.experiment.add_scalar(f"{metric[split].name}/{split}", epoch_m, self.current_epoch)



    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "Train")



    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "Val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "Test")









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




class MaskedAdaptiveAvgPool3d(nn.Module):
    """Assumes input shape (B, C, D, H, W),
       output shape (B, C, output_size, output_size)
        Takes adaptive average only where mask is True"""
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x, mask):
        x = x * mask
        x = F.adaptive_avg_pool3d(x, self.output_size)
        # don't compensate for masked contribution to avg might be better?
        #number_false = (mask == False).sum(dim=(2, 3), keepdims=True)
        #number_true = (mask == True).sum(dim=(2, 3), keepdims=True)
        #x = x * (number_false + number_true) / (number_true + 1e-6)
        return x



class MaskedAdaptiveMaxPool3d(nn.Module):
    """Assumes input shape (B, C, D, H, W),
       output shape (B, C, output_size, output_size)
        Takes adaptive max only where mask is True"""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x, mask):
        # set masked out values to low value
        x[mask == False] = -10
        # take normal max pooling
        x = F.adaptive_max_pool3d(x, self.output_size)
        return x









if __name__ == '__main__':
    print("nothing here")
