import itertools

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from skimage.metrics import hausdorff_distance

from .criterions import Diceloss, UnderfittingDiceloss
from .segmentation_models import (CA_2d_Unet, Floor_2d_Unet,
                                  Floor_3D_full_Unet, Floor_3D_half_Unet,
                                  Simple_2d_Unet)
from .transfer_learning_utils import freeze


class MyDice(torchmetrics.Metric):
    def __init__(self, threshold=0.5, dist_sync_on_step=False, smoothing=0, name='Dice'):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = name
        self.threshold = threshold
        self.smoothing = smoothing
        self.add_state('sum_dice_scores', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, f"shapes {preds.shape} and {target.shape} don't match"
        # select only masks that have a valid fibrosis label
        non_batch_dims = tuple([i for i in range(1, len(target.shape))])
        labeled =  (target.amin(dim=non_batch_dims, keepdim=False) > -0.01)
        if labeled.sum() == 0:
            return
        preds = preds[labeled]
        target = target[labeled]

        preds = preds > self.threshold
        target = target > 0.0

        # without smoothing, dice is undefined when both pred and target empty, only evaluate for fibrotic patients
        if self.smoothing == 0:
            non_empty =  (target.amax(dim=non_batch_dims, keepdim=False) != 0)
            if non_empty.sum() == 0:
                return
            preds = preds[non_empty]
            target = target[non_empty]

        # calculate dice separately for each image in batch
        intersection = (preds * target).sum(dim=non_batch_dims)
        dice = (2 * intersection +  self.smoothing) / (preds.sum(dim=non_batch_dims) + target.sum(dim=non_batch_dims) + self.smoothing)
        self.sum_dice_scores += dice.sum()
        self.total += dice.numel()

    def compute(self):
        return self.sum_dice_scores / self.total



class ImageLevelClassificationStats(torchmetrics.Metric):
    """ Calculates image-level True Positive, False Positive, False Negative and True Negative,
        given pixel-level (pseudo)labels and predictions"""
    def __init__(self, threshold=0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state('tp', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, f"shapes {preds.shape} and {target.shape} don't match"
        non_batch_dims = tuple([i for i in range(1, len(target.shape))])
        # select only masks that have a valid fibrosis label
        labeled =  (target.amin(dim=non_batch_dims, keepdim=False) > -0.01)
        if labeled.sum() == 0:
            return
        preds = preds[labeled]
        target = target[labeled]

        preds = preds > self.threshold
        target = target > 0.0

        negative_targets = (target.amax(dim=non_batch_dims, keepdim=False) == 0)
        positive_targets = (target.amax(dim=non_batch_dims, keepdim=False) == 1)
        negative_preds = (preds.amax(dim=non_batch_dims, keepdim=False) == 0)
        positive_preds = (preds.amax(dim=non_batch_dims, keepdim=False) == 1)

        self.tp += (positive_targets * positive_preds).sum()
        self.fp += (negative_targets * positive_preds).sum()
        self.fn += (positive_targets * negative_preds).sum()
        self.tn += (negative_targets * negative_preds).sum()

    def compute(self):
        return {'tp' : self.tp, 'fp' : self.fp, 'fn' : self.fn, 'tn' : self.tn}



class HausdorffDistance(torchmetrics.Metric):
    """Calculate Hausdorff distance on 2D images for non-empty targets and predictions """
    def __init__(self, threshold=0.5, dist_sync_on_step=False, name='Hausdorff'):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.name = name
        self.add_state('dist_sum', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, crop_corners, spacing):
        assert preds.shape == target.shape, f"shapes {preds.shape} and {target.shape} don't match"
        non_batch_dims = tuple([i for i in range(1, len(target.shape))])
        # select only masks that have a valid fibrosis label
        labeled =  (target.amin(dim=(1, 2, 3), keepdim=False) != -1)
        if labeled.sum() == 0:
            return
        preds = preds[labeled]
        target = target[labeled]

        preds = preds > self.threshold
        target = target > 0.0

        # to compute hausdorff distance, both prediction and target need to be non-empty
        non_empty = (preds.amax(dim=(1, 2, 3), keepdim=False) != 0) * (target.amax(dim=(1, 2, 3), keepdim=False) != 0)
        if non_empty.sum() == 0:
            return
        preds = preds[non_empty]
        target = target[non_empty]

        for i in range(len(preds)):
            dist = hausdorff_distance(preds[i].cpu().numpy(), target[i].cpu().numpy())
            pixel_dist = spacing[0][i] * (crop_corners[1][i] - crop_corners[0][i]) / preds[i].shape[-1]
            self.dist_sum += dist * pixel_dist
            self.total += 1

    def compute(self):
        return self.dist_sum / self.total


class DetectionStats(torchmetrics.Metric):
    def __init__(self, threshold=0.5, dice_threshold=0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.dice_threshold = dice_threshold
        # TRUE/FALSE POSITIVE/NEGATIVE  
        self.add_state('tp', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, f"shapes {preds.shape} and {target.shape} don't match"
        non_batch_dims = tuple([i for i in range(1, len(target.shape))])
        # select only masks that have a valid fibrosis label
        labeled =  (target.amin(dim=non_batch_dims, keepdim=False) > -0.01)
        if labeled.sum() == 0:
            return
        preds = preds[labeled]
        target = target[labeled]

        preds = preds > self.threshold
        target = target > 0.0

        negative_targets = (target.amax(dim=non_batch_dims, keepdim=False) == 0)
        positive_targets = (target.amax(dim=non_batch_dims, keepdim=False) == 1)
        negative_preds = (preds.amax(dim=non_batch_dims, keepdim=False) == 0)
        positive_preds = (preds.amax(dim=non_batch_dims, keepdim=False) == 1)

        self.tn  += (negative_preds * negative_targets).sum()
        self.fn += (positive_targets * negative_preds).sum()
        self.fp += (negative_targets * positive_preds).sum()

        # without smoothing, dice is undefined when both pred and target empty
        non_empty = (preds.amax(dim=non_batch_dims, keepdim=False) != 0) * (target.amax(dim=non_batch_dims, keepdim=False) != 0)
        if non_empty.sum() == 0:
            return
        preds = preds[non_empty]
        target = target[non_empty]

        # calculate dice separately for each image in batch
        intersection = (preds * target).sum(dim=non_batch_dims)
        dice = (2 * intersection) / (preds.sum(dim=non_batch_dims) + target.sum(dim=(1, 2, 3)))

        self.tp += (dice > self.dice_threshold).sum()
        self.fp += (dice <= self.dice_threshold).sum()


    def compute(self):
        return {'tp' : self.tp, 'fp' : self.fp, 'fn' : self.fn, 'tn' : self.tn}



class SegmentationModel(pl.LightningModule):

    def __init__(self, lr=1e-3, weight_decay=0,
                schedule=None, max_iters=-1, warmup_iters=0,
                num_classes=1, heavy_log=10,
                model_name='UNet2D', loss_function_string='dice',
                bilinear=True, in_chans=1,
                feature_multiplication=4, train_with_gt=False,
                freeze_n=None, train_bn=True,
                underfitting_warmup=20,
                underfitting_k=0.5,
                **kwargs):
        super().__init__()
        self.heavy_log = heavy_log
        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.schedule = schedule
        # finetuning
        self.train_with_gt = train_with_gt
        self.freeze_n = freeze_n
        self.train_bn = train_bn
        # model
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.feature_multiplication = feature_multiplication
        self.model_name = model_name
        self.loss_function_string = loss_function_string
        self.underfitting_warmup = underfitting_warmup if underfitting_warmup != None else np.inf
        self.save_hyperparameters()
        if loss_function_string == 'dice' and underfitting_warmup == None:
            self.loss_function = Diceloss()
        elif loss_function_string == 'dice':
            self.loss_function = UnderfittingDiceloss(k=underfitting_k)
        else:
            raise ValueError(f"Loss function {loss_function_string} not known")

        if self.model_name == 'UNet2D':
            self.model = Simple_2d_Unet(1, num_classes, bilinear=bilinear, adapted=False)
        elif self.model_name == 'UNet2D_stacked':
            self.model = Simple_2d_Unet(2, num_classes, bilinear=bilinear, adapted=False)
        elif self.model_name == 'UNet2D_small':
            self.model = Simple_2d_Unet(1, num_classes, bilinear=bilinear, adapted=True)
        elif self.model_name == 'UNet2D_stacked_small':
            self.model = Simple_2d_Unet(2, num_classes, bilinear=bilinear, adapted=True)
        elif self.model_name == 'CANet2D':
            if bilinear:
                raise Exception('Upsample implementation not allowed for CANet2D')
            self.model = CA_2d_Unet(1, num_classes, bilinear=bilinear)
        elif self.model_name == 'CANet2D_stacked':
            if bilinear:
                raise Exception('Upsample implementation not allowed for CANet2D_stacked')
            self.model = CA_2d_Unet(2, num_classes, bilinear=bilinear)
        elif self.model_name == 'Floor_UNet2D':
            self.model = Floor_2d_Unet(1, num_classes, bilinear=bilinear, feature_multiplication=self.feature_multiplication)
        elif self.model_name == 'Floor_UNet2D_stacked':
            self.model = Floor_2d_Unet(2, num_classes, bilinear=bilinear, feature_multiplication=self.feature_multiplication)
        elif self.model_name == 'UNet3D':
            self.model = Floor_3D_full_Unet(1, num_classes, bilinear=bilinear, feature_multiplication=self.feature_multiplication)
        elif self.model_name == 'UNet3D_channels':
            feat_mult = 2
            print('Feature multiplication factor:', feat_mult)
            self.model = Floor_2d_Unet(13, 13, bilinear=bilinear, feature_multiplication=feat_mult)
        elif self.model_name == 'UNet3D_channels_stacked':
            self.model = Floor_2d_Unet(26, 13, bilinear=bilinear)
        elif self.model_name == 'UNet3D_half':
            self.model = Floor_3D_half_Unet(1, num_classes, bilinear=bilinear)
        elif self.model_name == 'UNet3D_cascaded':
            self.model = Floor_3D_full_Unet(2, num_classes, bilinear=bilinear, feature_multiplication=self.feature_multiplication)
        elif self.model_name == 'UNet3D_cascaded_stacked':
            self.model = Floor_3D_full_Unet(3, num_classes, bilinear=bilinear, feature_multiplication=self.feature_multiplication)
        else:
            raise ValueError(f"model_name  {model_name} not known")

        # freeze layers for transfer learning
        if self.freeze_n != None:
            children = list(self.model.children())
            n_max = int(self.freeze_n)
            print("Finetuning layers:")
            for child in children[-n_max:]:
                print(f"    {child}")


        # setup metrics & logging parameters
        def _make_metric_dict(metric, splits=["Train", "Val", "Test"], **kwargs):
            return nn.ModuleDict({split : metric(**kwargs) for split in splits })

        self.dice_pred_pseudo_smooth = _make_metric_dict(MyDice, threshold=0.5, smoothing=1, name="Dice_pred_pseudo_smoothed")
        self.dice_pred_gt_smooth = _make_metric_dict(MyDice, threshold=0.5, smoothing=1, name="Dice_pred_gt_smoothed")
        self.dice_pseudo_gt_smooth = _make_metric_dict(MyDice, threshold=0.5, smoothing=1, name="Dice_pseudo_gt_smoothed")

        self.dice_pred_pseudo_unsmoothed = _make_metric_dict(MyDice, threshold=0.5, smoothing=0, name="Dice_pred_pseudo_unsmoothed")
        self.dice_pred_gt_unsmoothed = _make_metric_dict(MyDice, threshold=0.5, smoothing=0, name="Dice_pred_gt_unsmoothed")
        self.dice_pseudo_gt_unsmoothed = _make_metric_dict(MyDice, threshold=0.5, smoothing=0, name="Dice_pseudo_gt_unsmoothed")

        self.hausdorff_pred_pseudo = _make_metric_dict(HausdorffDistance, threshold=0.5, name="Hausdorf_pred_pseudo")
        self.hausdorff_pred_gt = _make_metric_dict(HausdorffDistance, threshold=0.5, name="Hausdorf_pred_gt")
        self.hausdorff_pseudo_gt = _make_metric_dict(HausdorffDistance, threshold=0.5, name="Hausdorf_pseudo_gt")

        self.dices = [self.dice_pred_pseudo_smooth, self.dice_pred_gt_smooth, self.dice_pseudo_gt_smooth,
                        self.dice_pred_pseudo_unsmoothed, self.dice_pred_gt_unsmoothed, self.dice_pseudo_gt_unsmoothed]

        self.hausdorffs = [ self.hausdorff_pred_pseudo, self.hausdorff_pred_gt, self.hausdorff_pseudo_gt]

        self.classification_stats_pseudo = _make_metric_dict(ImageLevelClassificationStats, threshold=0.5)
        self.classification_stats_gt = _make_metric_dict(ImageLevelClassificationStats, threshold=0.5)

        self.detection_stats_pseudo = _make_metric_dict(DetectionStats, threshold=0.5, dice_threshold=0.5)
        self.detection_stats_gt = _make_metric_dict(DetectionStats, threshold=0.5, dice_threshold=0.5)

        self.example_batch = {"Train" : None, "Val" : None , "Test": None}



    def forward(self, imgs):
        if self.model_name in  ['CANet2D', 'CANet2D_stacked']:
            output, attention_coefs = self.model(imgs)
        else:
            output = self.model(imgs)
            attention_coefs = None
        return output, attention_coefs


    def train(self, mode=True):
        """Overrided train to freeze layers"""
        super().train(mode=mode)
        if mode and self.freeze_n != None:
            # freeze model (except last layers and potentially batchnorm)
            freeze(module=self.model,
                   n=-self.freeze_n,
                   train_bn=self.train_bn)


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



    def _step(self, batch, split, logging=True):
        assert split in ["Train", "Val", "Test"]

        image = batch['img']
        pseudoseg = batch['pseudo_fib']
        myo_seg = batch['pred_myo']
        gt_seg = batch['gt_fib']
        crop_corners = batch['crop_corners']
        spacing = batch['spacing']
        img_path = batch['img_path']

        # format model input
        pred_myo_input = ('stacked' in self.model_name)
        pred_fib_input = ('cascaded' in self.model_name)
        if (not pred_myo_input) and (not pred_fib_input):
            input = image
        elif pred_myo_input and not pred_fib_input:
            input = torch.stack([image.squeeze(1), myo_seg.squeeze(1)], dim=1)
        elif pred_fib_input and not pred_myo_input:
            input = torch.stack([image.squeeze(1), batch['pred_fib'].squeeze(1)], dim=1)
        elif pred_fib_input and pred_myo_input:
            input = torch.stack([image.squeeze(1), myo_seg.squeeze(1), batch['pred_fib'].squeeze(1)], dim=1)

        # forward model
        output, attention_coefs = self.forward(input.float())

        # compute loss
        if self.train_with_gt == True:
            assert gt_seg.amin() != -1, f"File: {img_path} does not have a valid ground truth label"
            loss = self.loss_function(output, (gt_seg > 0).float(), underfitting=(self.current_epoch > self.underfitting_warmup))
        else:
            loss = self.loss_function(output, (pseudoseg > 0.5).float(), underfitting=(self.current_epoch > self.underfitting_warmup))

        batch_dict = {
            "loss" : loss
        }

        if logging == True:
            output = output.detach()
            self.dice_pred_pseudo_smooth[split](output, pseudoseg)
            self.dice_pred_gt_smooth[split](output, gt_seg)
            self.dice_pseudo_gt_smooth[split](pseudoseg, gt_seg)
            self.dice_pred_pseudo_unsmoothed[split](output, pseudoseg)
            self.dice_pred_gt_unsmoothed[split](output, gt_seg)
            self.dice_pseudo_gt_unsmoothed[split](pseudoseg, gt_seg)

            if (self.current_epoch + 1) % self.heavy_log == 0:
                if not '3D' in self.model_name:
                    self.hausdorff_pred_pseudo[split](output, pseudoseg, crop_corners, spacing)
                    self.hausdorff_pred_gt[split](output, gt_seg, crop_corners, spacing)
                    self.hausdorff_pseudo_gt[split](pseudoseg, gt_seg, crop_corners, spacing)

                self.classification_stats_pseudo[split](output, pseudoseg)
                self.classification_stats_gt[split](output, gt_seg)

                self.detection_stats_pseudo[split](output, pseudoseg)
                self.detection_stats_gt[split](output, gt_seg)
                if self.example_batch[split] == None:
                    self.example_batch[split] = batch
        return batch_dict, output



    def training_step(self, batch, batch_idx):
        batch_dict, _ = self._step(batch, "Train")
        return batch_dict

    def validation_step(self, batch, batch_idx):
        batch_dict, _ = self._step(batch, "Val")
        return batch_dict

    def test_step(self, batch, batch_idx):
        batch_dict, _ = self._step(batch, "Test")
        return batch_dict

    def log_images(self, split):
        example_batch = self.example_batch[split]
        example_img_batch = example_batch['img']
        example_pseudoseg_batch = example_batch['pseudo_fib']
        example_myo_seg_batch = example_batch['pred_myo']
        example_gt_seg_batch = example_batch['gt_fib']
        example_img_path = example_batch['img_path']

        batch_dict, output = self._step(example_batch, split, logging=False)
        # possible depth dimension to batch dimension for visualization
        if len(example_img_batch.shape) == 5:
            B, C, D, H, W = example_img_batch.shape
            example_img_batch = example_img_batch.view(B*D, C, H, W)
            example_pseudoseg_batch = example_pseudoseg_batch.view(B*D, C, H, W)
            example_myo_seg_batch = example_myo_seg_batch.view(B*D, C, H, W)
            example_gt_seg_batch = example_gt_seg_batch.view(B*D, C, H, W)
            output = output.view(B*D, C, H, W)
        # image
        grid_imgs = torchvision.utils.make_grid(example_img_batch, nrow=4, padding=2).cpu().detach()
        # myocard predictions (from other model)
        grid_myo = torchvision.utils.make_grid(example_myo_seg_batch, nrow=4, padding=2).cpu().detach()
        grid_myo_masked = np.ma.masked_where(grid_myo < 0.5, grid_myo)
        # ground truth fibrosis segmentation (if available)
        grid_gt_seg = torchvision.utils.make_grid(example_gt_seg_batch, nrow=4, padding=2).cpu().detach()
        grid_gt_seg_masked = np.ma.masked_where(grid_gt_seg <= 0, grid_gt_seg)
        # pseudo fibrosis segmentation
        grid_pseudoseg = torchvision.utils.make_grid(example_pseudoseg_batch, nrow=4, padding=2).cpu().detach()
        grid_pseudoseg_masked = np.ma.masked_where(grid_pseudoseg <= 0, grid_pseudoseg)
        # fibrosis segmentation predictions
        grid_output = torchvision.utils.make_grid(output, nrow=4, padding=2).cpu().detach()
        grid_output_masked = np.ma.masked_where(grid_output <= 0, grid_output)

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        fig.colorbar(cm.ScalarMappable(cmap="coolwarm"), ax=axs.ravel().tolist())

        axs[0, 0].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[0, 0].set_title('Image')

        axs[0, 1].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[0, 1].imshow(grid_myo_masked[0,:,:], cmap="Blues", alpha=0.5, vmin=0, vmax=1)
        axs[0, 1].imshow(grid_gt_seg_masked[0,:,:], cmap="Reds", alpha=0.8, vmin=0, vmax=1)
        axs[0, 1].set_title('Myo seg + fibrosis ground truth')

        axs[1, 0].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[1, 0].imshow(grid_pseudoseg_masked[0,:,:], cmap="Reds", alpha=0.8, vmin=0, vmax=1)
        axs[1, 0].set_title('Fibrosis pseudo-label')

        axs[1, 1].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[1, 1].imshow(grid_output_masked[0,:,:], cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)
        axs[1, 1].set_title('Fibrosis prediction')

        axs[0, 0].set_axis_off()
        axs[0, 1].set_axis_off()
        axs[1, 0].set_axis_off()
        axs[1, 1].set_axis_off()

        self.logger.experiment.add_figure(f"Images/{split}", fig, self.current_epoch)
        # show a different sample each time
        self.example_batch[split] = None
        return



    def log_confusion_matrix(self, split):
        class_names = ["Fibrotic", "Normal"]
        for name, classification_stats in zip(["Image_level_confusion_gt", "Image_level_confusion_pseudo", "Detection_confusion_gt", "Detection_confusion_pseudo"], [self.classification_stats_gt, self.classification_stats_pseudo, self.detection_stats_gt, self.detection_stats_pseudo]):
            stats = classification_stats[split].compute()
            classification_stats[split].reset()
            cm = np.array([
                            [stats['tp'].item(), stats['fn'].item()],
                            [stats['fp'].item(), stats['tn'].item()]
                         ])
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
            self.logger.experiment.add_figure(f"{name}/{split}", figure, self.current_epoch)



    def _epoch_end(self, outputs, split):
        assert split in ["Train", "Val", "Test"]
        epoch_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f"Loss/{split}", epoch_loss, self.current_epoch)

        # log dice
        for metric in self.dices:
            epoch_m = metric[split].compute()
            metric[split].reset()
            self.logger.experiment.add_scalar(f"{metric[split].name}/{split}", epoch_m, self.current_epoch)

        # heavy log
        if (self.current_epoch + 1) % self.heavy_log == 0:
            self.log_images(split)
            self.log_confusion_matrix(split)
            if not '3D' in self.model_name:
                for metric in self.hausdorffs:
                    epoch_m = metric[split].compute()
                    metric[split].reset()
                    self.logger.experiment.add_scalar(f"{metric[split].name}/{split}", epoch_m, self.current_epoch)


    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "Train")


    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "Val")


    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "Test")





if __name__ == '__main__':
    print("")
    m = SegmentationModel()
