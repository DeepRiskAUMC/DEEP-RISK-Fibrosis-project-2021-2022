import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from .criterions import Diceloss
from .pl_fib_segmentation_model import MyDice
from .segmentation_models import (CA_2d_Unet, Floor_2d_Unet,
                                  Floor_3D_full_Unet, Floor_3D_half_Unet,
                                  Simple_2d_Unet)


class MyoSegmentationModel(pl.LightningModule):

    def __init__(self, lr=1e-3, weight_decay=0,
                schedule=None, max_iters=-1, warmup_iters=0,
                num_classes=1, heavy_log=10,
                model_name='UNet2D', loss_function_string='dice',
                bilinear=True, in_chans=1,
                feature_multiplication=4,
                **kwargs):
        super().__init__()
        self.heavy_log = heavy_log
        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.schedule = schedule
        # model
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.feature_multiplication = feature_multiplication
        self.model_name = model_name
        self.loss_function_string = loss_function_string
        self.save_hyperparameters()
        if loss_function_string == 'dice':
            self.loss_function = Diceloss()
        else:
            raise ValueError(f"Loss function {loss_function_string} not known")

        if self.model_name == 'UNet2D':
            self.model = Simple_2d_Unet(1, num_classes, bilinear=bilinear, adapted=False)
        elif self.model_name == 'UNet2D_small':
            self.model = Simple_2d_Unet(1, num_classes, bilinear=bilinear, adapted=True)
        elif self.model_name == 'CANet2D':
            if bilinear:
                raise Exception('Upsample implementation not allowed for CANet2D')
            self.model = CA_2d_Unet(1, num_classes, bilinear=bilinear)
        elif self.model_name == 'Floor_UNet2D':
            self.model = Floor_2d_Unet(1, num_classes, bilinear=bilinear, feature_multiplication=self.feature_multiplication)
        elif self.model_name == 'UNet3D':
            self.model = Simple_3d_Unet(1, num_classes, bilinear=bilinear)
        elif self.model_name == 'UNet3D_channels':
            feat_mult = 2
            print('Feature multiplication factor:', feat_mult)
            self.model = Floor_2d_Unet(13, 13, bilinear=bilinear, feature_multiplication=feat_mult)
        elif self.model_name == 'UNet3D_half':
            self.model = Floor_3D_half_Unet(1, num_classes, bilinear=bilinear, feature_multiplication=self.feature_multiplication)
        elif self.model_name == 'UNet3D_cascaded':
            self.model = Floor_3D_full_Unet(2, num_classes, bilinear=bilinear, feature_multiplication=self.feature_multiplication)
        else:
            raise ValueError(f"model_name  {model_name} not known")

        # setup metrics & logging parameters
        def _make_metric_dict(metric, splits=["Train", "Val", "Test"], **kwargs):
            return nn.ModuleDict({split : metric(**kwargs) for split in splits })

        self.dice_smoothed = _make_metric_dict(MyDice, threshold=0.5, smoothing=1, name="Dice_smoothed")

        self.dice_unsmoothed = _make_metric_dict(MyDice, threshold=0.5, smoothing=0, name="Dice_unsmoothed")


        self.metrics = [self.dice_smoothed, self.dice_unsmoothed]
        self.example_batch = {"Train" : None, "Val" : None , "Test": None}



    def forward(self, imgs):
        if self.model_name in  ['CANet2D', 'CANet2D_stacked']:
            output, attention_coefs = self.model(imgs)
        else:
            output = self.model(imgs)
            attention_coefs = None
        return output, attention_coefs


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
        gt_seg = batch['gt_myo']
        spacing = batch['spacing']
        img_path = batch['img_path']

        if self.model_name == 'UNet3D_channels':
            input = image.squeeze(1)
            gt_seg = gt_seg.squeeze(1)
        elif 'cascaded' in self.model_name:
            input = torch.stack([image.squeeze(1), batch['pred_myo'].squeeze(1)], dim=1)
        else:
            input = image
        # forward model
        output, attention_coefs = self.forward(input.float())

        # compute loss
        loss = self.loss_function(output, gt_seg.float())

        batch_dict = {
            "loss" : loss
        }

        if logging == True:
            output = output.detach()
            self.dice_smoothed[split](output, gt_seg)
            self.dice_unsmoothed[split](output, gt_seg)

            if (self.current_epoch + 1) % self.heavy_log == 0:
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
        example_gt_seg_batch = example_batch['gt_myo']
        example_img_path = example_batch['img_path']

        batch_dict, output = self._step(example_batch, split, logging=False)
        # depth dimension to batch dimension for visualization for 3D models
        if len(example_img_batch.shape) == 5:
            B, C, D, H, W = example_img_batch.shape
            example_img_batch = example_img_batch.view(B*D, C, H, W)
            example_gt_seg_batch = example_gt_seg_batch.view(B*D, C, H, W)
            output = output.view(B*D, C, H, W)

            plot_preliminary_myo = False
            if plot_preliminary_myo == True:
                preliminary_myo = example_batch['pred_myo'].view(B*D, C, H, W)
                grid_preliminary_myo = torchvision.utils.make_grid(preliminary_myo, nrow=4, padding=2).cpu().detach()
                grid_preliminary_myo_masked = np.ma.masked_where(grid_preliminary_myo <= 0, grid_preliminary_myo)

        # image
        grid_imgs = torchvision.utils.make_grid(example_img_batch, nrow=4, padding=2).cpu().detach()
        # ground truth myocardium segmentation
        grid_gt_seg = torchvision.utils.make_grid(example_gt_seg_batch, nrow=4, padding=2).cpu().detach()
        grid_gt_seg_masked = np.ma.masked_where(grid_gt_seg <= 0, grid_gt_seg)
        # myocardium segmentation predictions
        grid_output = torchvision.utils.make_grid(output, nrow=4, padding=2).cpu().detach()
        grid_output_masked = np.ma.masked_where(grid_output <= 0, grid_output)

        fig, axs = plt.subplots(1, 3, figsize=(16, 8))
        fig.colorbar(cm.ScalarMappable(cmap="coolwarm"), ax=axs.ravel().tolist())

        axs[0].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[0].set_title('Image')
        if plot_preliminary_myo == True:
            axs[0].imshow(grid_preliminary_myo_masked[0,:,:], cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)

        axs[1].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[1].imshow(grid_gt_seg[0,:,:], cmap="coolwarm", alpha=0.8)
        axs[1].set_title('Myo seg ground truth')

        axs[2].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[2].imshow(grid_output_masked[0,:,:], cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)
        axs[2].set_title('Myo prediction')

        axs[0].set_axis_off()
        axs[1].set_axis_off()
        axs[2].set_axis_off()

        self.logger.experiment.add_figure(f"Images/{split}", fig, self.current_epoch)
        # show a different sample each time
        self.example_batch[split] = None
        plt.show()
        return



    def _epoch_end(self, outputs, split):
        assert split in ["Train", "Val", "Test"]
        epoch_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f"Loss/{split}", epoch_loss, self.current_epoch)

        # log dice
        for metric in self.metrics:
            epoch_m = metric[split].compute()
            metric[split].reset()
            self.logger.experiment.add_scalar(f"{metric[split].name}/{split}", epoch_m, self.current_epoch)

        # heavy log
        if (self.current_epoch + 1) % self.heavy_log == 0:
            self.log_images(split)



    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "Train")


    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "Val")


    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "Test")





if __name__ == '__main__':
    print("")
    m = MyoSegmentationModel()
