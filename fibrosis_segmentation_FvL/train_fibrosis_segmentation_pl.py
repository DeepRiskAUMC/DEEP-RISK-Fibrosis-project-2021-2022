import os
from datetime import datetime
import sys
from prettytable import PrettyTable
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loading.load_data import load_data
from utils_functions.criterions import Diceloss, smoothed_dice_score, get_accuracy
from utils_functions.utils import get_model_version_no
from models.segmentation_models import Floor_3D_half_Unet, Simple_2d_Unet, CA_2d_Unet, Floor_3D_full_Unet, Floor_2d_Unet

class SegmentationModel(pl.LightningModule):

    def __init__(self, model_name, n_classes, loss_function_string, lr, bilinear=True):
        super().__init__()
        self.model_name = model_name
        self.loss_function_string = loss_function_string
        self.test_val_mode = 'test'
        self.save_hyperparameters()
        if loss_function_string == 'dice':
            self.loss_function = Diceloss()
        else:
            raise ValueError(f"Loss function {loss_function_string} not known")

        if self.model_name == 'UNet2D':
            self.model = Simple_2d_Unet(1, n_classes, bilinear=bilinear, adapted=False)
            # print(count_parameters(self.model, self.model_name))
        elif self.model_name == 'UNet2D_stacked':
            self.model = Simple_2d_Unet(2, n_classes, bilinear=bilinear, adapted=False)
            # print(count_parameters(self.model, self.model_name))
        elif self.model_name == 'UNet2D_small':
            self.model = Simple_2d_Unet(1, n_classes, bilinear=bilinear, adapted=True)
            # print(count_parameters(self.model, self.model_name))
        elif self.model_name == 'UNet2D_stacked_small':
            self.model = Simple_2d_Unet(2, n_classes, bilinear=bilinear, adapted=True)
            # print(count_parameters(self.model, self.model_name))
        elif self.model_name == 'CANet2D':
            if bilinear:
                raise Exception('Upsample implementation not allowed for CANet2D')
            self.model = CA_2d_Unet(1, n_classes, bilinear=bilinear)
        elif self.model_name == 'CANet2D_stacked':
            if bilinear:
                raise Exception('Upsample implementation not allowed for CANet2D_stacked')
            self.model = CA_2d_Unet(2, n_classes, bilinear=bilinear)
        elif self.model_name == 'Floor_UNet2D':
            self.model = Floor_2d_Unet(1, n_classes, bilinear=bilinear)
            # print(count_parameters(self.model, self.model_name))
        elif self.model_name == 'Floor_UNet2D_stacked':
            self.model = Floor_2d_Unet(2, n_classes, bilinear=bilinear)
        elif self.model_name == 'UNet3D':
            self.model = Simple_3d_Unet(1, n_classes, bilinear=bilinear)
        elif self.model_name == 'UNet3D_channels':
            feat_mult = 2
            print('Feature multiplication factor:', feat_mult)
            self.model = Floor_2d_Unet(13, 13, bilinear=bilinear, feature_multiplication=feat_mult)   
        elif self.model_name == 'UNet3D_channels_stacked':
            self.model = Floor_2d_Unet(26, 13, bilinear=bilinear)
        elif self.model_name == 'UNet3D_half':
            self.model = Floor_3D_half_Unet(1, n_classes, bilinear=bilinear)
        else:
            raise ValueError(f"model_name  {model_name} not known")
        
    def forward(self, imgs):
        if self.model_name in  ['CANet2D', 'CANet2D_stacked']:
            output, attention_coefs = self.model(imgs)
        else:
            output = self.model(imgs)
            attention_coefs = None
        return output, attention_coefs
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10), "monitor": "val_loss"}
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        LGE_image, myo_pred, fibrosis_mask = batch[:3]
        batch_size = LGE_image.shape[0]
        if 'stacked' in self.model_name:
            input = torch.stack([LGE_image.squeeze(), myo_pred.squeeze()], dim=1)
        elif self.model_name == 'UNet3D_channels':
            input = LGE_image.squeeze(1)
            fibrosis_mask = fibrosis_mask.squeeze(1)
        else:
            input = LGE_image
        output, attention_coefs = self.forward(input.float())
        loss = self.loss_function(output, fibrosis_mask.float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        prediction = torch.round(output)
        self.log("train_dicescore", smoothed_dice_score(prediction, fibrosis_mask), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        LGE_image, myo_pred, fibrosis_mask = batch[:3]
        batch_size = LGE_image.shape[0]
        if 'stacked' in self.model_name:
            input = torch.stack([LGE_image.squeeze(), myo_pred.squeeze()], dim=1)
        elif self.model_name == 'UNet3D_channels':
            input = LGE_image.squeeze(1)
            fibrosis_mask = fibrosis_mask.squeeze(1)
        else:
            input = LGE_image
        output, attention_coefs = self.forward(input.float())
        loss = self.loss_function(output, fibrosis_mask.float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        prediction = torch.round(output)
        self.log("val_dicescore", smoothed_dice_score(prediction, fibrosis_mask), on_step=False, on_epoch=True, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        LGE_image, myo_pred, fibrosis_mask = batch[:3]
        batch_size = LGE_image.shape[0]
        if 'stacked' in self.model_name:
            input = torch.stack([LGE_image.squeeze(), myo_pred.squeeze()], dim=1)
        elif self.model_name == 'UNet3D_channels':
            input = LGE_image.squeeze(1)
            fibrosis_mask = fibrosis_mask.squeeze(1)
        else:
            input = LGE_image
        output, attention_coefs = self.forward(input.float())
        loss = self.loss_function(output, fibrosis_mask.float())
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        prediction = torch.round(output)
        self.log("test_dicescore", smoothed_dice_score(prediction, fibrosis_mask), on_step=False, on_epoch=True, batch_size=batch_size)
        
class GenerateCallback(pl.Callback):
    def __init__(self, batch_size=64, every_n_epochs=5, save_to_disk=False):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
    
    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        elogs = trainer.logged_metrics
        train_loss, val_loss = None, None
        for log_value in elogs:
            if 'train' in log_value and 'loss' in log_value:
                train_loss = elogs[log_value]
            elif 'val' in log_value and 'loss' in log_value:
                val_loss = elogs[log_value]
        print(f"Epoch ({trainer.current_epoch+1}/{trainer.max_epochs}: train loss = {train_loss} | val loss = {val_loss}")

def train(args):
    if ('2D' in args.dataset and '3D' in args.model) or ('3D' in args.dataset and '2D' in args.model):
        raise ValueError('Invalid combination of dataset and model. Should both be 2D or 3D.')
    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader, test_loader, loss_weights = load_data(dataset=args.dataset,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    transformations=args.transformations,
                                                    fibrosis_model=args.model,
                                                    myocard_model_version=args.version_myocard_preds,
                                                    use_only_fib = args.use_only_fib_slices,
                                                    resize=args.resize,
                                                    size=args.size,
                                                    normalize=args.normalize)
    gen_callback = GenerateCallback()
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss", save_last=True)
    logging_dir = os.path.join(args.log_dir, 'fibrosis')
    trainer = pl.Trainer(default_root_dir=logging_dir,
                         log_every_n_steps=5,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         callbacks=[checkpoint_callback, lr_monitor, gen_callback],
                         progress_bar_refresh_rate=1 if args.progress_bar else 0) 
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Create model
    pl.seed_everything(args.seed)  # To be reproducible
    bilinear = True if args.upsampling == 'upsample' else False
    if args.continue_from_path == 'None':
        model = SegmentationModel(model_name=args.model, n_classes=args.n_classes, loss_function_string=args.loss_function, lr=args.lr, bilinear=bilinear)
        trainer.fit(model, train_loader, val_loader)
    else:
        model = SegmentationModel.load_from_checkpoint(args.continue_from_path)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.model_path)
    
    #Testing
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)
    # test_dice, val_dice = evaluate(trainer, model, test_loader, val_loader)
    return test_result

def evaluate(trainer, model, test_dataloader, val_dataloader, loss_function):
    """
    Tests a model on test and validation set.
    Args:
        trainer (pl.Trainer) - Lightning trainer to use.
        model (DocumentClassifier) - The Lightning Module which should be used.
        test_dataloader (DataLoader) - Data loader for the test split.
        val_dataloader (DataLoader) - Data loader for the validation split.
    Returns:
        test_accuracy (float) - The achieved test accuracy.
        val_accuracy (float) - The achieved validation accuracy.
    """

    print('Testing model on validation and test ..........\n')

    test_start = time.time()

    model.test_val_mode = 'test'
    test_result = trainer.test(model, test_dataloaders=test_dataloader, verbose=False)[0]
    test_accuracy = test_result[f"test_{str(loss_function)}_dice"]

    model.test_val_mode = 'val'
    val_result = trainer.test(model, test_dataloaders=val_dataloader, verbose=False)[0]
    val_accuracy = val_result["test_accuracy"] if "val_accuracy" not in val_result else val_result["val_accuracy"]
    model.test_val_mode = 'test'

    test_end = time.time()
    test_elapsed = test_end - test_start

    print(f'\nRequired time for testing: {int(test_elapsed / 60)} minutes.\n')
    print(f'Test Results:\n test accuracy: {round(test_accuracy, 3)} ({test_accuracy})\n '
          f'validation accuracy: {round(val_accuracy, 3)} ({val_accuracy})'
          f'\n epochs: {trainer.current_epoch + 1}\n')

    return test_accuracy, val_accuracy

def count_parameters(model, model_name): 
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    encoder_params = 0
    decoder_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
        if (name.startswith('inc') or name.startswith('down')) and 'Floor' not in model_name:
            encoder_params+=params
        elif 'Floor' not in model_name:
            decoder_params+=params
        elif 'down' in name and 'Floor' in model_name:
            encoder_params+=params
        elif 'Floor' in model_name:
            decoder_params+=params

    print(table)
    print(f"Total Encoder Params: {encoder_params}")
    print(f"Total Decoder Params: {decoder_params}")
    print(f"Total Trainable Params: {total_params}")
    # return total_params

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='UNet2D', type=str,
                        help='What model to use for the segmentation',
                        choices=['UNet2D', 'UNet2D_stacked', 'UNet2D_masked', 'UNet2D_small', 'UNet2D_stacked_small', 'CANet2D', 'CANet2D_stacked', 'UNet3D_channels', 'UNet3D_half', 'UNet3D_full', 'Floor_UNet2D', 'Floor_UNet2D_stacked'])
    parser.add_argument('--n_classes', default=1, type=int,
                        help='Number of classes.')
    parser.add_argument('--upsampling', default='upsample', type=str,
                        help='What kind of model upsampling we want to use',
                        choices=['upsample', 'convtrans'])
    parser.add_argument('--continue_from_path', default='None', type=str,
                        help='Path to model checkpoint from which we want to continue training.')

    # Optimizer hyperparameters
    parser.add_argument('--loss_function', default='dice', type=str,
                        help='What loss funciton to use for the segmentation',
                        choices=['dice'])
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--dataset', default='AUMC2D', type=str,
                        help='What dataset to use for the segmentation',
                        choices=['AUMC2D', 'AUMC3D', 'Myops'])  
    parser.add_argument('--resize', default='resize', type=str,
                        help='Whether to resize all images to 256x256 or to crop images to the size of the smallest image width and height',
                        choices=['resize', 'crop'])    
    parser.add_argument('--size', default=['smallest'], nargs='+', type=str,
                        help='Shape to which the images need to be cropped. Elements of lists are Strings which are later converted to ints.')     
    parser.add_argument('--normalize', default=[], nargs='+', type=str,
                        help='Type of normalization thats performed on the data',
                        choices=['clip', 'scale_before_gamma'])     
    parser.add_argument('--use_only_fib_slices', default='no', type=str,
                        help='If we want to only use the 2D slices that contain fibrosis. Only useful when using dataset AUMC2D',
                        choices=['yes', 'no'])
    parser.add_argument('--version_myocard_preds', default=11, type=int,
                        help='Model version of which the myocard predictions should be used')  
    parser.add_argument('--transformations', nargs='*', default=['hflip', 'vflip', 'rotate', 'gamma_correction', 'shear'],
                        choices=['hflip', 'vflip', 'rotate', 'gamma_correction', 'shear'])
    parser.add_argument('--epochs', default=400, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--log_dir', default='segment_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    #write prints to file
    if args.continue_from_path == 'None':
        version_nr = get_model_version_no(args.log_dir, 'fibrosis')
    else:
        folders = args.continue_from_path.split('/')
        for folder in folders:
            if 'version' in folder:
                version_nr = int(folder.split('_')[-1])
    file_name = f'train_fibrosis_segmentation_version_{version_nr}.txt'
    first_path = os.path.join(args.log_dir, 'fibrosis', 'lightning_logs', file_name)
    second_path = os.path.join(args.log_dir, 'fibrosis', 'lightning_logs', f"version_{version_nr}", file_name)
    print('Segmentation training has started!')
    if args.continue_from_path == 'None':
        sys.stdout = open(first_path, "w")
    else:
        sys.stdout = open(first_path, 'a')
        print(f'\nResumed training from checkpoint: {args.continue_from_path}')
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print(f"Dataset: {args.dataset} | model: {args.model} | resize: {args.resize} | normalize: {args.normalize} | augmentation: {args.transformations} | Upsampling method: {args.upsampling} | only using fibrosis slices?: {args.use_only_fib_slices} | loss_function: {args.loss_function} | lr: {args.lr} | batch_size: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed} | version_no: {version_nr} | version_myocard_preds: {args.version_myocard_preds}")
    train(args)
    sys.stdout.close()
    sys.stdout = open("/dev/stdout", "w")
    os.rename(first_path, second_path)
    print('Segmentation completed')