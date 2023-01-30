import os
from datetime import datetime
import sys
import argparse
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loading.load_data import load_data
from utils_functions.criterions import Diceloss, smoothed_dice_score, WeightedDiceLoss, AdaptiveWeightedDiceLoss
from utils_functions.utils import get_model_version_no
from models.segmentation_models import Simple_2d_Unet, Floor_2d_Unet

class SegmentationModel(pl.LightningModule):
    def __init__(self, model_name, in_channels, n_classes, loss_function_string, lr, bilinear=True, feature_multiplication=4):
        super().__init__()
        self.loss_function_string = loss_function_string
        self.feature_multiplication = feature_multiplication
        self.lr = lr
        self.test_val_mode = 'test'
        self.model_name = model_name
        self.save_hyperparameters()
        if loss_function_string == 'dice':
            self.loss_function = Diceloss()
        elif loss_function_string == 'weighted_dice':
            self.loss_function = WeightedDiceLoss()
        elif loss_function_string == 'adaptive_weighted_dice':
            self.alpha = 0.0
            self.loss_function = AdaptiveWeightedDiceLoss()
        else:
            raise ValueError(f"Loss function {loss_function_string} not known")
        if model_name == 'UNet2D':
            self.model = Simple_2d_Unet(in_channels, n_classes, bilinear=bilinear)
        elif model_name == 'Floor_UNet2D':
            self.model = Floor_2d_Unet(in_channels, n_classes, bilinear=bilinear, feature_multiplication=self.feature_multiplication)
        elif model_name == 'UNet3D':
            raise ValueError(f"Model {model_name} not implemented yet")
        else:
            raise ValueError(f"model_name  {model_name} not known")

    def change_alpha(self):
        if self.alpha < 0.5:
            self.alpha = self.alpha + 0.05
        
    def forward(self, imgs):
        output = self.model(imgs)
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True), "monitor": "val_loss"}
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        LGE_image, myo_mask, fib_mask, _, _ = batch
        batch_size = LGE_image.shape[0]
        output = self.forward(LGE_image.float())
        if self.loss_function_string == 'weighted_dice':
            loss = self.loss_function(output, myo_mask.float(), fib_mask.float())
        elif self.loss_function_string == 'adaptive_weighted_dice':
            loss = self.loss_function(output, myo_mask.float(), fib_mask.float(), self.alpha)
        else:
            loss = self.loss_function(output, myo_mask.float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        prediction = torch.round(output)
        self.log("train_dicescore", smoothed_dice_score(prediction, myo_mask), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        LGE_image, myo_mask, fib_mask, _, _ = batch
        batch_size = LGE_image.shape[0]
        output = self.forward(LGE_image.float())
        if self.loss_function_string == 'weighted_dice':
            loss = self.loss_function(output, myo_mask.float(), fib_mask.float())
        elif self.loss_function_string == 'adaptive_weighted_dice':
            loss = self.loss_function(output, myo_mask.float(), fib_mask.float(), self.alpha)
        else:
            loss = self.loss_function(output, myo_mask.float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        prediction = torch.round(output)
        self.log("val_dicescore", smoothed_dice_score(prediction, myo_mask), on_step=False, on_epoch=True, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        LGE_image, myo_mask, fib_mask, _, _ = batch
        batch_size = LGE_image.shape[0]
        output = self.forward(LGE_image.float())
        if self.loss_function_string == 'weighted_dice':
            loss = self.loss_function(output, myo_mask.float(), fib_mask.float())
        elif self.loss_function_string == 'adaptive_weighted_dice':
            loss = self.loss_function(output, myo_mask.float(), fib_mask.float(), self.alpha)
        else:
            loss = self.loss_function(output, myo_mask.float())
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        prediction = torch.round(output)
        self.log("test_dicescore", smoothed_dice_score(prediction, myo_mask), on_step=False, on_epoch=True, batch_size=batch_size)
        
class GenerateCallback(pl.Callback):
    def __init__(self, minimal_epochs=0, epoch_interval=1):
        super().__init__()
        self.current_epoch = None
        self.train = False
        self.minimal_epochs = minimal_epochs
        self.epoch_interval = epoch_interval
        self.count_decrease = 0
        self.previous_val_loss = None
    
    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if self.train == False:
            self.train = True
            val_loss = None
        else:
            elogs = trainer.logged_metrics
            train_loss, val_loss = None, None
            for log_value in elogs:
                if 'train' in log_value and 'loss' in log_value:
                    train_loss = elogs[log_value]
                elif 'val' in log_value and 'loss' in log_value:
                    val_loss = elogs[log_value]
            print(f"Epoch ({trainer.current_epoch+1}/{trainer.max_epochs}): train loss = {train_loss} | val loss = {val_loss}")
            self.train = False
            if pl_module.loss_function_string == 'adaptive_weighted_dice' and trainer.current_epoch % 10 == 0 and trainer.current_epoch != 0:
                pl_module.change_alpha()
                print(f"Alpha has been changed to {pl_module.alpha}")

def train(args):

    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader, test_loader, loss_weights = load_data(dataset=args.dataset,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    transformations=args.transformations,
                                                    resize=args.resize,
                                                    size=args.size,
                                                    normalize=args.normalize)
    gen_callback = GenerateCallback()
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss", save_last=True)
    logging_dir = os.path.join(args.log_dir, 'myocard')
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
    model = SegmentationModel(model_name=args.model, in_channels=args.in_channels, n_classes=args.n_classes, loss_function_string=args.loss_function, lr=args.lr, bilinear=bilinear, feature_multiplication=args.feature_multiplication)
    trainer.fit(model, train_loader, val_loader)
    
    #Testing
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    val_result = trainer.test(model, val_loader, verbose=True)
    # test_dice, val_dice = evaluate(trainer, model, test_loader, val_loader)
    return val_result

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

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='Floor_UNet2D', type=str,
                        help='What model to use for the segmentation',
                        choices=['UNet2D', 'Floor_UNet2D', 'UNet3D', 'FCNN'])
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of input channels for the convolutional networks.')
    parser.add_argument('--n_classes', default=1, type=int,
                        help='Number of classes.')
    parser.add_argument('--upsampling', default='upsample', type=str,
                        help='What kind of model upsampling we want to use',
                        choices=['upsample', 'convtrans'])
    parser.add_argument('--feature_multiplication', default='4', type=int,
                        help='The factor by which the number of features in the model are is multiplied')

    # Optimizer hyperparameters
    parser.add_argument('--loss_function', default='dice', type=str,
                        help='What loss funciton to use for the segmentation',
                        choices=['dice', 'weighted_dice', 'adaptive_weighted_dice'])
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--dataset', default='AUMC2D', type=str,
                        help='What dataset to use for the segmentation',
                        choices=['AUMC2D', 'AUMC2D_30', 'AUMC3D', 'Myops'])  
    parser.add_argument('--resize', default='crop', type=str,
                        help='Whether to resize all images to 256x256 or to crop images to the size of the smallest image width and height',
                        choices=['resize', 'crop', 'none'])   
    parser.add_argument('--size', default=['176', '168'], nargs='+', type=str,
                        help='Shape to which the images need to be cropped. Elements of lists are Strings which are later converted to ints.')  
    parser.add_argument('--normalize', default=[], nargs='+', type=str,
                        help='Type of normalization thats performed on the data',
                        choices=['clip', 'scale_before_gamma'])        
    parser.add_argument('--transformations', nargs='*', default=['hflip', 'vflip', 'rotate'],
                        choices=['hflip', 'vflip', 'rotate'])
    parser.add_argument('--epochs', default=80, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--log_dir', default='outputs/segment_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    #write prints to file
    version_nr = get_model_version_no(args.log_dir)
    file_name = f'train_segmentation_version_{version_nr}.txt'
    first_path = os.path.join(args.log_dir, 'myocard', 'lightning_logs', file_name)
    second_path = os.path.join(args.log_dir, 'myocard', 'lightning_logs', f"version_{version_nr}", file_name)
    # if str(device) == 'cuda:0':
    #     sys.stdout = open(os.path.join(args.print_dir, file_name), "w")
    # else:
    #     file_name = file_name.replace(':', ';')
    #     sys.stdout = open(os.path.join('.', args.print_dir, file_name), "w")
    print('Segmentation training has started!')
    sys.stdout = open(first_path, "w")
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print(f"Dataset: {args.dataset} | model: {args.model} | resize: {args.resize} | normalize: {args.normalize} | augmentation: {args.transformations} | loss_function: {args.loss_function} | lr: {args.lr} | batch_size: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed} | version_no: {version_nr}")
    train(args)
    sys.stdout.close()
    sys.stdout = open("/dev/stdout", "w")
    os.rename(first_path, second_path)
    print('Segmentation completed')