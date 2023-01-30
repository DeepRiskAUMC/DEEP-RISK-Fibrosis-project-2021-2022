import os
import sys
from datetime import datetime
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loading.load_data import load_data
from utils_functions.criterions import Diceloss, smoothed_dice_score, Dice_WBCE_loss
from utils_functions.utils import get_model_version_no
from models.segmentation_models import Simple_2d_Unet, CA_2d_Unet, Floor_2d_Unet, Floor_3D_half_Unet, Floor_3D_full_Unet
from train_myocard_segmentation_pl import SegmentationModel as MyocardModel

class SegmentationModel(pl.LightningModule):

    def __init__(self, model_name, n_classes, myo_checkpoint, loss_function_string, lr, bilinear=True, loss_weights=None, feature_multiplication=4):
        super().__init__()
        adapted = True
        self.model_name = model_name
        self.loss_function_string = loss_function_string
        # self.bilinear = bilinear
        self.save_hyperparameters()

        try:
            self.myo_model = MyocardModel.load_from_checkpoint(myo_checkpoint)
        except:
            myo_checkpoint_splits = myo_checkpoint.split('segment_logs/')
            print(myo_checkpoint_splits)
            if myo_checkpoint_splits[0] == '':
                print(1)
                myo_checkpoint_adapted_path = os.path.join('/home/flieshout/deep_risk_models/fibrosis_segmentation_FvL/outputs/segment_logs', myo_checkpoint_splits[1])
            else:
                print(2)
                myo_checkpoint_adapted_path = os.path.join(myo_checkpoint_splits[0], 'segment_logs', myo_checkpoint_splits[1])
            print(myo_checkpoint_adapted_path)
            self.myo_model = MyocardModel.load_from_checkpoint(myo_checkpoint_adapted_path)
        self.myo_model.freeze()
        self.sigmoid_finish = True
        self.feature_multiplication = feature_multiplication

        if loss_function_string == 'dice':
            self.loss_function = Diceloss()
        elif loss_function_string == 'dice+WCE':
            self.loss_function = Dice_WBCE_loss(torch.tensor(loss_weights[1]))
            self.sigmoid_finish = False
            self.sigmoid = torch.nn.Sigmoid()
        else:
            raise ValueError(f"Loss function {loss_function_string} not known")

        if self.model_name == 'UNet2D_stacked':
            self.fib_model = Simple_2d_Unet(2, n_classes, bilinear=bilinear, adapted=adapted)
            # print(count_parameters(self.fib_model, self.model_name))
        elif self.model_name == 'CANet2D_stacked':
            if bilinear:
                raise Exception('Upsample implementation not allowed for CANet2D_stacked')
            self.fib_model = CA_2d_Unet(2, n_classes, bilinear=bilinear)
        elif self.model_name == 'Floor_UNet2D_stacked':
            self.fib_model = Floor_2d_Unet(2, n_classes, bilinear=bilinear, sigmoid_finish=self.sigmoid_finish, feature_multiplication=self.feature_multiplication)
        elif self.model_name == 'UNet3D_channels_stacked':
            feat_mult = 2
            print('Feature multiplication factor:', feat_mult)
            self.fib_model = Floor_2d_Unet(2*13, 13, bilinear=bilinear, sigmoid_finish=self.sigmoid_finish, feature_multiplication=self.feature_multiplication)
        elif self.model_name == 'UNet3D_half_stacked':
            self.fib_model = Floor_3D_half_Unet(2, n_classes, bilinear=bilinear, feature_multiplication=self.feature_multiplication)
        elif self.model_name == 'UNet3D_full_stacked':
            self.fib_model = Floor_3D_full_Unet(2, n_classes, bilinear=bilinear, feature_multiplication=self.feature_multiplication)
        else:
            raise ValueError(f"model_name  {model_name} not known")

    def forward(self, imgs):
        if '2D' in self.model_name:
            myo_pred = self.myo_model(imgs).squeeze()
            imgs = imgs.squeeze()
            if myo_pred.dim() == 2:
                myo_pred = myo_pred.unsqueeze(dim=0)
            if imgs.dim() == 2:
                imgs = imgs.unsqueeze(dim=0)
        elif '3D' in self.model_name:
            imgs = imgs.squeeze()
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(dim=0)
            myo_pred = torch.zeros_like(imgs)
            for i in range(imgs.shape[1]):
                if torch.any(imgs[:,i,:,:] > 0):
                    slice = imgs[:,i,:,:].unsqueeze(dim=1)
                    myo_pred_slice = self.myo_model(slice).squeeze()
                    myo_pred[:,i,:,:] = myo_pred_slice
        if self.model_name == 'UNet3D_channels_stacked':
            input = torch.cat([imgs, myo_pred], dim=1)
        else:
            input = torch.stack([imgs, myo_pred], dim=1)
        if self.model_name in  ['CANet2D_stacked']:
            output, attention_coefs = self.fib_model(input)
        else:
            output = self.fib_model(input)
            attention_coefs = None
        return output, attention_coefs
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True), "monitor": "val_loss"}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        LGE_image, _, fibrosis_mask = batch[:3]
        # save_image(fibrosis_mask, os.path.join(f"test_fibrosis_mask_{pat_id[0]}_slice{slice_nr.item()}.png"))
        batch_size = LGE_image.shape[0]

        input = LGE_image.float()
        output, attention_coefs = self.forward(input.float())
        loss = self.loss_function(output, fibrosis_mask.float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        if self.sigmoid_finish:
            prediction = torch.round(output)
        else:
            prediction = torch.round(self.sigmoid(output))
        if fibrosis_mask.dim() > prediction.dim():
            fibrosis_mask = fibrosis_mask.squeeze()
        self.log("train_dicescore", smoothed_dice_score(prediction, fibrosis_mask), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        LGE_image, _, fibrosis_mask = batch[:3]
        # save_image(fibrosis_mask, os.path.join(f"test_fibrosis_mask_{pat_id[0]}_slice{slice_nr.item()}.png"))
        batch_size = LGE_image.shape[0]

        input = LGE_image.float()
        output, attention_coefs = self.forward(input.float())
        loss = self.loss_function(output, fibrosis_mask.float())
        self.log("val_loss", loss, batch_size=batch_size)
        if self.sigmoid_finish:
            prediction = torch.round(output)
        else:
            prediction = torch.round(self.sigmoid(output))
        if fibrosis_mask.dim() > prediction.dim():
            fibrosis_mask = fibrosis_mask.squeeze()
        self.log("val_dicescore", smoothed_dice_score(prediction, fibrosis_mask), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        LGE_image, _, fibrosis_mask = batch[:3]
        batch_size = LGE_image.shape[0]

        input = LGE_image.float()
        output, attention_coefs = self.forward(input.float())
        loss = self.loss_function(output, fibrosis_mask.float())
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        if self.sigmoid_finish:
            prediction = torch.round(output)
        else:
            prediction = torch.round(self.sigmoid(output))
        if fibrosis_mask.dim() > prediction.dim():
            fibrosis_mask = fibrosis_mask.squeeze()
        self.log("test_dicescore", smoothed_dice_score(prediction, fibrosis_mask), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

class GenerateCallback(pl.Callback):
    def __init__(self, batch_size=64, every_n_epochs=5, save_to_disk=False):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.prev_train_loss = None
    
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
        if train_loss != self.prev_train_loss:
            print(f"Epoch ({trainer.current_epoch+1}/{trainer.max_epochs}): train loss = {train_loss} | val loss = {val_loss})")
        self.prev_train_loss = train_loss

def train(args):
    if ('2D' in args.dataset and '3D' in args.model) or ('3D' in args.dataset and '2D' in args.model):
        raise ValueError('Invalid combination of dataset and model. Should both be 2D or 3D.')
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
    checkpoint_callback = ModelCheckpoint(mode="min", monitor="val_loss", save_last=True)
    logging_dir = os.path.join(args.log_dir, 'fibrosis')
    trainer = pl.Trainer(default_root_dir=logging_dir,
                         log_every_n_steps=5,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         callbacks=[checkpoint_callback, lr_monitor, gen_callback],
                         enable_progress_bar = False) 
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Create model
    pl.seed_everything(args.seed)  # To be reproducible   
    bilinear = True if args.upsampling == 'upsample' else False  
    if args.continue_from_path == 'None':   
        model = SegmentationModel(model_name=args.model, n_classes=args.n_classes, myo_checkpoint=args.myo_checkpoint, loss_function_string=args.loss_function, lr=args.lr, bilinear=bilinear, loss_weights=loss_weights, feature_multiplication=args.feature_multiplication)
        trainer.fit(model, train_loader, val_loader)
    else:
        print(f'Continued training from checkpoint {args.continue_from_path}')
        model = SegmentationModel.load_from_checkpoint(args.continue_from_path)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.continue_from_path)
    
    #Testing
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    final_val_result = trainer.validate(model, val_loader, verbose=True)
    # test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)
    # test_dice, val_dice = evaluate(trainer, model, test_loader, val_loader)
    return final_val_result

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='UNet3D_half_stacked', type=str,
                        help='What model to use for the segmentation',
                        choices=['UNet2D_stacked', 'CANet2D_stacked', 'Floor_UNet2D_stacked', 'UNet3D_channels_stacked', 'UNet3D_half_stacked', 'UNet3D_full_stacked'])
    # parser.add_argument('--in_channels', default=1, type=int,
    #                     help='Number of input channels for the convolutional networks.')
    parser.add_argument('--n_classes', default=1, type=int,
                        help='Number of classes.')
    parser.add_argument('--upsampling', default='upsample', type=str,
                        help='What kind of model upsampling we want to use',
                        choices=['upsample', 'convtrans'])
    parser.add_argument('--feature_multiplication', default='4', type=int,
                        help='The factor by which the number of features in the model are is multiplied')
    parser.add_argument('--continue_from_path', default='None', type=str,
                        help='Path to model checkpoint from which we want to continue training.')

    # Optimizer hyperparameters
    parser.add_argument('--loss_function', default='dice+WCE', type=str,
                        help='What loss funciton to use for the segmentation',
                        choices=['dice', 'dice+WCE'])
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--dataset', default='AUMC3D', type=str,
                        help='What dataset to use for the segmentation',
                        choices=['AUMC2D', 'AUMC3D', 'Myops'])  
    parser.add_argument('--resize', default='resize', type=str,
                        help='Whether to resize all images to 256x256 or to crop images to the size of the smallest image width and height',
                        choices=['resize', 'crop'])    
    parser.add_argument('--size', default=['smallest'], nargs='+', type=str,
                        help='Shape to which the images need to be cropped. Elements of lists are Strings which are later converted to ints.')
    parser.add_argument('--normalize', default=['clip', 'scale_before_gamma'], nargs='+', type=str,
                        help='Type of normalization thats performed on the data',
                        choices=['clip', 'scale_before_gamma'])
    parser.add_argument('--myo_checkpoint', default='segment_logs\\myocard\\lightning_logs\\version_6\\checkpoints\\epoch=399-step=9999.ckpt', type=str,
                        help='Path to model checkpoint for the myocard segmentations')  
    parser.add_argument('--transformations', nargs='*', default=['hflip', 'vflip', 'rotate'],
                        choices=['hflip', 'vflip', 'rotate', 'gamma_correction', 'scale_after_gamma'])
    parser.add_argument('--epochs', default=400, type=int,
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
    print(f"Dataset: {args.dataset} | model: {args.model} | resize: {args.resize} | normalize: {args.normalize} | augmentation: {args.transformations} | Upsampling method: {args.upsampling} | myo checkpoint: {args.myo_checkpoint} | loss_function: {args.loss_function} | lr: {args.lr} | batch_size: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed} | version_no: {version_nr}")
    final_val_result = train(args)
    sys.stdout.close()
    sys.stdout = open("/dev/stdout", "w")
    os.rename(first_path, second_path)
    print('Segmentation completed')