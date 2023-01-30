import os
import sys
import gc
from datetime import datetime
import argparse
from utils_functions.utils import get_model_version_no_classification, load_pretrained_layers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from train_fibrosis_segmentation_with_myo_pl import SegmentationModel
from models.classification_models import Simple_CNN_clinical
from data_loading.load_classification_data import load_classification_data_clinical
from models.segmentation_models import Floor_2d_Unet_without_final_conv

class ClassificationModel(pl.LightningModule):
    def __init__(self, model_name, n_classes, myo_checkpoint, fibrosis_checkpoint, loss_function_string, lr, prediction_task, extra_input, hidden_layers_conv=[32,32], hidden_layers_lin=[64], train_loss_weights=None, val_loss_weights=None, use_val_weights=False, use_MRI_features=False, cross_validate=False):
        super().__init__()
        self.model_name = model_name
        self.save_hyperparameters()
        self.sigmoid_finish = True
        self.prediction_task = prediction_task
        self.use_val_weights = use_val_weights
        self.extra_input = extra_input
        self.cross_validate = cross_validate
        self.use_MRI_features = use_MRI_features
        if use_MRI_features:
            print('Using MRI features')
        else:
            print('Not using MRI features')
        self.train_y_pred, self.train_y_t = None, None
        self.validation_y_pred, self.validation_y_t = None, None
        self.test_y_pred, self.test_y_t = None, None

        if loss_function_string == 'BCEwithlogits':
            self.train_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(train_loss_weights))
            if self.use_val_weights:
                print('using validation weights')
                self.val_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(val_loss_weights))
            else:
                self.val_loss_function = torch.nn.BCEWithLogitsLoss()
            self.sigmoid_finish = False
            self.sigmoid = torch.nn.Sigmoid()
        else:
            raise ValueError(f"Loss function {loss_function_string} not known")

        full_segmentation_model = SegmentationModel.load_from_checkpoint(fibrosis_checkpoint)
        self.segment_model_name = full_segmentation_model.model_name
        self.myo_model = full_segmentation_model.myo_model
        full_fib_model = full_segmentation_model.fib_model

        self.bilinear = full_fib_model.bilinear
        self.feature_multiplication = full_fib_model.feature_multiplication
        
        fib_model_name = full_fib_model._get_name()
        if fib_model_name == 'Floor_2d_Unet':
            self.fib_model = Floor_2d_Unet_without_final_conv(2, 1, bilinear=self.bilinear, feature_multiplication=self.feature_multiplication)
            self.fib_model = load_pretrained_layers(self.fib_model, full_fib_model)

        self.myo_model.freeze()
        self.fib_model.requires_grad_(False)
        del full_fib_model

        self.segment_model_output_size = 16 * self.feature_multiplication
        if 'LGE_image' in extra_input:
            self.LGE_conv_layer = torch.nn.Conv3d(1, self.segment_model_output_size, kernel_size=1)
        if 'myocard' in extra_input:
            self.myocard_conv_layer = torch.nn.Conv3d(1, self.segment_model_output_size, kernel_size=1)
        count_extra_input = len(extra_input) + 1
        self.classification_model = Simple_CNN_clinical(self.segment_model_output_size*count_extra_input, cnn_channels=hidden_layers_conv, lin_channels=hidden_layers_lin, sigmoid_finish=self.sigmoid_finish, use_MRI_features=self.use_MRI_features)

    def forward(self, imgs, clinical_features):
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(dim=0)
        myo_pred = torch.zeros_like(imgs)
        for i in range(imgs.shape[2]):
            if torch.any(imgs[:,:,i,:,:] > 0):
                LGE_slice = imgs[:,:,i,:,:]
                myo_pred_slice = self.myo_model(LGE_slice)
                myo_pred[:,:,i,:,:] = myo_pred_slice
                del myo_pred_slice
        
        if '2D' in self.segment_model_name:
            try:
                input_datatype = imgs.dtype
            except:
                input_datatype = imgs.type()
            device = 'cuda' if imgs.is_cuda else 'cpu'
            fib_pred = torch.zeros((imgs.shape[0], self.segment_model_output_size, imgs.shape[2], imgs.shape[3], imgs.shape[4]), dtype=input_datatype, device=device)
            for i in range(imgs.shape[2]):
                if torch.any(imgs[:,:,i,:,:] > 0):
                    LGE_slice = imgs[:,:,i,:,:]
                    myo_slice = myo_pred[:,:,i,:,:]
                    concat_slice = torch.cat([LGE_slice, myo_slice], dim=1)
                    fib_pred_slice = self.fib_model(concat_slice)
                    fib_pred[:,:,i,:,:] = fib_pred_slice
                    del fib_pred_slice

        concat_input = fib_pred
        if 'myocard' in self.extra_input:
            projected_myocard = self.myocard_conv_layer(myo_pred)
            concat_input = torch.cat([projected_myocard, concat_input], dim=1)
        if 'LGE_image' in self.extra_input:
            projected_LGE_imgs = self.LGE_conv_layer(imgs)
            concat_input = torch.cat([projected_LGE_imgs, concat_input], dim=1)
        
        output = self.classification_model(concat_input, clinical_features)
        return output.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # if self.cross_validate:
        #     scheduler = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)}
        # else:
        #     scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True), "monitor": "val_loss"}
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True), "monitor": "val_loss"}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        LGE_imgs, clinical, labels, pat_ids = batch
        if self.prediction_task == 'ICD_therapy':
            labels = labels[0]
        elif self.prediction_task == 'ICD_therapy_365days':
            labels = labels[1]
        elif self.prediction_task == 'mortality':
            labels = labels[2]
        elif self.prediction_task == 'mortality_365days':
            labels = labels[3]
        # print('batch_size =', LGE_imgs.shape)
        output = self.forward(LGE_imgs.float(), clinical)
        loss = self.train_loss_function(output, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=LGE_imgs.shape[0])
        if self.sigmoid_finish:
            prediction = torch.round(output)
        else:
            output = self.sigmoid(output)
            prediction = torch.round(output)
        accuracy = ((prediction == labels).float().sum()/len(labels))
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, batch_size=LGE_imgs.shape[0])
        if self.train_y_pred is None:
            self.train_y_pred, self.train_y_t = output.cpu().detach(), labels.cpu().detach()
        else:
            self.train_y_pred = torch.cat((self.train_y_pred, output.cpu().detach()), 0)
            self.train_y_t = torch.cat((self.train_y_t, labels.cpu().detach()), 0)
        return loss

    def validation_step(self, batch, batch_idx):
        LGE_imgs, clinical, labels, pat_ids = batch
        if self.prediction_task == 'ICD_therapy':
            labels = labels[0]
        elif self.prediction_task == 'ICD_therapy_365days':
            labels = labels[1]
        elif self.prediction_task == 'mortality':
            labels = labels[2]
        elif self.prediction_task == 'mortality_365days':
            labels = labels[3]
        output = self.forward(LGE_imgs.float(), clinical)
        loss = self.val_loss_function(output, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=LGE_imgs.shape[0])
        if self.sigmoid_finish:
            prediction = torch.round(output)
        else:
            output = self.sigmoid(output)
            prediction = torch.round(output)
        accuracy = ((prediction == labels).float().sum()/len(labels))
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, batch_size=LGE_imgs.shape[0])
        if self.validation_y_pred is None:
            self.validation_y_pred, self.validation_y_t = output.cpu().detach(), labels.cpu().detach()
        else:
            self.validation_y_pred = torch.cat((self.validation_y_pred, output.cpu().detach()), 0)
            self.validation_y_t = torch.cat((self.validation_y_t, labels.cpu().detach()), 0)
        return loss

    def test_step(self, batch, batch_idx):
        LGE_imgs, clinical, labels, pat_ids = batch
        if self.prediction_task == 'ICD_therapy':
            labels = labels[0]
        elif self.prediction_task == 'ICD_therapy_365days':
            labels = labels[1]
        elif self.prediction_task == 'mortality':
            labels = labels[2]
        elif self.prediction_task == 'mortality_365days':
            labels = labels[3]
        output = self.forward(LGE_imgs.float(), clinical)
        loss = self.val_loss_function(output, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=LGE_imgs.shape[0])
        if self.sigmoid_finish:
            prediction = torch.round(output)
        else:
            output = self.sigmoid(output)
            prediction = torch.round(output)
        accuracy = ((prediction == labels).float().sum()/len(labels))
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, batch_size=LGE_imgs.shape[0])
        if self.test_y_pred is None:
            self.test_y_pred, self.test_y_t = output.cpu().detach(), labels.cpu().detach()
        else:
            self.test_y_pred = torch.cat((self.test_y_pred, output.cpu().detach()), 0)
            self.test_y_t = torch.cat((self.test_y_t, labels.cpu().detach()), 0)
        return accuracy


class GenerateCallback(pl.Callback):
    def __init__(self, version_nr):
        super().__init__()
        self.version_nr = version_nr
        print('version_nr', self.version_nr)
        self.prev_train_loss = None
        self.prev_test_loss = None
        self.patience = 20
        self.best_auc = 0
        self.prev_checkpoint_name = None
        self.final_check = False

    def on_train_epoch_end(self, trainer, pl_module):
        print('finished training epoch')

    def on_validation_epoch_end(self, trainer, pl_module):
        print('finished validation epoch')
        train_loss, val_loss, val_auc = None, None, None
        elogs = trainer.logged_metrics
        for log_value in elogs:
            if 'train' in log_value and 'loss' in log_value:
                train_loss = elogs[log_value]
            elif 'val' in log_value and 'loss' in log_value:
                val_loss = elogs[log_value]
        if train_loss is not None:
            print(f"Epoch ({trainer.current_epoch}/{trainer.max_epochs}: train loss = {train_loss} | val loss = {val_loss})")
            #save AUC for train and validation
            train_auc = roc_auc_score(pl_module.train_y_t.numpy(), pl_module.train_y_pred.numpy())
            pl_module.log("train_AUC", train_auc)
            val_auc = roc_auc_score(pl_module.validation_y_t.numpy(), pl_module.validation_y_pred.numpy())
            print(f'validation auc: {val_auc}')
            pl_module.log("val_AUC", val_auc)
            pl_module.train_y_pred, pl_module.train_y_t, pl_module.validation_y_t, pl_module.validation_y_pred = None, None, None, None
        # save the model with the highest validation auc after 20 epochs
        if (trainer.current_epoch > self.patience and val_auc is not None) or self.final_check:
            if self.final_check:
                val_auc = roc_auc_score(pl_module.validation_y_t.numpy(), pl_module.validation_y_pred.numpy())
                print(f'validation auc: {val_auc}')
            if val_auc > self.best_auc:
                print('saved AUC model with val_auc:', val_auc)
                self.best_auc = val_auc
                checkpoint_filename = os.path.join(trainer.default_root_dir, 'lightning_logs', f'version_{self.version_nr}', 'checkpoints', f"val_auc_epochs={trainer.current_epoch}.ckpt")
                if checkpoint_filename != self.prev_checkpoint_name:
                    trainer.save_checkpoint(checkpoint_filename)
                    # print(f'checkpoint: {checkpoint_filename} saved!')
                    if self.prev_checkpoint_name is not None:
                        os.remove(self.prev_checkpoint_name)
                    self.prev_checkpoint_name = checkpoint_filename

    def on_test_epoch_end(self, trainer, pl_module):
        test_loss = None
        elogs = trainer.logged_metrics
        for log_value in elogs:
            if 'test' in log_value and 'loss' in log_value:
                test_loss = elogs[log_value]
        if test_loss is not None:
            print('labels / predictions:')
            print(pl_module.test_y_t.numpy())
            print(pl_module.test_y_pred.numpy())
            test_auc = roc_auc_score(pl_module.test_y_t.numpy(), pl_module.test_y_pred.numpy())
            pl_module.log("test_AUC", test_auc)
            pl_module.test_y_t, pl_module.test_y_pred = None, None
        self.prev_test_loss = test_loss
    
def train(args):
    # print(torch.cuda.memory_summary())
    if args.cross_validate:
        logging_dir = os.path.join(args.log_dir, args.model, args.split_name)
    else:
        logging_dir = os.path.join(args.log_dir, args.model) 
    os.makedirs(logging_dir, exist_ok=True)

    try:
        version_nr = args.version_nr
        version_nr_2 = get_model_version_no_classification(logging_dir)
        if version_nr != version_nr_2:
            version_nr = version_nr_2
    except:
        version_nr = get_model_version_no_classification(logging_dir)
        args.version_nr = version_nr
    train_loader, val_loader, _, (train_loss_weights, val_loss_weights) = load_classification_data_clinical(args.dataset,
                                                            batch_size=args.batch_size,
                                                            val_batch_size='same',
                                                            num_workers=args.num_workers,
                                                            only_test=False,
                                                            transformations=args.transformations,
                                                            resize=args.resize,
                                                            size = args.size,
                                                            normalize=args.normalize,
                                                            cross_validation=args.cross_validate)
    gen_callback = GenerateCallback(args.version_nr)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(mode="min", monitor="val_loss", save_last=True)
    trainer = pl.Trainer(default_root_dir=logging_dir,
                         log_every_n_steps=5,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         callbacks=[checkpoint_callback, lr_monitor, gen_callback],
                         enable_progress_bar = False)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Create model
    pl.seed_everything(args.seed)  # To be reproducible   
    if args.continue_from_path == 'None':
        if args.prediction_task == 'ICD_therapy':
            train_loss_weights = train_loss_weights[0]
        elif args.prediction_task == 'ICD_therapy_365days':
            train_loss_weights = train_loss_weights[1]
        elif args.prediction_task == 'mortality':
            train_loss_weights = train_loss_weights[2]
        elif args.prediction_task == 'mortality_365days':
            train_loss_weights = train_loss_weights[3]
        model = ClassificationModel(args.model, 
                                    1, 
                                    None, 
                                    args.fib_checkpoint, 
                                    args.loss_function, 
                                    args.lr, 
                                    prediction_task=args.prediction_task, 
                                    extra_input=args.extra_input, 
                                    hidden_layers_conv=args.hidden_layers_conv, 
                                    hidden_layers_lin=args.hidden_layers_lin, 
                                    train_loss_weights=train_loss_weights,
                                    val_loss_weights=val_loss_weights,
                                    use_val_weights=args.use_val_weights,
                                    use_MRI_features=args.use_MRI_features,
                                    cross_validate=args.cross_validate)
        trainer.fit(model, train_loader, val_loader)
    else:
        print(f'Continued training from checkpoint {args.continue_from_path}')
        model = ClassificationModel.load_from_checkpoint(args.continue_from_path)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.continue_from_path)

    #final validation round
    gen_callback.final_check = True
    last_validation_results = trainer.validate(model, val_loader, verbose=True)

    #Testing
    print(f'loading checkpoint {trainer.checkpoint_callback.best_model_path}')
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    best_val_result = trainer.test(model, val_loader, verbose=True)
    print(f'loading checkpoint {trainer.checkpoint_callback.last_model_path}')
    model = model.load_from_checkpoint(trainer.checkpoint_callback.last_model_path)
    last_val_result = trainer.test(model, val_loader, verbose=True)
    print(f'loading checkpoint {gen_callback.prev_checkpoint_name}')
    model = model.load_from_checkpoint(gen_callback.prev_checkpoint_name)
    auc_val_result = trainer.test(model, val_loader, verbose=True)

    return auc_val_result

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='CNN', type=str,
                        help='What model to use for the segmentation',
                        choices=['CNN'])
    parser.add_argument('--prediction_task', default='None', type=str,
                        help='Task to predict.',
                        choices=['ICD_therapy', 'ICD_therapy_365days', 'mortality', 'mortality_365days'])
    parser.add_argument('--extra_input', default=[], nargs='*', type=str,
                        help='What extra input to use for the classification model',
                        choices=['LGE_image', 'myocard'])
    parser.add_argument('--use_MRI_features', default='True', type=str,
                        help='Whether to use MRI features for the prediction or not',
                        choices=['True', 'False'])
    parser.add_argument('--hidden_layers_conv', default=[16,32,64,128], nargs='+', type=int,
                        help='Number of channels for the hidden convolutional layers of the classifier network.')
    parser.add_argument('--hidden_layers_lin', default=[64], nargs='+', type=int,
                        help='Number of channels for the hidden linear layers of the classifier network.')
    parser.add_argument('--continue_from_path', default='None', type=str,
                        help='Path to model checkpoint from which we want to continue training.')

    # Optimizer hyperparameters
    parser.add_argument('--loss_function', default='BCEwithlogits', type=str,
                        help='What loss funciton to use for the segmentation',
                        choices=['BCEwithlogits'])
    parser.add_argument('--use_val_weights', default='False', type=str,
                        help='If True, uses seperate loss weights for the validation loss function. If False, uses no weights for the validation loss.',
                        choices=['True',  'False'])
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--dataset', default='AUMC3D_ICM', type=str,
                        help='What dataset to use for the segmentation',
                        choices=['AUMC3D', 'AUMC3D_version2', 'AUMC3D_version3', 'AUMC3D_ICM'])  
    parser.add_argument('--resize', default='crop', type=str,
                        help='Whether to resize all images to 256x256 or to crop images to the size of the smallest image width and height',
                        choices=['resize', 'crop'])    
    parser.add_argument('--size', default=['132', '132'], nargs='+', type=str,
                        help='Shape to which the images need to be cropped. Elements of lists are Strings which are later converted to ints.')
    parser.add_argument('--normalize', default=['clip', 'scale_before_gamma'], nargs='+', type=str,
                        help='Type of normalization thats performed on the data',
                        choices=['clip', 'scale_before_gamma'])
    parser.add_argument('--fib_checkpoint', default=r'segment_logs\fibrosis\lightning_logs\version_61\checkpoints\epoch=113-step=8663.ckpt', type=str,
                        help='Path to model checkpoint for the fibrosis segmentations')  
    parser.add_argument('--transformations', nargs='*', default=['hflip', 'vflip', 'rotate'],
                        choices=['hflip', 'vflip', 'rotate', 'gamma_correction', 'scale_after_gamma'])
    parser.add_argument('--epochs', default=200, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--log_dir', default='outputs/classification_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')

    args = parser.parse_args()

    args.use_MRI_features = True if args.use_MRI_features == 'True' else False
    args.use_val_weights = True if args.use_val_weights == 'True' else False

    #write prints to file
    if args.continue_from_path == 'None':
        version_nr = get_model_version_no_classification(os.path.join(args.log_dir, args.model))
    else:
        folders = args.continue_from_path.split('/')
        for folder in folders:
            if 'version' in folder:
                version_nr = int(folder.split('_')[-1])
    file_name = f'train_classification_version_{version_nr}.txt'
    first_path = os.path.join(args.log_dir, args.model, 'lightning_logs', file_name)
    second_path = os.path.join(args.log_dir, args.model, 'lightning_logs', f"version_{version_nr}", file_name)
    print('Classification training has started!')
    if args.continue_from_path == 'None':
        sys.stdout = open(first_path, "w")
    else:
        sys.stdout = open(first_path, 'a')
        print(f'\nResumed training from checkpoint: {args.continue_from_path}')
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print(f"Dataset: {args.dataset} | model: {args.model} | layers_conv: {args.hidden_layers_conv} | layers_lin: {args.hidden_layers_lin} | resize: {args.resize} | normalize: {args.normalize} | augmentation: {args.transformations} | segmentation checkpoint: {args.fib_checkpoint} | loss_function: {args.loss_function} | use_val_weights: {args.use_val_weights} | lr: {args.lr} | batch_size: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed} | version_no: {version_nr}")
    final_val_result = train(args)
    sys.stdout.close()
    sys.stdout = open("/dev/stdout", "w")
    os.rename(first_path, second_path)
    print('Classification completed')
