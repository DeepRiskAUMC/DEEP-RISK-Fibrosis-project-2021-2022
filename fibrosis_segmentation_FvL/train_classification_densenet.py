import os
import sys
from datetime import datetime
import argparse
from utils_functions.utils import get_model_version_no_classification
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from train_myocard_segmentation_pl import SegmentationModel as MyocardModel
from models.classification_models import DenseNetPaddingClassificationModel
from data_loading.load_classification_data import load_classification_data_clinical
from torchvision import transforms

class ClassificationDenseNetModel(pl.LightningModule):
    def __init__(self, model_name, dropout, myo_checkpoint, loss_function_string, lr, prediction_task, train_loss_weights=None, val_loss_weights=None, use_val_weights=False, cross_validate=False):
        super().__init__()
        self.model_name = model_name
        self.save_hyperparameters()
        self.sigmoid_finish = True
        self.prediction_task = prediction_task
        self.use_val_weights = use_val_weights
        self.cross_validate = cross_validate
        self.dropout = dropout
        self.resize = transforms.Resize((100, 100))
        self.train_y_pred, self.train_y_t = None, None
        self.validation_y_pred, self.validation_y_t = None, None
        self.test_y_pred, self.test_y_t = None, None

        if loss_function_string == 'BCEwithlogits':
            self.train_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(train_loss_weights))
            if self.use_val_weights:
                self.val_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(val_loss_weights))
            else:
                self.val_loss_function = torch.nn.BCEWithLogitsLoss()
            self.sigmoid_finish = False
            self.sigmoid = torch.nn.Sigmoid()
        else:
            raise ValueError(f"Loss function {loss_function_string} not known")

        self.myo_model = MyocardModel.load_from_checkpoint(myo_checkpoint)
        self.myo_model.freeze()

        # add classification model
        if self.model_name == 'densenet_maxpool':
            self.model = DenseNetMaxpoolClassificationModel(1, n_classes=1, sigmoid_finish=self.sigmoid_finish)
        elif self.model_name == 'densenet_padding':
            self.model = DenseNetPaddingClassificationModel(1, n_classes=1, dropout=self.dropout, sigmoid_finish=self.sigmoid_finish)
        else:
            raise ValueError(f'Model name {model_name} not recognized')
    
    def forward(self, imgs):
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(dim=0)
        # get myocard predictions and crop the LGE img based on the myocard prediction
        myo_pred = torch.zeros_like(imgs).to(imgs.device)
        for i in range(imgs.shape[2]):
            if torch.any(imgs[:,:,i,:,:] > 0):
                LGE_slice = imgs[:,:,i,:,:]
                myo_pred_slice = self.myo_model(LGE_slice)
                myo_pred[:,:,i,:,:] = myo_pred_slice
                del myo_pred_slice
        myo_pred = torch.round(myo_pred)

        reshaped_tensors = torch.zeros((imgs.shape[0], 1, 13, 100, 100), dtype=float, requires_grad=True).to(imgs.device)
        for i in range(imgs.shape[0]):
            cropped_LGE_img = self.crop_single_img(imgs[i], myo_pred[i])
            resized_img = self.resize(cropped_LGE_img)
            reshaped_tensors[i] = resized_img

        #run through classification model
        output = self.model(reshaped_tensors.float())
        return output.squeeze()

    def remove_classifier(self):
        self.model.remove_classifier()

    def get_cropping_values(self, myo_pred):
        myo_pred = myo_pred.squeeze()
        x_min, y_min = 1e3, 1e3
        x_max, y_max = -1, -1
        for i in range(myo_pred.shape[0]):
            slice = myo_pred[i]
            if torch.any(slice > 0):
                # print('slice shape',slice.shape)
                # print(slice[:10, 0], slice[:10,-1])
                non_zero = (slice != 0).nonzero(as_tuple=True)
                # print(non_zero)
                x_min_slice = torch.amin(non_zero[1])
                x_max_slice = torch.amax(non_zero[1])
                y_min_slice = torch.amin(non_zero[0])
                y_max_slice = torch.amax(non_zero[0])
                # print(f'slice y_min, y_max, x_min, x_max: {[y_min_slice, y_max_slice, x_min_slice, x_max_slice]}')
                if y_max_slice > y_max: y_max = y_max_slice.item()
                if y_min_slice < y_min: y_min = y_min_slice.item()
                if x_max_slice > x_max: x_max = x_max_slice.item()
                if x_min_slice < x_min: x_min = x_min_slice.item()
        # print(f'y_min, y_max, x_min, x_max: {[y_min, y_max, x_min, x_max]}')
        if y_min == 1e3: y_min = 0
        if x_min == 1e3: x_min = 0
        if y_max == -1: y_max = myo_pred.shape[1]
        if x_max == -1: x_max = myo_pred.shape[2]
        return y_min, y_max, x_min, x_max
        
    def crop_single_img(self, LGE_img, myo_pred):
        # print(LGE_img.shape, myo_pred.shape)
        y_min, y_max, x_min, x_max = self.get_cropping_values(myo_pred)
        height, width = y_max-y_min, x_max-x_min
        extra_boundary = 5
        if height >= width:
            if height % 2 != 0:
                height = height + 1
            y_crop_start = y_min - extra_boundary
            y_crop_end = y_min + height + extra_boundary + 1
            x_middle = int((x_min + x_max)/2)
            x_crop_start = x_middle - int(height/2) - extra_boundary
            x_crop_end = x_middle + int(height/2) + extra_boundary + 1
        else:
            if width % 2 != 0:
                width = width + 1
            x_crop_start = x_min - extra_boundary
            x_crop_end = x_min + width + extra_boundary + 1
            y_middle = int((y_min + y_max)/2)
            y_crop_start = y_middle - int(width/2) - extra_boundary
            y_crop_end = y_middle + int(width/2) + extra_boundary + 1
        if y_crop_start < 0: y_crop_start = 0
        if x_crop_start < 0: x_crop_start = 0
        if y_crop_end > LGE_img.shape[2]: y_crop_end = LGE_img.shape[2]
        if x_crop_end > LGE_img.shape[3]: x_crop_end = LGE_img.shape[3]
        try:
            cropped_img = LGE_img[:, :, y_crop_start:y_crop_end, x_crop_start:x_crop_end]
        except:
            print(LGE_img.shape, myo_pred.shape)
            print(y_min, y_max, x_min, x_max)
            print(y_crop_start, y_crop_end, x_crop_start, x_crop_end)
            raise ValueError()
        return cropped_img
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # if self.cross_validate:
        #     scheduler = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)}
        # else:
        #     scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True), "monitor": "val_loss"}
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True), "monitor": "val_loss"}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        LGE_imgs, _, labels, pat_ids = batch
        if self.prediction_task == 'ICD_therapy':
            labels = labels[0]
        elif self.prediction_task == 'ICD_therapy_365days':
            labels = labels[1]
        elif self.prediction_task == 'mortality':
            labels = labels[2]
        elif self.prediction_task == 'mortality_365days':
            labels = labels[3]
            
        output = self.forward(LGE_imgs.float())
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
        LGE_imgs, _, labels, pat_ids = batch
        if self.prediction_task == 'ICD_therapy':
            labels = labels[0]
        elif self.prediction_task == 'ICD_therapy_365days':
            labels = labels[1]
        elif self.prediction_task == 'mortality':
            labels = labels[2]
        elif self.prediction_task == 'mortality_365days':
            labels = labels[3]
            
        output = self.forward(LGE_imgs.float())
        # print('devices', output.get_device(), labels.get_device())
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
        LGE_imgs, _, labels, pat_ids = batch
        if self.prediction_task == 'ICD_therapy':
            labels = labels[0]
        elif self.prediction_task == 'ICD_therapy_365days':
            labels = labels[1]
        elif self.prediction_task == 'mortality':
            labels = labels[2]
        elif self.prediction_task == 'mortality_365days':
            labels = labels[3]
            
        output = self.forward(LGE_imgs.float())
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
        return loss

class GenerateCallback(pl.Callback):
    def __init__(self, version_nr):
        super().__init__()
        self.version_nr = version_nr
        print('version_nr', self.version_nr)
        self.prev_train_loss = None
        self.prev_test_loss = None
        self.patience = 0
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
    pl.seed_everything(args.seed)  # To be reproducible 
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
    if args.continue_from_path == 'None':
        if args.prediction_task == 'ICD_therapy':
            train_loss_weights = train_loss_weights[0]
            val_loss_weights = val_loss_weights[0]
        elif args.prediction_task == 'ICD_therapy_365days':
            train_loss_weights = train_loss_weights[1]
            val_loss_weights = val_loss_weights[1]
        elif args.prediction_task == 'mortality':
            train_loss_weights = train_loss_weights[2]
            val_loss_weights = val_loss_weights[2]
        elif args.prediction_task == 'mortality_365days':
            train_loss_weights = train_loss_weights[3]
            val_loss_weights = val_loss_weights[3]
        model = ClassificationDenseNetModel(args.model, 
                                            dropout=args.dropout,
                                            myo_checkpoint = args.myo_checkpoint, 
                                            loss_function_string = args.loss_function, 
                                            lr = args.lr, 
                                            prediction_task = args.prediction_task, 
                                            train_loss_weights=train_loss_weights, 
                                            val_loss_weights=val_loss_weights,
                                            use_val_weights=args.use_val_weights,
                                            cross_validate=args.cross_validate)
        trainer.fit(model, train_loader, val_loader)
    else:
        print(f'Continued training from checkpoint {args.continue_from_path}')
        model = ClassificationDenseNetModel.load_from_checkpoint(args.continue_from_path)
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
    parser.add_argument('--model', default='densenet', type=str,
                        help='What model to use for the classification',
                        choices=['densenet_maxpool', 'densenet_padding'])
    parser.add_argument('--prediction_task', default='None', type=str,
                        help='Task to predict.',
                        choices=['ICD_therapy', 'ICD_therapy_365days', 'mortality', 'mortality_365days'])
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate')
    parser.add_argument('--continue_from_path', default='None', type=str,
                        help='Path to model checkpoint from which we want to continue training.')

    # Optimizer hyperparameters
    parser.add_argument('--loss_function', default='BCEwithlogits', type=str,
                        help='What loss funciton to use for the segmentation',
                        choices=['BCEwithlogits'])
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--dataset', default='AUMC3D', type=str,
                        help='What dataset to use for the segmentation',
                        choices=['AUMC3D', 'AUMC3D_version2', 'AUMC3D_version3'])  
    parser.add_argument('--resize', default='crop', type=str,
                        help='Whether to resize all images to 256x256 or to crop images to the size of the smallest image width and height',
                        choices=['resize', 'crop'])    
    parser.add_argument('--size', default=['132', '132'], nargs='+', type=str,
                        help='Shape to which the images need to be cropped. Elements of lists are Strings which are later converted to ints.')
    parser.add_argument('--normalize', default=['clip', 'scale_before_gamma'], nargs='+', type=str,
                        help='Type of normalization thats performed on the data',
                        choices=['clip', 'scale_before_gamma'])
    parser.add_argument('--myo_checkpoint', default=r'segment_logs\fibrosis\lightning_logs\version_61\checkpoints\epoch=113-step=8663.ckpt', type=str,
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

    #write prints to file
    if args.continue_from_path == 'None':
        version_nr = get_model_version_no_classification(os.path.join(args.log_dir, 'densenet'))
    else:
        folders = args.continue_from_path.split('/')
        for folder in folders:
            if 'version' in folder:
                version_nr = int(folder.split('_')[-1])
    args.version_nr = version_nr
    file_name = f'train_classification_version_{version_nr}.txt'
    first_path = os.path.join(args.log_dir, 'densenet', 'lightning_logs', file_name)
    second_path = os.path.join(args.log_dir, 'densenet', 'lightning_logs', f"version_{version_nr}", file_name)
    print('Classification training has started!')
    if args.continue_from_path == 'None':
        sys.stdout = open(first_path, "w")
    else:
        sys.stdout = open(first_path, 'a')
        print(f'\nResumed training from checkpoint: {args.continue_from_path}')
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print(f"Dataset: {args.dataset} | model: {args.model} | dataset: {args.dataset} | labels: {args.prediction_task} | resize: {args.resize} | normalize: {args.normalize} | augmentation: {args.transformations} | myocard checkpoint: {args.myo_checkpoint} | loss_function: {args.loss_function} | lr: {args.lr} | batch_size: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed} | version_no: {version_nr}")
    final_val_result = train(args)
    sys.stdout.close()
    sys.stdout = open("/dev/stdout", "w")
    os.rename(first_path, second_path)
    print('Classification completed')
