# import os
# import sys
# from datetime import datetime
# import argparse
# import torch
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from train_fibrosis_segmentation_with_myo_pl import SegmentationModel
# from train_classification_densenet import ClassificationDenseNetModel
# from models.segmentation_models import Floor_2d_UnetAllOutputs
# from models.classification_models import MultiInputDenseNetClassificationModel
# from data_loading.load_classification_data import load_classification_data_clinical
# from utils_functions.utils import load_pretrained_layers, get_model_version_no_classification

# class MultiInputClassificationModel(pl.LightningModule):
#     def __init__(self, loss_function_string, prediction_task, fib_checkpoint, densenet_checkpoint, train_loss_weights, val_loss_weights) -> None:
#         super().__init__()
#         self.prediction_task = prediction_task

#         full_segmentation_model = SegmentationModel.load_from_checkpoint(fib_checkpoint)
#         self.myo_model = full_segmentation_model.myo_model
#         fib_model = full_segmentation_model.fib_model
#         self.segmentation_model = Floor_2d_UnetAllOutputs(1, 1, bilinear=False, sigmoid_finish=fib_model.sigmoid_finish, feature_multiplication=fib_model.feature_multiplication)
#         self.segmentation_model = load_pretrained_layers(self.segmentation_model, fib_model)
#         self.segment_model_output_size = 32

#         self.densenetmodel = ClassificationDenseNetModel.load_from_checkpoint(densenet_checkpoint)

#         if loss_function_string == 'BCEwithlogits':
#             self.train_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(train_loss_weights))
#             self.val_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(val_loss_weights))
#             self.sigmoid_finish = False
#             self.sigmoid = torch.nn.Sigmoid()
#         else:
#             raise ValueError(f"Loss function {loss_function_string} not known")
        
#         self.final_classifier = MultiInputDenseNetClassificationModel(self.myo_model, self.segmentation_model, self.densenetmodel, sigmoid_finish=self.sigmoid_finish)

#     def forward(self, imgs, MRI_features):
#         device = imgs.device
#         if imgs.dim() == 4:
#             imgs = imgs.unsqueeze(dim=0)
#         output = self.final_classifier(imgs, MRI_features)
#         return output.squeeze()

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#         scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True), "monitor": "val_loss"}
#         return [optimizer], [scheduler]

#     def training_step(self, batch, batch_idx):
#         LGE_imgs, MRI_features, labels, pat_ids = batch
#         if self.prediction_task == 'ICD_therapy':
#             labels = labels[0]
#         elif self.prediction_task == 'ICD_therapy_365days':
#             labels = labels[1]
#         elif self.prediction_task == 'mortality':
#             labels = labels[2]
#         elif self.prediction_task == 'mortality_365days':
#             labels = labels[3]
            
#         output = self.forward(LGE_imgs.float(), MRI_features.float())
#         loss = self.train_loss_function(output, labels)
#         self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=LGE_imgs.shape[0])
#         if self.sigmoid_finish:
#             prediction = torch.round(output)
#         else:
#             prediction = torch.round(self.sigmoid(output))
#         accuracy = ((prediction == labels).float().sum()/len(labels))
#         self.log("train_acc", accuracy, on_step=False, on_epoch=True, batch_size=LGE_imgs.shape[0])
#         return loss

#     def validation_step(self, batch, batch_idx):
#         LGE_imgs, MRI_features, labels, pat_ids = batch
#         if self.prediction_task == 'ICD_therapy':
#             labels = labels[0]
#         elif self.prediction_task == 'ICD_therapy_365days':
#             labels = labels[1]
#         elif self.prediction_task == 'mortality':
#             labels = labels[2]
#         elif self.prediction_task == 'mortality_365days':
#             labels = labels[3]
            
#         output = self.forward(LGE_imgs.float(), MRI_features.float())
#         loss = self.val_loss_function(output, labels)
#         self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=LGE_imgs.shape[0])
#         if self.sigmoid_finish:
#             prediction = torch.round(output)
#         else:
#             prediction = torch.round(self.sigmoid(output))
#         accuracy = ((prediction == labels).float().sum()/len(labels))
#         self.log("val_acc", accuracy, on_step=False, on_epoch=True, batch_size=LGE_imgs.shape[0])
#         return loss

#     def test_step(self, batch, batch_idx):
#         LGE_imgs, MRI_features, labels, pat_ids = batch
#         if self.prediction_task == 'ICD_therapy':
#             labels = labels[0]
#         elif self.prediction_task == 'ICD_therapy_365days':
#             labels = labels[1]
#         elif self.prediction_task == 'mortality':
#             labels = labels[2]
#         elif self.prediction_task == 'mortality_365days':
#             labels = labels[3]
            
#         output = self.forward(LGE_imgs.float(), MRI_features.float())
#         loss = self.val_loss_function(output, labels)
#         self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=LGE_imgs.shape[0])
#         if self.sigmoid_finish:
#             prediction = torch.round(output)
#         else:
#             prediction = torch.round(self.sigmoid(output))
#         accuracy = ((prediction == labels).float().sum()/len(labels))
#         self.log("test_acc", accuracy, on_step=False, on_epoch=True, batch_size=LGE_imgs.shape[0])
#         return loss
        
# class GenerateCallback(pl.Callback):
#     def __init__(self, batch_size=64, every_n_epochs=5, save_to_disk=False):
#         """
#         Inputs:
#             batch_size - Number of images to generate
#             every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
#             save_to_disk - If True, the samples and image means should be saved to disk as well.
#         """
#         super().__init__()
#         self.prev_train_loss = None
    
#     def on_epoch_end(self, trainer, pl_module):
#         """
#         This function is called after every epoch.
#         Call the save_and_sample function every N epochs.
#         """
#         elogs = trainer.logged_metrics
#         train_loss, val_loss = None, None
#         for log_value in elogs:
#             if 'train' in log_value and 'loss' in log_value:
#                 train_loss = elogs[log_value]
#             elif 'val' in log_value and 'loss' in log_value:
#                 val_loss = elogs[log_value]
#         if train_loss != self.prev_train_loss:
#             print(f"Epoch ({trainer.current_epoch+1}/{trainer.max_epochs}): train loss = {train_loss} | val loss = {val_loss})")
#         self.prev_train_loss = train_loss

# def train(args):
#     os.makedirs(args.log_dir, exist_ok=True)
#     train_loader, val_loader, _, (train_loss_weights, val_loss_weights) = load_classification_data_clinical(args.dataset,
#                                                             batch_size=args.batch_size,
#                                                             val_batch_size='same',
#                                                             num_workers=args.num_workers,
#                                                             only_test=False,
#                                                             transformations=args.transformations,
#                                                             resize=args.resize,
#                                                             size = args.size,
#                                                             normalize=args.normalize)
#     gen_callback = GenerateCallback()
#     lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
#     checkpoint_callback = ModelCheckpoint(mode="min", monitor="val_loss", save_last=True)
#     logging_dir = os.path.join(args.log_dir, args.model)
#     trainer = pl.Trainer(default_root_dir=logging_dir,
#                          log_every_n_steps=5,
#                          gpus=1 if torch.cuda.is_available() else 0,
#                          max_epochs=args.epochs,
#                          callbacks=[checkpoint_callback, lr_monitor, gen_callback],
#                          enable_progress_bar = False)
#     trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

#     # Create model
#     pl.seed_everything(args.seed)  # To be reproducible
#     if args.continue_from_path == 'None':
#         if args.prediction_task == 'ICD_therapy':
#             train_loss_weights = train_loss_weights[0]
#             val_loss_weights = val_loss_weights[0]
#         elif args.prediction_task == 'ICD_therapy_365days':
#             train_loss_weights = train_loss_weights[1]
#             val_loss_weights = val_loss_weights[1]
#         elif args.prediction_task == 'mortality':
#             train_loss_weights = train_loss_weights[2]
#             val_loss_weights = val_loss_weights[2]
#         elif args.prediction_task == 'mortality_365days':
#             train_loss_weights = train_loss_weights[3]
#             val_loss_weights = val_loss_weights[3]
#         model = MultiInputClassificationModel(args.loss_function, args.prediction_task, args.fib_checkpoint, args.densenet_checkpoint, train_loss_weights, val_loss_weights)
#         trainer.fit(model, train_loader, val_loader)
#     else:
#         print(f'Continued training from checkpoint {args.continue_from_path}')
#         model = MultiInputClassificationModel.load_from_checkpoint(args.continue_from_path)
#         trainer.fit(model, train_loader, val_loader, ckpt_path=args.continue_from_path)

#     #Testing
#     model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
#     final_val_result = trainer.test(model, val_loader, verbose=True)
#     return final_val_result

# if __name__ == '__main__':
#     # Feel free to add more argument parameters
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     # Model hyperparameters
#     parser.add_argument('--model', default='multi_input', type=str,
#                         help='What model to use for the segmentation',
#                         choices=['multi_input'])
#     parser.add_argument('--prediction_task', default='None', type=str,
#                         help='Task to predict.',
#                         choices=['ICD_therapy', 'ICD_therapy_365days', 'mortality', 'mortality_365days'])
#     parser.add_argument('--continue_from_path', default='None', type=str,
#                         help='Path to model checkpoint from which we want to continue training.')

#     # Optimizer hyperparameters
#     parser.add_argument('--loss_function', default='BCEwithlogits', type=str,
#                         help='What loss funciton to use for the segmentation',
#                         choices=['BCEwithlogits'])
#     parser.add_argument('--lr', default=1e-3, type=float,
#                         help='Learning rate to use')
#     parser.add_argument('--batch_size', default=16, type=int,
#                         help='Minibatch size')

#     # Other hyperparameters
#     parser.add_argument('--dataset', default='AUMC3D', type=str,
#                         help='What dataset to use for the segmentation',
#                         choices=['AUMC3D'])  
#     parser.add_argument('--resize', default='crop', type=str,
#                         help='Whether to resize all images to 256x256 or to crop images to the size of the smallest image width and height',
#                         choices=['resize', 'crop'])    
#     parser.add_argument('--size', default=['132', '132'], nargs='+', type=str,
#                         help='Shape to which the images need to be cropped. Elements of lists are Strings which are later converted to ints.')
#     parser.add_argument('--normalize', default=['clip', 'scale_before_gamma'], nargs='+', type=str,
#                         help='Type of normalization thats performed on the data',
#                         choices=['clip', 'scale_before_gamma'])
#     parser.add_argument('--fib_checkpoint', default=r'segment_logs\fibrosis\lightning_logs\version_61\checkpoints\epoch=113-step=8663.ckpt', type=str,
#                         help='Path to model checkpoint for the fibrosis segmentations')  
#     parser.add_argument('--densenet_checkpoint', default='', type=str,
#                         help='Path to model checkpoint for the densenet classifications')  
#     parser.add_argument('--transformations', nargs='*', default=['hflip', 'vflip', 'rotate'],
#                         choices=['hflip', 'vflip', 'rotate', 'gamma_correction', 'scale_after_gamma'])
#     parser.add_argument('--epochs', default=200, type=int,
#                         help='Max number of epochs')
#     parser.add_argument('--seed', default=42, type=int,
#                         help='Seed to use for reproducing results')
#     parser.add_argument('--num_workers', default=4, type=int,
#                         help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')
#     parser.add_argument('--log_dir', default='classification_clinical_logs', type=str,
#                         help='Directory where the PyTorch Lightning logs should be created.')

#     args = parser.parse_args()

#     #write prints to file
#     if args.continue_from_path == 'None':
#         version_nr = get_model_version_no_classification(args.log_dir, args.model)
#     else:
#         folders = args.continue_from_path.split('/')
#         for folder in folders:
#             if 'version' in folder:
#                 version_nr = int(folder.split('_')[-1])
#     file_name = f'train_classification_version_{version_nr}.txt'
#     first_path = os.path.join(args.log_dir, args.model, 'lightning_logs', file_name)
#     second_path = os.path.join(args.log_dir, args.model, 'lightning_logs', f"version_{version_nr}", file_name)
#     print('Classification training has started!')
#     if args.continue_from_path == 'None':
#         sys.stdout = open(first_path, "w")
#     else:
#         sys.stdout = open(first_path, 'a')
#         print(f'\nResumed training from checkpoint: {args.continue_from_path}')
#     print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
#     print(f"Dataset: {args.dataset} | model: {args.model} | resize: {args.resize} | normalize: {args.normalize} | augmentation: {args.transformations} | segmentation checkpoint: {args.fib_checkpoint} | densenet checkpoint: {args.densenet_checkpoint} | loss_function: {args.loss_function} | lr: {args.lr} | batch_size: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed} | version_no: {version_nr}")
#     final_val_result = train(args)
#     sys.stdout.close()
#     sys.stdout = open("/dev/stdout", "w")
#     os.rename(first_path, second_path)
#     print('Classification completed')