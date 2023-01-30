# import os 
# import sys
# from datetime import datetime
# import argparse
# import torch
# import pytorch_lightning as pl
# from sklearn.metrics import roc_auc_score
# from pytorch_lightning.callbacks import ModelCheckpoint
# from models.gnn_models import EGNN_Conv_Model, MPNN
# from data_loading.load_graph_data import load_graph_data
# from utils_functions.utils import get_model_version_no_classification
# from utils_functions.criterions import AUC_like_loss, roc_star_loss, epoch_update_gamma

# class ClassificationGNNModel(pl.LightningModule):
#     def __init__(self, 
#                 model_name, 
#                 loss_function_string, 
#                 node_attributes, 
#                 lr, 
#                 prediction_task, 
#                 hidden_channels, 
#                 num_gcn_layers, 
#                 update_pos=False, 
#                 dropout=0.1, 
#                 dist_info=True, 
#                 hidden_units_fc=16, 
#                 kernel_size=5, 
#                 distance_measure='euclidean', 
#                 probs_path='', 
#                 train_loss_weights=None, 
#                 val_loss_weights=None, 
#                 use_val_weights=False,
#                 b_norm = False,
#                 i_norm = False) -> None:
#         super().__init__()
#         self.save_hyperparameters()
#         self.loss_function_string = loss_function_string
#         self.probs_path = probs_path
#         self.sigmoid_finish = True
#         self.use_val_weights = use_val_weights
#         self.node_attributes = node_attributes
#         self.prediction_task = prediction_task
#         self.distance_measure = distance_measure
#         if self.node_attributes in ['grey_values',  'probs_myo', 'probs_fib']:
#             in_channels = 1
#         elif self.node_attributes == 'features_myo':
#             in_channels = 64
#         elif self.node_attributes == 'features_fib':
#             in_channels = 32
#         elif self.node_attributes == 'features_myo_fib':
#             in_channels = 96
#         else:
#             raise ValueError(f"{self.node_attributes} should be in ['grey_values',  'probs_myo', 'probs_fib', 'features_myo', 'features_fib', 'features_myo_fib']")

#         if model_name == 'GNN':
#             if dist_info:
#                 if distance_measure == 'euclidean':
#                     edge_features = 1
#                 elif distance_measure == 'relative_position':
#                     edge_features = 3
#                 else:
#                     raise ValueError(f'distance_measure {distance_measure} not recognized')
#             else:
#                 edge_features = 0
#             self.model = MPNN(node_features=in_channels,
#                                         edge_features=edge_features,
#                                         hidden_features=hidden_channels,
#                                         out_features=1,
#                                         num_layers=num_gcn_layers,
#                                         aggr="mean",
#                                         act=torch.nn.ReLU,
#                                         b_norm=b_norm,
#                                         i_norm=i_norm)
#         elif model_name == 'EGNN':
#             self.model = EGNN_Conv_Model(in_channels=in_channels,
#                                         num_classes=1,
#                                         hidden_channels=hidden_channels,
#                                         N=num_gcn_layers,
#                                         update_pos=update_pos,
#                                         infer_edges=False,
#                                         recurrent=False,
#                                         dropout=dropout,
#                                         dist_info=dist_info,
#                                         hidden_units_fc=hidden_units_fc,
#                                         kernel_size=kernel_size,
#                                         distance_measure=distance_measure)

#         if loss_function_string == 'BCEwithlogits':
#             self.train_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(train_loss_weights))
#             if self.use_val_weights:
#                 self.val_loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(val_loss_weights))
#             else:
#                 self.val_loss_function = torch.nn.BCEWithLogitsLoss()
#             self.sigmoid_finish = False
#             self.sigmoid = torch.nn.Sigmoid()
#         elif loss_function_string == 'MSE':
#             self.train_loss_function = torch.nn.MSELoss()
#             self.val_loss_function = torch.nn.MSELoss()
#             self.sigmoid_finish = False
#         elif loss_function_string == 'rankboost':
#             self.train_loss_function = AUC_like_loss()
#             self.val_loss_function = AUC_like_loss()
#             self.sigmoid_finish = False
#         elif loss_function_string == 'rocstar':
#             self.train_loss_function = roc_star_loss
#             self.val_loss_function = roc_star_loss
#             self.sigmoid_finish = True
#             self.sigmoid = torch.nn.Sigmoid()
#             self.train_y_pred, self.train_y_t = None, None
#             self.train_y_pred, self.train_y_t = None, None
#             self.epoch_gamma = 0.20
#             self.validation_y_pred, self.validation_y_t = None, None
#         else:
#             raise ValueError(f"Loss function {loss_function_string} not known")

#     def forward(self, graph):
#         output = self.model(graph)
#         return output.squeeze()
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#         scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True), "monitor": "val_loss"}
#         return [optimizer], [scheduler]

#     def training_step(self, batch, batch_idx):
#         graph = batch
#         labels = graph.y
#         if self.prediction_task == 'ICD_therapy':
#             labels = labels[:,0]
#         elif self.prediction_task == 'ICD_therapy_365days':
#             labels = labels[:,1]
#         elif self.prediction_task == 'mortality':
#             labels = labels[:,2]
#         elif self.prediction_task == 'mortality_365days':
#             labels = labels[:,3]
#         elif self.prediction_task == 'gender':
#             labels = labels[:,4]
#         elif self.prediction_task == 'LVEF':
#             labels = labels[:,4]
            
#         output = self.forward(graph)
#         if self.loss_function_string == 'rocstar':
#             output = self.sigmoid(output)
#             if self.last_epoch_y_pred is None:
#                 print('*Using Loss BxE for training step')
#                 loss = torch.nn.functional.binary_cross_entropy(output, labels)
#             else:
#                 print('*Using roc_star loss for training step')
#                 loss = roc_star_loss(labels, output, self.epoch_gamma, self.last_epoch_y_t, self.last_epoch_y_pred)
#         else:
#             loss = self.train_loss_function(output, labels)
#         self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
#         if self.prediction_task != 'LVEF':
#             if self.sigmoid_finish:
#                 prediction = torch.round(output)
#             else:
#                 output = self.sigmoid(output)
#                 prediction = torch.round(output)
#             accuracy = ((prediction == labels).float().sum()/len(labels))
#             self.log("train_acc", accuracy, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
#             if self.train_y_pred is None:
#                 self.train_y_pred, self.train_y_t = output.cpu().detach(), labels.cpu().detach()
#             else:
#                 self.train_y_pred = torch.cat((self.train_y_pred, output.cpu().detach()), 0)
#                 self.train_y_t = torch.cat((self.train_y_t, labels.cpu().detach()), 0)
#             # try:
#             #     auc = roc_auc_score(labels.cpu().detach().numpy(), output.cpu().detach().numpy())
#             #     self.log("train_AUC", auc, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
#             # except ValueError:
#             #     print('Value error occured in train step. AUC not logged')
#             # except:
#             #     raise ValueError(f'{sys.exc_info()[0]} occured')
#         return loss

#     def validation_step(self, batch, batch_idx):
#         graph = batch
#         labels = graph.y
#         if self.prediction_task == 'ICD_therapy':
#             labels = labels[:,0]
#         elif self.prediction_task == 'ICD_therapy_365days':
#             labels = labels[:,1]
#         elif self.prediction_task == 'mortality':
#             labels = labels[:,2]
#         elif self.prediction_task == 'mortality_365days':
#             labels = labels[:,3]
#         elif self.prediction_task == 'gender':
#             labels = labels[:,4]
#         elif self.prediction_task == 'LVEF':
#             labels = labels[:,4]
#         else:
#             raise ValueError(f'task {self.prediction_task} not recognized')
            
#         output = self.forward(graph)
#         if output.dim() == 0:
#                 output = output.unsqueeze(0)
#         if self.loss_function_string == 'rocstar':
#             output = self.sigmoid(output)
#             if self.last_epoch_y_pred is None:
#                 print('*Using Loss BxE for validation step')
#                 loss = torch.nn.functional.binary_cross_entropy(output, labels)
#             else:
#                 loss = roc_star_loss(labels,output, self.epoch_gamma, self.last_epoch_y_t, self.last_epoch_y_pred)
#         else:
#             loss = self.val_loss_function(output, labels)
#         self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
#         if self.prediction_task != 'LVEF':
#             if self.sigmoid_finish:
#                 prediction = torch.round(output)
#             else:
#                 output = self.sigmoid(output)
#                 prediction = torch.round(output)
#             accuracy = ((prediction == labels).float().sum()/len(labels))
#             self.log("val_acc", accuracy, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
#             if self.validation_y_pred is None:
#                 self.validation_y_pred, self.validation_y_t = output.cpu().detach(), labels.cpu().detach()
#             else:
#                 self.validation_y_pred = torch.cat((self.validation_y_pred, output.cpu().detach()), 0)
#                 self.validation_y_t = torch.cat((self.validation_y_t, labels.cpu().detach()), 0)
#             # try:
#             #     auc = roc_auc_score(labels.cpu().detach().numpy(), output.cpu().detach().numpy())
#             #     self.log("val_AUC", auc, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
#             # except ValueError:
#             #     print('Value error occured in validation step. AUC not logged')
#             # except:
#             #     raise ValueError(f'{sys.exc_info()[0]} occured')
#         return loss

#     def test_step(self, batch, batch_idx):
#         graph = batch
#         labels = graph.y
#         if self.prediction_task == 'ICD_therapy':
#             labels = labels[:,0]
#         elif self.prediction_task == 'ICD_therapy_365days':
#             labels = labels[:,1]
#         elif self.prediction_task == 'mortality':
#             labels = labels[:,2]
#         elif self.prediction_task == 'mortality_365days':
#             labels = labels[:,3]
#         elif self.prediction_task == 'gender':
#             labels = labels[:,4]
#         elif self.prediction_task == 'LVEF':
#             labels = labels[:,4]
            
#         output = self.forward(graph)
#         if self.loss_function_string == 'rocstar':
#             output = self.sigmoid(output)
#             if self.last_epoch_y_pred is None:
#                 print('*Using Loss BxE for test step')
#                 loss = torch.nn.functional.binary_cross_entropy(output, labels)
#             else:
#                 loss = roc_star_loss(labels,output, self.epoch_gamma, self.last_epoch_y_t, self.last_epoch_y_pred)
#         else:
#             loss = self.val_loss_function(output, labels)
#         self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
#         if self.prediction_task != 'LVEF':
#             if self.sigmoid_finish:
#                 prediction = torch.round(output)
#             else:
#                 prediction = torch.round(self.sigmoid(output))
#             accuracy = ((prediction == labels).float().sum()/len(labels))
#             self.log("test_acc", accuracy, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
#             # try:
#             #     auc = roc_auc_score(labels.cpu().detach().numpy(), output.cpu().detach().numpy())
#             #     self.log("test_AUC", auc, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
#             # except ValueError:
#             #     print('Value error occured in test step. AUC not logged')
#             # except:
#             #     raise ValueError(f'{sys.exc_info()[0]} occured')
#         return loss

# class GenerateCallback(pl.Callback):
#     def __init__(self):
#         super().__init__()
#         self.prev_train_loss = None
    
#     def on_epoch_end(self, trainer, pl_module):
#         """
#         This function is called after every epoch and prints the train and validation loss
#         """
#         elogs = trainer.logged_metrics
#         train_loss, val_loss = None, None
#         for log_value in elogs:
#             if 'train' in log_value and 'loss' in log_value:
#                 train_loss = elogs[log_value]
#             elif 'val' in log_value and 'loss' in log_value:
#                 val_loss = elogs[log_value]
#         if train_loss != self.prev_train_loss:
#             print(f"Epoch ({trainer.current_epoch+1}/{trainer.max_epochs}: train loss = {train_loss} | val loss = {val_loss})")
#             # update gamma
#             device = pl_module.device
#             pl_module.last_epoch_y_pred = pl_module.train_y_pred.to(device)
#             pl_module.last_epoch_y_t = pl_module.train_y_t.to(device)
#             pl_module.epoch_gamma = epoch_update_gamma(pl_module.last_epoch_y_t, pl_module.last_epoch_y_pred, trainer.current_epoch)

#             #save AUC for train and validation
#             train_auc = roc_auc_score(pl_module.train_y_t.numpy(), pl_module.train_y_pred.numpy())
#             pl_module.log("train_AUC", train_auc)
#             val_auc = roc_auc_score(pl_module.validation_y_t.numpy(), pl_module.validation_y_pred.numpy())
#             pl_module.log("val_AUC", val_auc)
#             pl_module.train_y_pred, pl_module.train_y_t, pl_module.validation_y_t, pl_module.validation_y_pred = None, None, None, None
#         self.prev_train_loss = train_loss

# def train(args):
#     # print(torch.cuda.memory_summary())
#     pl.seed_everything(args.seed)  # To be reproducible   
#     os.makedirs(args.log_dir, exist_ok=True)
#     if args.prediction_task in ['LVEF', 'gender']:
#         extra_label = args.prediction_task
#     else:
#         extra_label = None
#     train_loader, val_loader, _, (train_loss_weights, val_loss_weights) = load_graph_data(args.probs_path,
#                                                                             args.myo_feat_path,
#                                                                             args.fib_feat_path,
#                                                                             args.model,
#                                                                             args.num_myo, 
#                                                                             args.num_fib, 
#                                                                             args.edges_per_node, 
#                                                                             args.node_attributes, 
#                                                                             batch_size=args.batch_size, 
#                                                                             val_batch_size='same', 
#                                                                             num_workers=args.num_workers, 
#                                                                             only_test=False,
#                                                                             distance_measure=args.distance_measure,
#                                                                             extra_label=extra_label)
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
#     if args.continue_from_path == 'None':
#         if args.prediction_task == 'ICD_therapy':
#             train_loss_weights = train_loss_weights[:,0]
#             val_loss_weights = val_loss_weights[:,0]
#         elif args.prediction_task == 'ICD_therapy_365days':
#             train_loss_weights = train_loss_weights[:,1]
#             val_loss_weights = val_loss_weights[:,1]
#         elif args.prediction_task == 'mortality':
#             train_loss_weights = train_loss_weights[:,2]
#             val_loss_weights = val_loss_weights[:,2]
#         elif args.prediction_task == 'mortality_365days':
#             train_loss_weights = train_loss_weights[:,3]
#             val_loss_weights = val_loss_weights[:,3]
#         elif args.prediction_task == 'gender':
#             train_loss_weights = train_loss_weights[:,4]
#             val_loss_weights = val_loss_weights[:,4]
#         elif args.prediction_task == 'LVEF':
#             train_loss_weights, val_loss_weights = None, None
#         model = ClassificationGNNModel(args.model,
#                                         args.loss_function, 
#                                         args.node_attributes, 
#                                         args.lr,
#                                         prediction_task=args.prediction_task,
#                                         hidden_channels=args.hidden_channels_gcn,
#                                         num_gcn_layers=args.num_gcn_layers,
#                                         update_pos=args.update_pos,
#                                         dropout=args.dropout,
#                                         dist_info=args.dist_info,
#                                         hidden_units_fc=args.hidden_units_fc,
#                                         kernel_size=args.kernel_size,
#                                         distance_measure=args.distance_measure,
#                                         train_loss_weights=train_loss_weights,
#                                         val_loss_weights=val_loss_weights,
#                                         use_val_weights=args.use_val_weights,
#                                         b_norm = args.batch_norm,
#                                         i_norm = args.instance_norm)
#         trainer.fit(model, train_loader, val_loader)
#     else:
#         print(f'Continued training from checkpoint {args.continue_from_path}')
#         model = ClassificationGNNModel.load_from_checkpoint(args.continue_from_path)
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
#     parser.add_argument('--model', default='GNN', type=str,
#                         help='What model to use for the classification',
#                         choices=['GNN', 'EGNN'])
#     parser.add_argument('--prediction_task', default='ICD_therapy', type=str,
#                         help='Task to predict.',
#                         choices=['ICD_therapy', 'ICD_therapy_365days', 'mortality', 'mortality_365days', 'gender', 'LVEF'])
#     parser.add_argument('--node_attributes', default='features_fib', type=str,
#                         help='What values to use for the node features.',
#                         choices=['grey_values',  'probs_myo', 'probs_fib', 'features_myo', 'features_fib', 'features_myo_fib'])
#     parser.add_argument('--num_myo', default=10, type=int,
#                         help='Number of voxels samples from the myocardium segmentation model.')
#     parser.add_argument('--num_fib', default=10, type=int,
#                         help='Number of voxels samples from the fibrosis segmentation model.')
#     parser.add_argument('--edges_per_node', default=2, type=int,
#                         help='Number of edges per node.')
#     parser.add_argument('--hidden_channels_gcn', default=8, type=int,
#                         help='Number of channels/filters for the hidden layers')
#     parser.add_argument('--num_gcn_layers', default=4, type=int,
#                         help='Number of graph convolution layers')
#     parser.add_argument('--update_pos', default='False', type=str,
#                         help='If True, update node positions.',
#                         choices=['True',  'False'])
#     parser.add_argument('--dropout', default=0.1, type=float,
#                         help='Dropout rate')
#     parser.add_argument('--batch_norm', default='False', type=str,
#                         help='If True, use batch normalization.',
#                         choices=['True',  'False'])
#     parser.add_argument('--instance_norm', default='False', type=str,
#                         help='If True, use instance normalization.',
#                         choices=['True',  'False'])
#     parser.add_argument('--dist_info', default='True', type=str,
#                         help='If True, uses distance information. If False, this is a normal GNN and not EGNN.',
#                         choices=['True',  'False'])
#     parser.add_argument('--hidden_units_fc', default=16, type=int,
#                         help='Number of hidden units in the dense classification hidden layer')
#     parser.add_argument('--kernel_size', default=5, type=int,
#                         help='Convolution kernel size')
#     parser.add_argument('--distance_measure', default='euclidean', type=str,
#                         help='Type of distance to use between the different nodes',
#                         choices=['none', 'euclidean', 'relative_position', 'displacement'])
#     parser.add_argument('--n_classes', default=1, type=int,
#                         help='Number of classes.')
#     parser.add_argument('--continue_from_path', default='None', type=str,
#                         help='Path to model checkpoint from which we want to continue training.')

#     # Optimizer hyperparameters
#     parser.add_argument('--loss_function', default='rocstar', type=str,
#                         help='What loss funciton to use for the segmentation',
#                         choices=['BCEwithlogits', 'MSE', 'rankboost', 'rocstar'])
#     parser.add_argument('--use_val_weights', default='False', type=str,
#                         help='If True, uses seperate loss weights for the validation loss function. If False, uses no weights for the validation loss.',
#                         choices=['True',  'False'])
#     parser.add_argument('--lr', default=1e-3, type=float,
#                         help='Learning rate to use')
#     parser.add_argument('--batch_size', default=2, type=int,
#                         help='Minibatch size')

#     # Other hyperparameters
#     parser.add_argument('--probs_path', default=r'outputs\segment_output\segmentation_probs\version_71\deeprisk_myocard_fibrosis_probabilities_n=535.hdf5', type=str,
#                         help='Location where the file with the segment probabilities is stored')
#     parser.add_argument('--myo_feat_path', default=r'outputs\segment_output\segmentation_tensor_hdf5\myocardium_version_23\deeprisk_myocardium_features_n=535.hdf5', type=str,
#                         help='Location where the file with the myocardium segment outputs is stored')
#     parser.add_argument('--fib_feat_path', default=r'outputs\segment_output\segmentation_tensor_hdf5\fibrosis_version_71\deeprisk_fibrosis_features_n=535.hdf5', type=str,
#                         help='Location where the file with the fibrosis segment outputs is stored')
#     parser.add_argument('--epochs', default=100, type=int,
#                         help='Max number of epochs')
#     parser.add_argument('--seed', default=42, type=int,
#                         help='Seed to use for reproducing results')
#     parser.add_argument('--num_workers', default=4, type=int,
#                         help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')
#     parser.add_argument('--log_dir', default='outputs/classification_logs', type=str,
#                         help='Directory where the PyTorch Lightning logs should be created.')

#     args = parser.parse_args()

#     args.update_pos = True if args.update_pos == 'True' else False
#     args.dist_info = True if args.dist_info == 'True' else False
#     args.use_val_weights = True if args.use_val_weights == 'True' else False
#     args.batch_norm = True if args.batch_norm == 'True' else False
#     args.instance_norm = True if args.instance_norm == 'True' else False

#     if (args.loss_function == 'MSE' and args.prediction_task != 'LVEF') or (args.prediction_task == 'LVEF' and args.loss_function != 'MSE'):
#         raise ValueError(f'Prediction task {args.prediction_task} incompatible with loss function {args.loss_function}')
#     if args.instance_norm and args.batch_norm:
#         raise ValueError(f'Batch_norm and instance_norm cannot both be true')

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
#     print(f"Model: {args.model} | labels: {args.prediction_task} | node_attributes: {args.node_attributes} | num_myo: {args.num_myo} | num_fib: {args.num_fib} | edges_per_node: {args.edges_per_node} | hidden_channels_gcn: {args.hidden_channels_gcn}")
#     print(f"num_gcn_layers: {args.num_gcn_layers} | update_pos: {args.update_pos} | dropout: {args.dropout} | dist_info: {args.dist_info} | hidden_units_fc: {args.hidden_units_fc} | kernel_size: {args.kernel_size} | distance_measure: {args.distance_measure}")
#     print(f"loss_function: {args.loss_function} | use_val_weights: {args.use_val_weights} | lr: {args.lr} | batch_size: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed} | version_no: {version_nr}")
#     print(f"batch_norm: {args.batch_norm} | instance_norm: {args.instance_norm} |probs_path: {args.probs_path} | myo_feat_path: {args.myo_feat_path} | fib_feat_path: {args.fib_feat_path}")
#     final_val_result = train(args)
#     sys.stdout.close()
#     sys.stdout = open("/dev/stdout", "w")
#     os.rename(first_path, second_path)
#     print('Classification completed')
