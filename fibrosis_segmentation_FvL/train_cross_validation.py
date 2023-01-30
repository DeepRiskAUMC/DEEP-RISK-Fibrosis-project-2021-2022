import argparse
import sys
import os
import torch
from datetime import datetime
import pytorch_lightning as pl
from utils_functions.utils import get_model_version_no_classification
from utils_functions.utils import get_print_statement
from train_classification_CNN import train as train_CNN
from train_classification_encoder import train as train_encoder
from train_classification_gnn_pl import train as train_GNN
from train_classification_densenet import train as train_densenet
from train_classification_multi_input_pl import train as train_multimodel


def cross_validate(args):
    args.cross_validate = True
    args.continue_from_path = 'None'
    if 'CNN' in args.model:
        train_function = train_CNN
    elif 'encoder' in args.model:
        train_function = train_encoder
    elif 'GNN' in args.model:
        train_function = train_GNN
    elif 'densenet_padding' in args.model:
        train_function = train_densenet
    elif 'multi_input' in args.model:
        train_function = train_multimodel
    
    split_nr = args.fold
    args.split_name = f'fold{split_nr}'
    logging_dir = os.path.join(args.log_dir, args.model, args.split_name)
    os.makedirs(logging_dir, exist_ok=True)
    version_nr = get_model_version_no_classification(logging_dir)
    output_filename = f'train_cross_validation_{args.split_name}_version{version_nr}'
    # if args.new_dataset:
    #     output_filename = output_filename + '_newdataset'
    # if args.model == 'CNN_clinical' and args.hidden_layers_lin == [64]:
    #     output_filename = output_filename + '_otherhparams'
    output_filename = output_filename + '.txt'
    first_path = os.path.join(logging_dir, output_filename)
    print(f'Classification fold {split_nr} training has started!')
    sys.stdout = open(first_path, "w")
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print(get_print_statement(args))
    train_function(args)
    sys.stdout.close()
    sys.stdout = open("/dev/stdout", "w")
    print(f'Fold {split_nr} completed')

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='GNN', type=str,
                        help='What model to use for the classification',
                        choices=['CNN', 'encoder', 'GNN', 'densenet_padding', 'multi_input'])
    parser.add_argument('--prediction_task', default='LVEF_category', type=str,
                        help='Task to predict.',
                        choices=['ICD_therapy', 'ICD_therapy_365days', 'LVEF_category', 'gender'])
    parser.add_argument('--extra_input', default=[], nargs='*', type=str,
                        help='What extra input to use for the CNN classification model',
                        choices=['LGE_image', 'myocard'])
    parser.add_argument('--use_MRI_features', default='None', type=str,
                        help='Whether to use MRI features for the prediction or not',
                        choices=['None', 'True', 'False'])
    parser.add_argument('--hidden_layers_conv', default=[8,16], nargs='+', type=int,
                        help='Number of channels for the hidden convolutional layers of the classifier network.')
    parser.add_argument('--hidden_layers_lin', default=[], nargs='+', type=int,
                        help='Number of channels for the hidden linear layers of the classifier network.')
    parser.add_argument('--flatten_or_maxpool', default='flatten', type=str,
                        help='Whether to maxpool over the spacial dimensions of the output or flatten the array',
                        choices=['flatten', 'maxpool'])
    parser.add_argument('--node_attributes', default='grey_values', type=str,
                        help='What values to use for the node features.',
                        choices=['grey_values',  'features_myo', 'features_fib', 'features_myo_fib'])
    parser.add_argument('--num_myo', default=500, type=int,
                        help='Number of voxels samples from the myocardium segmentation model.')
    parser.add_argument('--num_fib', default=100, type=int,
                        help='Number of voxels samples from the fibrosis segmentation model.')
    parser.add_argument('--edges_per_node', default=25, type=int,
                        help='Number of edges per node.')
    parser.add_argument('--hidden_channels_gcn', default=64, type=int,
                        help='Number of channels/filters for the hidden layers')
    parser.add_argument('--num_gcn_layers', default=5, type=int,
                        help='Number of graph convolution layers')
    parser.add_argument('--update_pos', default='False', type=str,
                        help='If True, update node positions.',
                        choices=['False'])
    parser.add_argument('--dropout', default=0.1, type=float, #used for GNN and Densenet models
                        help='Dropout rate')
    parser.add_argument('--batch_norm', default='False', type=str,
                        help='If True, use batch normalization.',
                        choices=['True',  'False'])
    parser.add_argument('--instance_norm', default='False', type=str,
                        help='If True, use instance normalization.',
                        choices=['True',  'False'])
    parser.add_argument('--dist_info', default='False', type=str,
                        help='If True, uses distance information. If False, this is a normal GNN and not EGNN.',
                        choices=['True',  'False'])
    parser.add_argument('--distance_measure', default='none', type=str,
                        help='Type of distance to use between the different nodes',
                        choices=['none', 'euclidean', 'relative_position', 'displacement'])
    parser.add_argument('--hidden_units_fc', default=16, type=int, 
                        help='Number of hidden units in the dense classification hidden layer') #not used by model but needed to provide to run the code
    parser.add_argument('--kernel_size', default=5, type=int, 
                        help='Convolution kernel size') #not used by model but needed to provide to run the code

    # Optimizer hyperparameters
    parser.add_argument('--use_val_weights', default='False', type=str,
                        help='If True, uses seperate loss weights for the validation loss function. If False, uses no weights for the validation loss.',
                        choices=['True',  'False'])
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--fold', default=0, type=int,
                        help='What fold to use for the classification',
                        choices=[0,1,2,3,4]) 
    parser.add_argument('--new_dataset', default='True', type=str,
                        help='Whether to use the new split',
                        choices=['True',  'False'])
    parser.add_argument('--resize', default='crop', type=str,
                        help='Whether to resize all images to 256x256 or to crop images to the size of the smallest image width and height',
                        choices=['crop'])    
    parser.add_argument('--size', default=['132', '132'], nargs='+', type=str,
                        help='Shape to which the images need to be cropped. Elements of lists are Strings which are later converted to ints.')
    parser.add_argument('--normalize', default=['clip', 'scale_before_gamma'], nargs='+', type=str,
                        help='Type of normalization thats performed on the data',
                        choices=['clip', 'scale_before_gamma'])
    parser.add_argument('--fib_checkpoint', default=r'none', type=str,
                        help='Path to model checkpoint for the fibrosis segmentations') #needed for the CNN, encoder and multimodel models
    parser.add_argument('--probs_path', default=r'outputs\segment_output\segmentation_probs\version_77_AUMC3D_fold0\deeprisk_myocard_fibrosis_probabilities_AUMC3D_fold0_n=419.hdf5', type=str,
                        help='Location where the file with the segment probabilities is stored')
    parser.add_argument('--myo_feat_path', default=r'outputs\segment_output\segmentation_tensors_hdf5\myocardium_version_29_AUMC3D_fold0\deeprisk_myocardium_features_AUMC3D_fold0_n=419.hdf5', type=str,
                        help='Location where the file with the myocardium segment outputs is stored')
    parser.add_argument('--fib_feat_path', default=r'outputs\segment_output\segmentation_tensors_hdf5\fibrosis_version_77_AUMC3D_fold0\deeprisk_fibrosis_features_AUMC3D_fold0_n=419.hdf5', type=str,
                        help='Location where the file with the fibrosis segment outputs is stored') 
    parser.add_argument('--myo_checkpoint', default='', type=str,
                        help='Path to myo model checkpoint for cropping of the images in the DenseNet model')  
    parser.add_argument('--densenet_checkpoint', default='', type=str,
                        help='Path to model checkpoint for the densenet classifications') #needed for multimodel model
    parser.add_argument('--transformations', nargs='*', default=['hflip', 'vflip', 'rotate'],
                        choices=['hflip', 'vflip', 'rotate'])
    parser.add_argument('--epochs', default=100, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--log_dir', default='outputs/classification_logs/cross_val', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')

    args = parser.parse_args()

    if (args.model == 'CNN' or args.model == 'MLP') and args.use_MRI_features == 'None':
        raise ValueError('use_MRI_features cannot be None')
    args.update_pos = True if args.update_pos == 'True' else False
    args.dist_info = True if args.dist_info == 'True' else False
    args.use_val_weights = True if args.use_val_weights == 'True' else False
    args.batch_norm = True if args.batch_norm == 'True' else False
    args.instance_norm = True if args.instance_norm == 'True' else False
    args.use_MRI_features = True if args.use_MRI_features == 'True' else False
    args.new_dataset = True if args.new_dataset == 'True' else False

    
    if args.model == 'MLP' and len(args.extra_input) > 0:
        raise ValueError('Encoder model should not take extra input')
    # if (args.model == 'MLP' and args.hidden_layers_lin != [128, 64, 32]) or (args.model == 'CNN' and (args.hidden_layers_lin != [64,128,128] and args.hidden_layers_lin != [64])):
    #     raise ValueError('hidden_layers_lin is not correct for this model')
    if (args.dist_info is True and args.distance_measure == 'none') or (args.dist_info is False and args.distance_measure != 'none'):
        raise ValueError('combination of dist_info and distance_measure not valid')
    if args.new_dataset:
        args.dataset = f'AUMC3D_fold{args.fold}_new'
    else:
        args.dataset = f'AUMC3D_fold{args.fold}'

    if args.prediction_task == 'LVEF_category':
        args.loss_function = 'CE'
    else:
        args.loss_function = 'BCEwithlogits'

    if args.prediction_task == 'ICD_therapy':
        args.log_dir = os.path.join(args.log_dir, 'therapy')
    elif args.prediction_task == 'ICD_therapy_365days':
        args.log_dir = os.path.join(args.log_dir, 'therapy_365days')
    elif args.prediction_task == 'LVEF_category':
        args.log_dir = os.path.join(args.log_dir, 'LVEF')
    elif args.prediction_task == 'gender':
        args.log_dir = os.path.join(args.log_dir, 'gender')
    else:
        raise ValueError(f'Prediction task {args.prediction_task} not recognized')

    if args.model == 'CNN' and args.use_MRI_features is True:
        args.model = 'CNN_clinical'
    elif args.model == 'encoder' and args.use_MRI_features is True:
        if args.flatten_or_maxpool == 'maxpool':
            args.model = 'encoder_clinical'
        else:
            args.model = 'encoder_flatten_clinical'
    elif args.model == 'encoder' and args.flatten_or_maxpool == 'flatten':
        args.model = 'encoder_flatten'
    elif args.model == 'GNN':
        if args.node_attributes == 'grey_values':
            attr_abreviation = 'grey'
        elif args.node_attributes == 'features_myo':
            attr_abreviation = 'myo'
        elif args.node_attributes == 'features_fib':
            attr_abreviation = 'fib'
        elif args.node_attributes == 'features_myo_fib':
            attr_abreviation = 'myofib'
        if args.distance_measure == 'none':
            dist_abbrevation = 'none'
        elif args.distance_measure == 'euclidean':
            dist_abbrevation = 'euc'
        elif args.distance_measure == 'relative_position':
            dist_abbrevation = 'rel'
        args.model = f'GNN_{attr_abreviation}_{dist_abbrevation}'

    cross_validate(args)