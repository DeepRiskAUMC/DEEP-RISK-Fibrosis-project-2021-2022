import torch
import os
import sys
import argparse
from datetime import datetime
import numpy as np
from functools import partial
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import pytorch_lightning as pl
from utils_functions.utils import get_model_version_no_classification
from train_classification_CNN import train as train_cnn
# from train_classification_densenet import train as train_densenet
from train_classification_gnn_pl import train as train_gnn
# from train_classification_multi_input_pl import train as train_multi_input
from train_classification_encoder import train as train_encoder

def hyper_tune_cnn(args, params):
    args.extra_input, args.hidden_layers_conv, args.hidden_layers_lin = params['extra_input'], params['hidden_layers_conv'], params['hidden_layers_lin']
    output = train_cnn(args)[0]
    auc = output['test_AUC']
    bceloss = output['test_loss']
    results = {'loss' : 1.0 - auc,
                'bceloss' : bceloss,
                'status' : STATUS_OK}
    return results

def hyper_tune_gnn(args, params):
    args.num_myo, args.num_fib, args.edges_per_node  = params['num_myo'], params['num_fib'], params['edges_per_node']
    args.hidden_channels_gcn, args.num_gcn_layers = params['hidden_channels_gcn'], params['num_gcn_layers']
    output = train_gnn(args)[0]
    auc = output['test_AUC']
    bceloss = output['test_loss']
    results = {'loss' : 1.0 - auc,
                'bceloss' : bceloss,
                'status' : STATUS_OK}
    return results

def hyper_tune_encoder(args, params):
    args.hidden_layers_lin = params['hidden_layers_lin']
    output = train_encoder(args)[0]
    auc = output['test_AUC']
    bceloss = output['test_loss']
    results = {'loss' : 1.0 - auc,
                'bceloss' : bceloss,
                'status' : STATUS_OK}
    return results

def hyper_tune(function_name, param_dict, args):
    version_nr = get_model_version_no_classification(os.path.join(args.log_dir, args.model))
    file_name = f'hypertuning_{args.model_name}_{version_nr}.txt'
    first_path = os.path.join(args.log_dir, file_name)
    second_path = os.path.join(args.log_dir, args.model, 'lightning_logs', f"version_{version_nr}", file_name)
    print('Parameter tuning has started!')
    sys.stdout = open(first_path, "w")
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print('Hyperparameter dictionary:')
    print(param_dict)
    print('Other params:')
    print(vars(args))

    torch.manual_seed(args.seed)
    rstate = np.random.default_rng(args.seed)
    pl.seed_everything(args.seed)  # To be reproducible

    param_space = {}
    for param_key, param_list in param_dict.items():
        param_space[param_key] = hp.choice(param_key, param_list)
    
    if function_name == 'hyper_tune_cnn':
        function = hyper_tune_cnn
    elif function_name == 'hyper_tune_gnn':
        function = hyper_tune_gnn
    elif function_name == 'hyper_tune_encoder':
        function = hyper_tune_encoder

    optimization_function = partial(
                                function,
                                args
                            )
    algo = partial(tpe.suggest, n_startup_jobs=10)
    trials = Trials()
    hopt = fmin(fn=optimization_function,
                space=param_space,
                algo=algo,
                max_evals=20,
                rstate=rstate,
                trials=trials,
                verbose=False)

    for i, x in enumerate(trials.trials):
        print('trial', i)
        values = x['misc']['vals']
        print_string = ''
        for param_key, param_list in param_dict.items():
            print_string += f'{param_key}: {param_list[values[param_key][0]]}. '
        print(print_string)
        print(f"AUC: {1.0 - x['result']['loss']}. {x['result']}")
    sys.stdout.close()
    sys.stdout = open("/dev/stdout", "w")
    os.rename(first_path, second_path)

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model_name', default='CNN', type=str,
                        help='What model to use for the hyperparameter tuning',
                        choices=['CNN', 'CNN_clinical', 'encoder', 'encoder_clinical', 'encoder_flatten', 'encoder_flatten_clinical', 'GNN_myo_none', 
                                'GNN_myo_euc', 'GNN_myo_rel', 'GNN_fib_none', 'GNN_fib_euc', 'GNN_fib_rel', 'GNN_myofib_none', 'GNN_myofib_euc', 
                                'GNN_myofib_rel', 'GNN_grey_none', 'GNN_grey_euc', 'GNN_grey_rel'])
    parser.add_argument('--prediction_task', default='None', type=str,
                        help='Task to predict.',
                        choices=['ICD_therapy', 'ICD_therapy_365days', 'mortality', 'mortality_365days', 'gender', 'LVEF'])

    args = parser.parse_args()

    #general arguments
    args.n_classes = 1
    args.continue_from_path = 'None'
    args.lr = 1e-3
    args.dataset = 'AUMC3D_version3'
    args.resize = 'crop'
    args.size = ['132', '132']
    args.normalize = ['clip', 'scale_before_gamma']
    args.transformations = ['hflip', 'vflip', 'rotate']
    args.fib_checkpoint = r'outputs/segment_logs/fibrosis/lightning_logs/version_71/checkpoints/epoch=57-step=4408.ckpt'
    args.epochs = 100
    args.seed = 42
    args.num_workers = 4
    args.log_dir = os.path.join('hyperparameter_search', args.model_name)

    if args.model_name in ['CNN', 'CNN_clinical']:
        param_dict = {'extra_input' : [[], ['LGE_image'], ['myocard'], ['LGE_image', 'myocard']],
                        'hidden_layers_conv' : [[8, 16], [16, 16], [16, 32], [32, 32]],
                        'hidden_layers_lin' : [[64], [64, 128], [32, 64, 128], [64, 128, 128]]}
        if args.model_name == 'CNN':
            args.use_MRI_features = False
        else:
            args.use_MRI_features = True
        args.model = 'CNN'
        args.feature_multiplication = 4
        args.loss_function = 'BCEwithlogits'
        args.use_val_weights = False
        args.batch_size = 6

        hyper_tune(function_name='hyper_tune_cnn', param_dict=param_dict, args=args)
    elif args.model_name in ['encoder', 'encoder_clinical']:
        param_dict = {'hidden_layers_lin' : [[128, 64, 32], [32, 64, 128], [64, 128, 128]]}
        # param_dict = {'hidden_layers_lin' : [[32, 64, 128]]}
        if args.model_name == 'encoder':
            args.use_MRI_features = False
        else:
            args.use_MRI_features = True
        args.model = 'MLP'
        args.flatten_or_maxpool = 'maxpool'
        args.extra_input = []
        args.loss_function = 'BCEwithlogits'
        args.use_val_weights = False
        args.batch_size = 16

        hyper_tune(function_name='hyper_tune_encoder', param_dict=param_dict, args=args)
    elif args.model_name in ['encoder_flatten', 'encoder_flatten_clinical']:
        param_dict = {'hidden_layers_lin' : [[128, 64, 32], [32, 64, 128], [64, 128, 128]]}
        # param_dict = {'hidden_layers_lin' : [[32, 64, 128]]}
        if args.model_name == 'encoder_flatten':
            args.use_MRI_features = False
        else:
            args.use_MRI_features = True
        args.model = 'MLP'
        args.flatten_or_maxpool = 'flatten'
        args.extra_input = []
        args.loss_function = 'BCEwithlogits'
        args.use_val_weights = False
        args.batch_size = 16
        hyper_tune(function_name='hyper_tune_encoder', param_dict=param_dict, args=args)
    elif 'GNN' in args.model_name:
        param_dict = {'num_myo' : [100, 250, 500],
                        'num_fib' : [100, 250, 500],
                        'edges_per_node' : [25, 50, 100],
                        'hidden_channels_gcn' : [8, 16, 32, 64],
                        'num_gcn_layers' : [3, 4, 5],
                        }
        if 'myofib' in args.model_name:
            args.node_attributes = 'features_myo_fib'
        elif 'myo' in args.model_name:
            args.node_attributes = 'features_myo'
        elif 'fib' in args.model_name:
            args.node_attributes = 'features_fib'
        elif 'grey' in args.model_name:
            args.node_attributes = 'grey_values'
        else:
            raise ValueError(f'features not recognized in model name: {args.model_name}')
        if 'none' in args.model_name:
            args.distance_measure = 'none'
            args.dist_info = False
        elif 'euc' in args.model_name:
            args.distance_measure = 'euclidean'
            args.dist_info = True
        elif 'rel' in args.model_name:
            args.distance_measure = 'relative_position'
            args.dist_info = True
        else:
            raise ValueError(f'distance measure not recognized in model name: {args.model_name}')

        args.model = 'GNN'
        args.update_pos = False
        args.dropout = 0.1
        args.batch_norm = False
        args.instance_norm = False
        args.hidden_units_fc = None
        args.kernel_size = None
        if args.prediction_task == 'LVEF':
            args.loss_function = 'MSE'
        else:
            args.loss_function = 'BCEwithlogits'
        args.use_val_weights = False
        args.batch_size = 8
        args.probs_path = 'outputs/segment_output/segmentation_probs/version_71_AUMC3D_version3/deeprisk_myocard_fibrosis_probabilities_n=535.hdf5'
        args.myo_feat_path = 'outputs/segment_output/segmentation_tensors_hdf5/myocardium_version_23_AUMC3D_version3/deeprisk_myocardium_features_n=535.hdf5'
        args.fib_feat_path = 'outputs/segment_output/segmentation_tensors_hdf5/fibrosis_version_71_AUMC3D_version3/deeprisk_fibrosis_features_n=535.hdf5'

        hyper_tune(function_name='hyper_tune_gnn', param_dict=param_dict, args=args)
    
    print('Hyperparameter tuning completed')
