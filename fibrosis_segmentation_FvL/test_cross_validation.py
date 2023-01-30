import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from test_classification_pl import test
from test_classification_gnn import test as gnn_test

def test_cross_validate(args):
    args.cross_validate = True
    args.prediction_task = args.model_folder.split('cross_val/')[1].split('/')[0]
    args.model_name = args.model_folder.split('/')[-1]
    # args.prediction_task = args.model_folder.split('cross_val\\')[1].split('\\')[0]
    # args.model_name = args.model_folder.split('\\')[-1]
    args.clinical_features = True if ('clinical' in args.model_name or args.model_name == 'multi_input') else False

    file_name = f'{args.test_train_or_inference}_cv_version_{args.model_version}_{args.which_model}.txt'
    saving_folder = os.path.join(args.output_path, args.prediction_task, args.model_name)
    os.makedirs(saving_folder, exist_ok=True)
    first_path = os.path.join(saving_folder, file_name)
    sys.stdout = open(first_path, "w")
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print(f"New dataset: {args.new_dataset} | model: {args.model_name} | Version_nr = {args.model_version} | batch_size: {args.batch_size} | seed: {args.seed} | model_folder: {args.model_folder}")

    outputs_array, labels_array = None, None
    for split_version in args.folds:
        args.split_name = f'fold{split_version}'
        print(f'Testing {args.split_name}')
        args.dataset = f'AUMC3D_{args.split_name}' if args.new_dataset is False else f'AUMC3D_{args.split_name}_new'
        checkpoint_folder = os.path.join(args.model_folder, f'fold{split_version}', 'lightning_logs', f'version_{args.model_version}', 'checkpoints')
        checkpoint_names = os.listdir(checkpoint_folder)
        for fn in checkpoint_names:
            if fn.startswith('epoch=') and args.which_model == 'loss':
                args.model_path = os.path.join(checkpoint_folder, fn)
            elif fn.startswith('last') and args.which_model == 'last':
                args.model_path = os.path.join(checkpoint_folder, fn)
            elif fn.startswith('val_auc_epochs') and args.which_model == 'AUC':
                args.model_path = os.path.join(checkpoint_folder, fn)
        if 'GNN' in args.model_name:
            args.probs_path = os.path.join(f'{args.probs_prefix}_fold{split_version}', f'deeprisk_myocard_fibrosis_probabilities_AUMC3D_fold{split_version}_n=419.hdf5')
            args.myo_feat_path = os.path.join(f'{args.myo_feat_folder_prefix}_fold{split_version}', f'deeprisk_myocardium_features_AUMC3D_fold{split_version}_n=419.hdf5')
            args.fib_feat_path = os.path.join(f'{args.fib_feat_folder_prefix}_fold{split_version}', f'deeprisk_fibrosis_features_AUMC3D_fold{split_version}_n=419.hdf5')
            _, split_outputs_array, split_labels_array = gnn_test(args, args.model_version)
        else:
            _, split_outputs_array, split_labels_array = test(args, args.model_version)
        if outputs_array is None:
            outputs_array, labels_array = split_outputs_array.squeeze(), split_labels_array.squeeze()
        else:
            outputs_array, labels_array = np.concatenate([outputs_array, split_outputs_array.squeeze()]), np.concatenate([labels_array, split_labels_array.squeeze()])
    
    print(labels_array.shape, outputs_array.shape)
    # print(labels_array, outputs_array)
    if args.prediction_task == 'LVEF_category':
        roc_auc = roc_auc_score(labels_array, outputs_array, multi_class='ovo', labels=np.unique(labels_array))
    else:
        roc_auc = roc_auc_score(labels_array, outputs_array)
        fpr, tpr, thresholds = roc_curve(labels_array, outputs_array)
        plt.figure()
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic for {args.model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saving_folder, f'ROC_{args.model_name}_version{args.model_version}_{args.which_model}'))
        plt.show()
    print(f'Final AUC: {roc_auc}')
    sys.stdout.close()
    sys.stdout = open("/dev/stdout", "w")


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Task hyperparameters
    parser.add_argument('--test_train_or_inference', default='test', type=str,
                        help='Indicate whether you want to test the model (including metrics), transform the train images (no metrics) or want to use it for inference',
                        choices=['test', 'validation'])

    # Model hyperparameters
    parser.add_argument('--model_folder', default=r'outputs\classification_logs\cross_val\therapy\CNN', type=str,
                        help='Path to folder that contains the folds directories')
    parser.add_argument('--which_model', default='AUC', type=str,
                        help='Whether to use the new split',
                        choices=['AUC',  'last', 'loss'])
    parser.add_argument('--folds', default=[0, 1], nargs='+', type=int,
                        help='Which folds you want to test')
    parser.add_argument('--model_version', default=0, type=int,
                        help='Versions of the splits you want to test')  
    parser.add_argument('--num_myo', default=500, type=int,
                        help='Number of voxels samples from the myocardium segmentation model.')
    parser.add_argument('--num_fib', default=100, type=int,
                        help='Number of voxels samples from the fibrosis segmentation model.')
    parser.add_argument('--edges_per_node', default=25, type=int,
                        help='Number of edges per node.')       
    # Other hyperparameters
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
    parser.add_argument('--probs_prefix', default=r'', type=str,
                        help='Location where the file with the myocardium segment outputs is stored')
    parser.add_argument('--myo_feat_folder_prefix', default=r'', type=str,
                        help='Location where the file with the myocardium segment outputs is stored')
    parser.add_argument('--fib_feat_folder_prefix', default=r'', type=str,
                        help='Location where the file with the fibrosis segment outputs is stored') 
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Minibatch size')
    parser.add_argument('--output_path', default='outputs/classification_output/cross_val', type=str,
                        help='Path to store the segmented images')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')

    args = parser.parse_args()

    args.new_dataset = True if args.new_dataset == 'True' else False

    test_cross_validate(args)
    print('Segmentation completed')