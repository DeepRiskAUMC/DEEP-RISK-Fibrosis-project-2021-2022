import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from train_classification_gnn_pl import ClassificationGNNModel
from data_loading.load_graph_data import load_graph_data
from utils_functions.criterions import evaluate_regression

def create_saving_folder(output_path, model_name, version_nr, test_train_or_inference):
    task_folder = os.path.join(output_path, model_name)
    os.makedirs(task_folder, exist_ok=True)
    version_folder = os.path.join(task_folder, f"version_{version_nr}")
    os.makedirs(version_folder, exist_ok=True)
    saving_folder = os.path.join(version_folder, test_train_or_inference)
    os.makedirs(saving_folder, exist_ok=True)
    return version_folder, saving_folder

def get_model(model_name, model_path, prediction_task, probs_path, distance_measure):
    if model_name == 'GNN':
        try:
            model = ClassificationGNNModel.load_from_checkpoint(model_path, train_loss_weights=1.0, val_loss_weights=1.0)
        except:
            model = ClassificationGNNModel.load_from_checkpoint(model_path, prediction_task=prediction_task, probs_path=probs_path, train_loss_weights=1.0, val_loss_weights=1.0)
        if model.probs_path == '':
            model.probs_path = probs_path
        if model.distance_measure is None:
            model.distance_measure = distance_measure
        print('using distance measure:', model.distance_measure)
    else:
        raise ValueError(f'Model name {model_name} not recognized')
    return model

def test(args, version_nr):
    pl.seed_everything(args.seed)  # To be reproducible

    model = get_model(args.model_name, args.model_path, args.prediction_task, args.probs_path, args.distance_measure)
    model.eval()

    if model.prediction_task == 'LVEF':
        extra_label = 'LVEF'
    else:
        extra_label = None

    if args.test_train_or_inference == 'test':
        version_folder, saving_folder = create_saving_folder(args.output_path, args.model_name, version_nr, 'test')
        _, test_loader = load_graph_data(model.probs_path,
                                            args.myo_feat_path,
                                            args.fib_feat_path,
                                            args.model_name,
                                            args.num_myo, 
                                            args.num_fib,
                                            args.edges_per_node, 
                                            model.node_attributes, 
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers, 
                                            only_test=True,
                                            distance_measure=model.distance_measure,
                                            extra_label=extra_label)
    else:
        version_folder, saving_folder = create_saving_folder(args.output_path, args.model_name, version_nr, 'validation')
        test_loader, _ = load_graph_data(model.probs_path,
                                            args.myo_feat_path,
                                            args.fib_feat_path,
                                            args.model_name,
                                            args.num_myo, 
                                            args.num_fib,
                                            args.edges_per_node, 
                                            model.node_attributes, 
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers, 
                                            only_test=True,
                                            distance_measure=model.distance_measure,
                                            extra_label=extra_label)
    loss_function = torch.nn.MSELoss()
    prediction_task = model.prediction_task
    outputs_list, labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            # prepare input
            graph = batch
            labels = graph.y
            if prediction_task == 'LVEF':
                labels = labels[:,4]

            output = model(graph)
            if labels.shape[0] == 1:
                output = torch.tensor([output])
            output = output.detach()
            outputs_list += output.tolist()
            labels_list += labels.tolist()
            print('list lengths', len(outputs_list), len(labels_list))

            # calculate metrics
    labels_array, outputs_array = np.array(labels_list), np.array(outputs_list)
    print(labels_array, outputs_array)
    print(labels_array.shape, outputs_array.shape)
    metrics = evaluate_regression(labels_array, outputs_array)
    print_metrics = ''
    for metric, value in metrics.items():
        print_metrics += f'{metric}: {value}. '
    print(print_metrics)

    return version_folder

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Task hyperparameters
    parser.add_argument('--test_train_or_inference', default='train', type=str,
                        help='Indicate whether you want to test the model (including metrics), transform the train images (no metrics) or want to use it for inference',
                        choices=['test', 'train', 'validation', 'inference'])

    # Model hyperparameters
    parser.add_argument('--model_path', default='segment_logs\myocard\lightning_logs\\version_1\checkpoints\epoch=399-step=9999.ckpt', type=str,
                        help='Path to trained model')
    parser.add_argument('--prediction_task', default='LVEF', type=str,
                        help='Task to predict.',
                        choices=['LVEF'])    
    parser.add_argument('--num_myo', default=500, type=int,
                        help='Number of voxels samples from the myocardium segmentation model.')
    parser.add_argument('--num_fib', default=500, type=int,
                        help='Number of voxels samples from the fibrosis segmentation model.')
    parser.add_argument('--edges_per_node', default=50, type=int,
                        help='Number of edges per node.')
    parser.add_argument('--distance_measure', default='', type=str,
                        help='Type of distance to use between the different nodes',
                        choices=['none', 'euclidean', 'relative_position', 'displacement'])

    # Other hyperparameters
    parser.add_argument('--probs_path', default='AUMC3D', type=str,
                        help='Location where the file with the segment outputs is stored')
    parser.add_argument('--myo_feat_path', default=r'outputs\segment_output\segmentation_tensor_hdf5\myocardium_version_23\deeprisk_myocardium_features_n=535.hdf5', type=str,
                        help='Location where the file with the myocardium segment outputs is stored')
    parser.add_argument('--fib_feat_path', default=r'outputs\segment_output\segmentation_tensor_hdf5\fibrosis_version_71\deeprisk_fibrosis_features_n=535.hdf5', type=str,
                        help='Location where the file with the fibrosis segment outputs is stored')
    parser.add_argument('--model_name', default='GNN', type=str,
                        help='What type of model is used for the classification',
                        choices=['GNN', 'EGNN'])
    parser.add_argument('--batch_size', default=24, type=int,
                        help='Minibatch size')
    parser.add_argument('--output_path', default='outputs/classification_output', type=str,
                        help='Path to store the segmented images')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    #write prints to file
    if torch.cuda.is_available():
        version_nr = args.model_path.split('version_')[-1].split('/')[0]
    else:
        version_nr = args.model_path.split('version_')[-1].split('\\')[0]
    print('Regression has started!')
    if args.test_train_or_inference in ['test', 'validation']:
        if 'last.ckpt' in args.model_path:
            file_name = f'{args.test_train_or_inference}_last_classification_version_{version_nr}.txt'
            args.save_images = False
        else:
            file_name = f'{args.test_train_or_inference}_classification_version_{version_nr}.txt'
            args.save_images = True
        first_path = os.path.join(args.output_path, file_name)
        # second_path = os.path.join(args.output_path, f"version_{version_nr}", file_name)
        sys.stdout = open(first_path, "w")
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        print(f"testing on split: {args.test_train_or_inference} | batch_size: {args.batch_size} | seed: {args.seed} | version_no: {version_nr} | model_path: {args.model_path}")
        print(f"labels: {args.prediction_task} | num_myo: {args.num_myo} | num_fib: {args.num_fib} | edges_per_node: {args.edges_per_node}")
        version_folder = test(args, version_nr)
        sys.stdout.close()
        os.rename(first_path, os.path.join(version_folder, file_name))
    sys.stdout = open("/dev/stdout", "w")
    print('Regression completed')