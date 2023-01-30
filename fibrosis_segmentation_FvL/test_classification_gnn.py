import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.metrics import roc_curve, roc_auc_score
from train_classification_gnn_pl import ClassificationGNNModel
from data_loading.load_graph_data import load_graph_data
from utils_functions.criterions import get_accuracy, get_TPR, get_TNR, get_PPV, get_NPV

def create_saving_folder(output_path, model_name, version_nr, test_train_or_inference, cross_validate=False, prediction_task=None, split_name='foldx'):
    if cross_validate:
        prediction_task = 'therapy' if prediction_task == 'ICD_therapy' else 'therapy_365days'
        output_path = os.path.join(output_path, prediction_task)
        os.makedirs(output_path, exist_ok=True)
    model_folder = os.path.join(output_path, model_name)
    os.makedirs(model_folder, exist_ok=True)
    if cross_validate:
        model_folder = os.path.join(model_folder, split_name)
        os.makedirs(model_folder, exist_ok=True)
    version_folder = os.path.join(model_folder, f"version_{version_nr}")
    os.makedirs(version_folder, exist_ok=True)
    saving_folder = os.path.join(version_folder, test_train_or_inference)
    os.makedirs(saving_folder, exist_ok=True)
    return version_folder, saving_folder

def get_model(model_name, model_path, prediction_task, probs_path, distance_measure):
    if 'GNN' in model_name:
        # try:
        model = ClassificationGNNModel.load_from_checkpoint(model_path, train_loss_weights=1.0, val_loss_weights=1.0)
        # except:
        #     model = ClassificationGNNModel.load_from_checkpoint(model_path, prediction_task=prediction_task, probs_path=probs_path, train_loss_weights=1.0, val_loss_weights=1.0, distance_measure=args.distance_measure)
        if model.probs_path == '':
            model.probs_path = probs_path
        if model.distance_measure is None:
            model.distance_measure = distance_measure
    else:
        raise ValueError(f'Model name {model_name} not recognized')
    return model

def test(args, version_nr):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    pl.seed_everything(args.seed)  # To be reproducible

    model = get_model(args.model_name, args.model_path, args.prediction_task, args.probs_path, None)
    model.to(device)
    model.eval()

    args.node_attributes = model.node_attributes
    args.prediction_task = model.prediction_task
    args.distance_measure = model.distance_measure
    if args.prediction_task in ['LVEF', 'gender', 'LVEF_category']:
        extra_label = args.prediction_task
    else:
        extra_label = None

    if args.test_train_or_inference == 'test':
        version_folder, saving_folder = create_saving_folder(args.output_path, args.model_name, version_nr, 'test', args.cross_validate, args.prediction_task, split_name=args.split_name)
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
        version_folder, saving_folder = create_saving_folder(args.output_path, args.model_name, version_nr, 'validation', args.cross_validate, args.prediction_task, split_name=args.split_name)
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
    prediction_task = model.prediction_task
    print(f'Predictions task: {prediction_task}')
    outputs_list, labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            # prepare input
            graph = batch
            graph = graph.to(device)
            labels = graph.y
            if prediction_task == 'ICD_therapy':
                labels = labels[:,0]
            elif prediction_task == 'ICD_therapy_365days':
                labels = labels[:,1]
            elif prediction_task == 'mortality':
                labels = labels[:,2]
            elif prediction_task == 'mortality_365days':
                labels = labels[:,3]
            elif prediction_task == 'gender':
                labels = labels[:,4]
            elif prediction_task == 'LVEF_category':
                labels = labels[:,4].to(device=labels.device, dtype=torch.int64)

            output = model(graph)
            print('labels:', labels)
            print('output:', output)
            if labels.shape[0] == 1:
                output = torch.tensor([output])
            try:
                sigmoid_finish = model.sigmoid_finish
            except:
                sigmoid_finish = True
            if sigmoid_finish:
                prediction = torch.round(output)
            else:
                if prediction_task == 'LVEF_category':
                    output = torch.nn.Softmax(dim=1)(output)
                    prediction = torch.argmax(output, dim=1)
                    print('prediction:', prediction)
                else:
                    output = torch.nn.Sigmoid()(output)
                    prediction = torch.round(output)
            output = output.detach()
            prediction = prediction.detach()
            outputs_list += output.tolist()
            labels_list += labels.tolist()
            # print('list lengths', len(outputs_list), len(labels_list))

            # calculate metrics
    labels_array, outputs_array = np.array(labels_list), np.array(outputs_list)
    
    if prediction_task == 'LVEF_category':
        print('outputs_array:')
        print(outputs_array)
        # auc = roc_auc_score(labels_array, outputs_array[:,1])
        auc = roc_auc_score(labels_array, outputs_array, multi_class='ovo', labels=np.unique(labels_array))
        predictions = np.argmax(outputs_array, axis=1)
        print(labels_array)
        print(predictions)
        print(f"Accuracy: {get_accuracy(predictions, labels_array)}. AUC score: {auc}.")
    else:
        print(labels_array, outputs_array)
        print(labels_array.shape, outputs_array.shape)
        auc = roc_auc_score(labels_array, outputs_array)
        fpr, tpr, thresholds = roc_curve(labels_array, outputs_array)
        plt.plot(fpr,tpr,label="AUC="+str(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.tight_layout()
        if 'last.ckpt' in args.model_path:
            pdf_filename = 'last_ROC_curve.pdf'
        elif 'val_acc' in args.model_path:
            pdf_filename = 'val_auc_ROC_curve.pdf'
        elif 'val_acc' in args.model_path:
            pdf_filename = 'val_AUC_ROC_curve.pdf'
        else:
            pdf_filename = 'ROC_curve.pdf'
        plt.savefig(os.path.join(saving_folder, pdf_filename))

        predictions = np.round(outputs_array)
        print(f"Ratio of positive predictions: {np.sum(predictions)/len(predictions)}")
        print('fpr/tpr/thresholds:')
        print(fpr)
        print(tpr)
        print(thresholds)
        print(f"Accuracy: {get_accuracy(predictions, labels_array)}. AUC score: {auc}. TPR: {get_TPR(predictions, labels_array)}. TNR: {get_TNR(predictions, labels_array)}. PPV: {get_PPV(predictions, labels_array)}. NPV: {get_NPV(predictions, labels_array)}.")
    # print(f"AUC score: {np.round(auc, 3)}. TPR: {np.round(tpr, 3)}. TNR: {np.round(get_TNR(predictions, labels_array), 3)}. PPV: {np.round(get_PPV(predictions, labels_array), 3)}. NPV: {np.round(get_NPV(predictions, labels_array), 3)}.")

    return version_folder, outputs_array, labels_array

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
    parser.add_argument('--prediction_task', default='', type=str,
                        help='Task to predict.',
                        choices=['ICD_therapy', 'ICD_therapy_365days', 'mortality', 'mortality_365days', 'gender'])    
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
    print('Classification has started!')
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
    print('Classification completed')