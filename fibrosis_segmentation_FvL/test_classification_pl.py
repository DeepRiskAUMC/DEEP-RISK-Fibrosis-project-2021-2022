import os
import argparse
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
import torch
from datetime import datetime
import pytorch_lightning as pl
from sklearn.metrics import roc_curve, roc_auc_score
from train_classification_CNN import ClassificationModel as CNN_classification_model
# from train_classification_clinical_features import ClassificationModel as Clinical_classification_model
from train_classification_encoder import ClassificationSegmentationModel as Encoder_classification_model
from train_classification_densenet import ClassificationDenseNetModel
from train_classification_multi_input_pl import MultiInputClassificationModel
from data_loading.load_classification_data import load_classification_data, load_classification_data_clinical
from utils_functions.criterions import get_accuracy, get_TPR, get_TNR, get_PPV, get_NPV
from utils_functions.utils import rename_pretrained_layers

MEAN_CLINICAL_VALUES_TRAIN_SET = [256.9652767527675, 177.80496296296295, 33.637121771217714]
MEAN_CLINICAL_VALUES_TRAIN_SET_V2 = [254.7481749049429, 175.74805343511449, 33.94661596958174]
MEAN_CLINICAL_VALUES_TRAIN_SET_V3 = [262.6642750929368, 183.12481343283582, 32.971561338289966]
MEAN_CLINICAL_VALUES_TRAIN_SET_ICM = [261.56007142857146, 183.92535714285714, 31.202928571428572]
MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD0 = [255.1534520547945, 174.28365384615384, 34.949285714285715]
MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD1 = [256.6355858310627, 174.72994535519126, 34.89713114754098]
MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD2 = [257.16016260162604, 174.2210081743869, 34.91763586956522]
MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD3 = [254.34923497267758, 172.62909340659343, 34.919534246575346]
MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD4 = [253.93165760869567, 171.92640326975476, 35.119239130434785]
MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD0_new = [253.6706539509537, 172.05731506849315, 34.903333333333336]
MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD1_new = [259.35758807588076, 178.38564032697548, 34.06307065217391]
MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD2_new = [249.3472086720867, 166.81392370572206, 36.01388586956522]
MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD3_new = [258.47815217391303, 176.11184782608697, 34.65423913043478]
MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD4_new = [256.9466301369863, 174.71647382920108, 35.004450549450546]

def create_saving_folder(output_path, model_name, version_nr, test_train_or_inference, clinical_features, cross_validate=False, prediction_task=None, split_name='foldx'):
    if clinical_features and not cross_validate:
        output_path = os.path.join(output_path, 'clinical_features')
        os.makedirs(output_path, exist_ok=True)
    elif cross_validate:
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

def get_model(model_name, model_path, prediction_task, clinical_features):
    if not clinical_features:
        if 'encoder' in model_name:
            try:
                model = Encoder_classification_model.load_from_checkpoint(model_path, loss_weights=1.0)
            except:
                model = Encoder_classification_model.load_from_checkpoint(model_path, prediction_task=prediction_task, train_loss_weights=1.0, val_loss_weights=1.0)
        elif model_name == 'unet_single_multi_input' or model_name == 'CNN':
            try:
                model = CNN_classification_model.load_from_checkpoint(model_path, loss_weights=1.0)
            except:
                model = CNN_classification_model.load_from_checkpoint(model_path, prediction_task=prediction_task, loss_weights=1.0)
        elif model_name in ['densenet_maxpool', 'densenet_padding']:
            try:
                model = ClassificationDenseNetModel.load_from_checkpoint(model_path, train_loss_weights=1.0, val_loss_weights=1.0)
            except:
                try:
                    model = ClassificationDenseNetModel.load_from_checkpoint(model_path, model_name=model_name, prediction_task=prediction_task, train_loss_weights=1.0, val_loss_weights=1.0)
                except:
                    model = ClassificationDenseNetModel(model_name, 'outputs/segment_logs/myocard/lightning_logs/version_23/checkpoints/epoch=198-step=15124.ckpt', 'BCEwithlogits', 0.01, prediction_task, train_loss_weights=1.0, val_loss_weights=1.0)
                    old_checkpoint = torch.load(model_path)['state_dict']
                    new_state_dict = rename_pretrained_layers(old_checkpoint)
                    model.load_state_dict(new_state_dict)
        elif model_name == 'multi_input':
            model = MultiInputClassificationModel.load_from_checkpoint(model_path, train_loss_weights=1.0, val_loss_weights=1.0)
        else:
            raise ValueError(f'Model name {model_name} not recognized')
    else:
        if 'encoder' in model_name:
            try:
                model = Encoder_classification_model.load_from_checkpoint(model_path, train_loss_weights=1.0)
            except:
                model = Encoder_classification_model.load_from_checkpoint(model_path, prediction_task=prediction_task, train_loss_weights=1.0)
        elif 'CNN' in model_name:
            model = CNN_classification_model.load_from_checkpoint(model_path, train_loss_weights=1.0)
        elif model_name == 'multi_model' or model_name == 'multi_input':
            try:
                model = MultiInputClassificationModel.load_from_checkpoint(model_path, train_loss_weights=1.0, val_loss_weights=1.0)
            except:
                model = MultiInputClassificationModel.load_from_checkpoint(model_path, prediction_task=prediction_task, train_loss_weights=1.0, val_loss_weights=1.0)
    return model

def test(args, version_nr):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    pl.seed_everything(args.seed)  # To be reproducible
    if 'cross_val' in args.model_path:
        cross_validation = True
    else:
        cross_validation = False
    model = get_model(args.model_name, args.model_path, args.prediction_task, args.clinical_features)
    model.to(device)
    model.eval()

    # test on either the validation or the test set
    if args.clinical_features:
        print('clinical features')
        load_data_function = load_classification_data_clinical
    else:
        print('no clinical features')
        load_data_function = load_classification_data
        mean_clinical_values = []
    dataset_dict = {'AUMC3D' : MEAN_CLINICAL_VALUES_TRAIN_SET,
                    'AUMC3D_version2' : MEAN_CLINICAL_VALUES_TRAIN_SET_V2,
                    'AUMC3D_version3' : MEAN_CLINICAL_VALUES_TRAIN_SET_V3,
                    'AUMC3D_ICM' : MEAN_CLINICAL_VALUES_TRAIN_SET_ICM,
                    'AUMC3D_fold0' : MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD0,
                    'AUMC3D_fold1' : MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD1,
                    'AUMC3D_fold2' : MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD2,
                    'AUMC3D_fold3' : MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD3,
                    'AUMC3D_fold4' : MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD4,
                    'AUMC3D_fold0_new' : MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD0_new,
                    'AUMC3D_fold1_new' : MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD1_new,
                    'AUMC3D_fold2_new' : MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD2_new,
                    'AUMC3D_fold3_new' : MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD3_new,
                    'AUMC3D_fold4_new' : MEAN_CLINICAL_VALUES_TRAIN_SET_FOLD4_new}
    mean_clinical_values = dataset_dict[args.dataset]
    print(f'Using mean values: {mean_clinical_values}')
    val_loader, test_loader = load_data_function(args.dataset,
                                                batch_size=args.batch_size,
                                                val_batch_size='same',
                                                num_workers=args.num_workers,
                                                only_test=True,
                                                resize=args.resize,
                                                size = args.size,
                                                normalize=args.normalize,
                                                mean_values=mean_clinical_values)
    loss_function = torch.nn.BCEWithLogitsLoss()
    prediction_task = model.prediction_task
    if prediction_task == '':
        raise ValueError(f'Predictions task not stored in model. Provide the prediction task using the argument parser.')
    print('Prediction task:', prediction_task)

    if args.test_train_or_inference == 'test':
        version_folder, saving_folder = create_saving_folder(args.output_path, args.model_name, version_nr, 'test', clinical_features=args.clinical_features, cross_validate=cross_validation, prediction_task=prediction_task, split_name=args.split_name)
        test_loader = test_loader
    else:
        version_folder, saving_folder = create_saving_folder(args.output_path, args.model_name, version_nr, 'validation', clinical_features=args.clinical_features, cross_validate=cross_validation, prediction_task=prediction_task, split_name=args.split_name)
        test_loader = val_loader
    
    outputs_list, labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            # prepare input
            if args.clinical_features:
                LGE_imgs, clinical_features, labels, pat_ids = batch
            else:
                LGE_imgs, labels, pat_ids = batch
            LGE_imgs = LGE_imgs.to(device)
            if prediction_task == 'ICD_therapy':
                labels = labels[0]
            elif prediction_task == 'ICD_therapy_365days':
                labels = labels[1]
            elif prediction_task == 'mortality':
                labels = labels[2]
            elif prediction_task == 'mortality_365days':
                labels = labels[3]
            # print(pat_ids)
            if args.clinical_features:
                clinical_features = clinical_features.to(device)
                output = model(LGE_imgs.float(), clinical_features)
            else:
                if args.model_name in ['encoder_mlp', 'encoder', 'encoder_flatten', 'CNN']:
                    output = model(LGE_imgs.float(), None)
                else:
                    output = model(LGE_imgs.float())
            # print(LGE_imgs.shape, output.shape, labels.shape)
            # print(output, labels) 
            if labels.shape[0] == 1:
                output = torch.tensor([output])
            labels = labels.to(device)
            loss = loss_function(output, labels)
            try:
                sigmoid_finish = model.sigmoid_finish
            except:
                sigmoid_finish = True
            if sigmoid_finish:
                prediction = torch.round(output)
            else:
                output = torch.nn.Sigmoid()(output)
                prediction = torch.round(output)
            output = output.detach()
            prediction = prediction.detach()
            outputs_list += output.tolist()
            labels_list += labels.detach().tolist()
            # print('list lengths', len(outputs_list), len(labels_list))

    # calculate metrics
    labels_array, outputs_array = np.array(labels_list), np.array(outputs_list)
    print(labels_array, outputs_array)
    print(labels_array.shape, outputs_array.shape)
    fpr, tpr, thresholds = roc_curve(labels_array, outputs_array)
    auc = roc_auc_score(labels_array, outputs_array)
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
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
    parser.add_argument('--test_train_or_inference', default='validation', type=str,
                        help='Indicate whether you want to test the model (including metrics), transform the train images (no metrics) or want to use it for inference',
                        choices=['test', 'train', 'validation', 'inference'])

    # Model hyperparameters
    parser.add_argument('--model_path', default=r'outputs\classification_logs\densenet\lightning_logs\version_5\checkpoints\last.ckpt', type=str,
                        help='Path to trained model')
    parser.add_argument('--prediction_task', default='', type=str,
                        help='Task to predict.',
                        choices=['ICD_therapy', 'ICD_therapy_365days', 'mortality', 'mortality_365days'])
    parser.add_argument('--extra_input', default=[], nargs='+', type=str,
                        help='What extra input to use for the classification model',
                        choices=['MRI_features', 'LGE_image', 'myocard'])
    parser.add_argument('--normalize', default=['clip', 'scale_before_gamma'], nargs='+', type=str,
                        help='Type of normalization thats performed on the data',
                        choices=['clip', 'scale_before_gamma', 'scale_after_gamma'])
    parser.add_argument('--resize', default='crop', type=str,
                        help='Whether to resize all images to 256x256 or to crop images',
                        choices=['resize', 'crop', 'none'])   
    parser.add_argument('--size', default=['132', '132'], nargs='+', type=str,
                        help='Shape to which the images need to be cropped. Elements of lists are Strings which are later converted to ints.')    
    parser.add_argument('--metrics', default='none', nargs='+', type=str,
                        help='Metrics (other than Dice score) to evaluate the segmentations on.')          
    

    # Other hyperparameters
    parser.add_argument('--dataset', type=str,
                        help='What dataset to use for the segmentation',
                        choices=['AUMC3D', 'AUMC3D_version2', 'AUMC3D_version3', 'AUMC3D_ICM'])
    parser.add_argument('--model_name', default='densenet_padding', type=str,
                        help='What type of model is used for the classification',
                        choices=['CNN', 'encoder_mlp', 'unet_single_multi_input', 'densenet_maxpool', 'densenet_padding', 'multi_model'])
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

    if 'classification_clinical_logs' in args.model_path or 'MRI_features' in args.extra_input:
        args.clinical_features = True
    else:
        args.clinical_features = False

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
        elif 'val_acc' in args.model_path:
            file_name = f'{args.test_train_or_inference}_val_ckpt_classification_version_{version_nr}.txt'
            args.save_images = False
        else:
            file_name = f'{args.test_train_or_inference}_classification_version_{version_nr}.txt'
            args.save_images = True
        first_path = os.path.join(args.output_path, file_name)
        # second_path = os.path.join(args.output_path, f"version_{version_nr}", file_name)
        sys.stdout = open(first_path, "w")
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        print(f"normalize: {args.normalize} | testing on split: {args.test_train_or_inference} | batch_size: {args.batch_size} | seed: {args.seed} | version_no: {version_nr} | model_path: {args.model_path}")
        version_folder = test(args, version_nr)
        sys.stdout.close()
        os.rename(first_path, os.path.join(version_folder, file_name))
    sys.stdout = open("/dev/stdout", "w")
    print('Classification completed')