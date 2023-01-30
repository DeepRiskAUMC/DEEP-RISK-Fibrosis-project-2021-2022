import os
import argparse
import sys
import numpy as np
from sklearn.model_selection import cross_validate
import torch
from torchvision.utils import save_image as save_image_utils
from datetime import datetime
import pytorch_lightning as pl
import SimpleITK as sitk
from train_myocard_segmentation_pl import evaluate, SegmentationModel as MyocardModel
# from train_fibrosis_segmentation_pl import SegmentationModel as FibrosisModel
from train_fibrosis_segmentation_with_myo_pl import SegmentationModel as MyoFibModel
from data_loading.load_data import load_data
from data_loading.load_classification_data import load_classification_data
from utils_functions.criterions import dice_score, neg_dice_score, Diceloss, hausdorff_distance, average_hausdorff_distance, get_TPR, get_TNR, get_PPV, get_NPV

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
TEST_NIFTI_FOLDER = '/home/flieshout/deep_risk_models/data/AUMC_data'
TEST_NIFTI_FOLDER_30 = '/home/flieshout/deep_risk_models/data/AUMC_data_n30'
INFERENCE_NIFTI_FOLDER = '/home/flieshout/deep_risk_models/data/AUMC_classification_data'

def create_saving_folder(output_path, dataset, segment_task, version_nr, test_train_or_inference, cross_validate=False):
    dataset_folder = os.path.join(output_path, dataset)
    os.makedirs(dataset_folder, exist_ok=True)
    task_folder = os.path.join(dataset_folder, segment_task)
    os.makedirs(task_folder, exist_ok=True)
    if cross_validate:
        task_folder = os.path.join(task_folder, 'cross_validation')
        os.makedirs(task_folder, exist_ok=True)
    version_folder = os.path.join(task_folder, f"version_{version_nr}")
    os.makedirs(version_folder, exist_ok=True)
    saving_folder = os.path.join(version_folder, test_train_or_inference)
    os.makedirs(saving_folder, exist_ok=True)
    return version_folder, saving_folder

def get_model(segment_task, model_path, bilinear=False):
    if segment_task == 'myocard':
        model = MyocardModel.load_from_checkpoint(model_path)
        fibrosis_model = None
    elif segment_task == 'fibrosis':
        try:
            model = FibrosisModel.load_from_checkpoint(model_path)
        except:
            model = FibrosisModel.load_from_checkpoint(model_path, bilinear=bilinear)
        fibrosis_model = model.model_name
    elif segment_task == 'myofib':
        try:
            model = MyoFibModel.load_from_checkpoint(model_path)
        except:
            model = MyoFibModel.load_from_checkpoint(model_path, bilinear=bilinear)
        fibrosis_model = model.model_name
    else:
        raise ValueError(f'Segmentation task {args.segment_task} not recognized')
    model.eval()
    return model, fibrosis_model

def train(args, version_nr):
    raise NotImplementedError()

def perform_inference(args, version_nr):
    if args.inference_dataset == 'classification_AUMC_train':
        data_loader, _, _, _ = load_classification_data(args.dataset,
                                                            batch_size=args.batch_size,
                                                            val_batch_size='same',
                                                            num_workers=args.num_workers,
                                                            only_test=False,
                                                            transformations=[],
                                                            resize=args.resize,
                                                            size = args.size,
                                                            normalize=args.normalize)
    elif args.inference_dataset == 'classification_AUMC_validation':
        _, data_loader, _, _ = load_classification_data(args.dataset,
                                                            batch_size=args.batch_size,
                                                            val_batch_size='same',
                                                            num_workers=args.num_workers,
                                                            only_test=False,
                                                            transformations=[],
                                                            resize=args.resize,
                                                            size = args.size,
                                                            normalize=args.normalize)
    elif args.inference_dataset == 'classification_AUMC_test':
        _, _, data_loader, _ = load_classification_data(args.dataset,
                                                            batch_size=args.batch_size,
                                                            val_batch_size='same',
                                                            num_workers=args.num_workers,
                                                            only_test=False,
                                                            transformations=[],
                                                            resize=args.resize,
                                                            size = args.size,
                                                            normalize=args.normalize)
    os.makedirs(args.output_path, exist_ok=True)
    inference(args, version_nr, data_loader)

def inference(args, version_nr, data_loader):
    bilinear = True if args.upsampling == 'upsample' else False
    model, _ = get_model(args.segment_task, args.model_path, bilinear=bilinear)
    version_folder, saving_folder = create_saving_folder(args.output_path, args.inference_dataset, args.segment_task, version_nr, args.test_train_or_inference, cross_validate=False)

    for batch in data_loader:
        # prepare input
        LGE_image = batch[0]
        if 'classification_AUMC' in args.inference_dataset:
            pat_id = batch[2]
        else:
            pat_id = batch[3]
        if '2D' in args.dataset:
            slice_nr = batch[4].item()
        else:
            slice_nr = None

         #put input through model
        if args.segment_task == 'myocard':
            raise NotImplementedError()
        else:
            if '2D' in model.model_name and '3D' in args.dataset:
                try:
                    input_datatype = LGE_image.dtype
                except:
                    input_datatype = LGE_image.type()
                device = 'cuda' if LGE_image.is_cuda else 'cpu'
                output = torch.zeros((LGE_image.shape[0], 1, LGE_image.shape[2], LGE_image.shape[3], LGE_image.shape[4]), dtype=input_datatype, device=device)
                for i in range(LGE_image.shape[2]):
                    if torch.any(LGE_image[:,:,i,:,:] > 0):
                        LGE_slice = LGE_image[:,:,i,:,:]
                        fib_pred_slice = model.forward(LGE_slice.float())
                        output[:,:,i,:,:], _ = fib_pred_slice
            elif '3D' in model.model_name and '2D' in args.dataset:
                raise ValueError()
            else:
                output = model.forward(LGE_image.float())

        try:
            sigmoid_finish = model.sigmoid_finish
        except:
            sigmoid_finish = True
        if model.sigmoid_finish:
            prediction = torch.round(output)
        else:
            prediction = torch.round(torch.nn.Sigmoid()(output))
        prediction = prediction.detach()
        output = output.detach()

        if '2D' in args.dataset:
            save_image(prediction, saving_folder, args.dataset, pat_id[0], args.inference_dataset.split('_')[-1], slice_nr=slice_nr, prediction=True, inference=True)
        else:
            save_image(LGE_image, saving_folder, args.dataset, pat_id[0], args.inference_dataset.split('_')[-1], slice_nr=None, prediction=False, inference=True)
            save_image(prediction, saving_folder, args.dataset, pat_id[0], args.inference_dataset.split('_')[-1], slice_nr=None, prediction=True, inference=True)

def test(args, version_nr):
    pl.seed_everything(args.seed)  # To be reproducible

    bilinear = True if args.upsampling == 'upsample' else False
    if 'cross_validation' in args.model_path:
        cross_validation = True
    else:
        cross_validation = False
    model, _ = get_model(args.segment_task, args.model_path, bilinear=bilinear)
    model.to(device)

    # test on either the validation or the test set
    if args.test_train_or_inference == 'test':
        version_folder, saving_folder = create_saving_folder(args.output_path, args.dataset, args.segment_task, version_nr, 'test', cross_validate=cross_validation)
        _, _, test_loader, _ = load_data(dataset=args.dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                resize=args.resize,
                                size=args.size,
                                normalize=args.normalize,
                                only_test=False)
    else:
        version_folder, saving_folder = create_saving_folder(args.output_path, args.dataset, args.segment_task, version_nr, 'validation', cross_validate=cross_validation)
        _, test_loader, _, _ = load_data(dataset=args.dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                resize=args.resize,
                                size=args.size,
                                normalize=args.normalize,
                                only_test=False)
    
    loss_function = Diceloss()
    dice_losses, dice_scores, neg_dice_scores, hausdorff_scores, average_dist_scores, TPR_scores, TNR_scores, PPV_scores, NPV_scores = [], [], [], [], [], [], [], [], []
    if '3D' in args.dataset:
        dice_losses_3d = []
    for batch in test_loader:
        # prepare input
        LGE_image, myocard_mask, fibrosis_mask, pat_id = batch[:4]
        if '2D' in args.dataset:
            slice_nr = batch[4].item()
        else:
            slice_nr = None
        if args.segment_task == 'myocard':
            gt_mask = myocard_mask
            #put input through model
            output = model.forward(LGE_image.float().to(device))
        else:
            gt_mask = fibrosis_mask
            #put input through model
            output, _ = model.forward(LGE_image.float().to(device))

        if 'model.sigmoid_finish' not in locals() or model.sigmoid_finish is None:
            model.sigmoid_finish = True
        if model.sigmoid_finish:
            prediction = torch.round(output)
        else:
            prediction = torch.round(torch.nn.Sigmoid()(output))
        prediction = prediction.detach().cpu()
        output = output.detach().cpu()

        # calculate metrics
        if '3D' in args.dataset:
            dice_losses_3d.append(loss_function(output, gt_mask))
            # dice_loss_list, dice_list, hausdorff_list, average_dist_list, TPR_list, TNR_list, print_string_list
        dice_loss_2d_list, dice_list, neg_dice_list, hausdorff_list, average_dist_list, TPR_list, TNR_list, PPV_list, NPV_list, print_string_list = get_metrics(prediction, gt_mask, args.dataset, pat_id[0], slice_nr)
        neg_dice_list = [x for x in neg_dice_list if x != -1.0]
        dice_losses += dice_loss_2d_list
        dice_scores += dice_list
        neg_dice_scores += neg_dice_list
        hausdorff_scores += hausdorff_list
        average_dist_scores += average_dist_list
        TPR_scores += TPR_list
        TNR_scores += TNR_list
        PPV_scores += PPV_list
        NPV_scores += NPV_list

        # print metrics
        for print_string in print_string_list:
            print(print_string)
        if '3D' in args.dataset:
            print(f'Mean neg dice score: {np.mean(np.array(neg_dice_list))}')

        # save image
        if args.save_images:
            if '2D' in args.dataset:
                save_image(prediction, saving_folder, args.dataset, pat_id[0], args.test_train_or_inference, slice_nr=slice_nr, prediction=True)
            else:
                save_image(LGE_image, saving_folder, args.dataset, pat_id[0], args.test_train_or_inference, slice_nr=None, prediction=False)
                save_image(prediction, saving_folder, args.dataset, pat_id[0], args.test_train_or_inference, slice_nr=None, prediction=True)

    dice_scores_np = np.array(dice_scores)
    # neg_dice_scores = [x for x in neg_dice_scores if x != -1.0]
    neg_dice_scores_np = np.array(neg_dice_scores)
    final_print_string = f"Mean positive dice score: {np.round(np.mean(dice_scores_np), 3)}. Min dice score: {np.min(dice_scores_np)}. Max dice score: {np.max(dice_scores_np)}. Median dice score: {np.median(dice_scores_np)}. Mean negative dice score: {np.round(np.mean(neg_dice_scores_np), 3)}. Mean hausdorff distance: {round(sum(hausdorff_scores)/len(hausdorff_scores), 2)}. Mean average distance: {round(sum(average_dist_scores)/len(average_dist_scores), 2)}. Mean TPR: {round(sum(TPR_scores)/len(TPR_scores), 3)}. Mean TNR: {round(sum(TNR_scores)/len(TNR_scores), 4)}. Mean PPV: {round(sum(PPV_scores)/len(PPV_scores), 4)}. Mean NPV: {round(sum(NPV_scores)/len(NPV_scores), 4)}."
    print(final_print_string)
    print(f"Total slice count: {len(dice_scores)}. Slice count without negative samples: {len(neg_dice_scores)}")
    return version_folder

def get_metrics(prediction, gt_mask, dataset, pat_id, slice_nr=None):
    loss_function = Diceloss()
    if '2D' in dataset:
        dice_loss_list = [loss_function(prediction, gt_mask)]
        dice_list = [dice_score(prediction, gt_mask)]
        neg_dice_list = [neg_dice_score(prediction, gt_mask)]
        hausdorff_list = [hausdorff_distance(prediction, gt_mask)]
        average_dist_list = [average_hausdorff_distance(prediction, gt_mask)]
        TPR_list = [get_TPR(prediction, gt_mask)]
        TNR_list = [get_TNR(prediction, gt_mask)]
        PPV_list = [get_PPV(prediction, gt_mask)]
        NPV_list = [get_NPV(prediction, gt_mask)]
        print_string_list = [f"{pat_id}, slice {slice_nr}. Dice-loss: {dice_loss_list[0]}. Dice score: {dice_list[0]}. Negative dice score: {neg_dice_list[-1]}"]
    else:
        dice_loss_list, dice_list, neg_dice_list, hausdorff_list, average_dist_list, TPR_list, TNR_list, PPV_list, NPV_list, print_string_list = [], [], [], [], [], [], [], [], [], []
        prediction, gt_mask = prediction.squeeze(), gt_mask.squeeze()
        for i in range(prediction.shape[0]):
            dice_loss_list.append(loss_function(prediction[i], gt_mask[i]))
            dice_list.append(dice_score(prediction[i], gt_mask[i]))
            neg_dice_list.append(neg_dice_score(prediction[i], gt_mask[i]))
            hausdorff_list.append(hausdorff_distance(prediction[i], gt_mask[i]))
            average_dist_list.append(average_hausdorff_distance(prediction[i], gt_mask[i]))
            TPR_list.append(get_TPR(prediction[i], gt_mask[i]))
            TNR_list.append(get_TNR(prediction[i], gt_mask[i]))
            PPV_list.append(get_PPV(prediction[i], gt_mask[i]))
            NPV_list.append(get_NPV(prediction[i], gt_mask[i]))
            print_string_list.append(f"{pat_id}, slice {i}. Dice-loss: {dice_loss_list[-1]}. Positive dice score: {dice_list[-1]}. Negative dice score: {neg_dice_list[-1]}")
    return dice_loss_list, dice_list, neg_dice_list, hausdorff_list, average_dist_list, TPR_list, TNR_list, PPV_list, NPV_list, print_string_list

def save_image(image, saving_folder, dataset, pat_id, test_train_or_inference, slice_nr=None, prediction=True, inference=False):
    image = image.squeeze()
    if '2D' in dataset:
        file_name = os.path.join(saving_folder, f"prediction_{pat_id}_slice{slice_nr}.png")
        save_image_utils(image, file_name)
    elif '3D' in dataset:
        # save 2D images
        for i in range(image.shape[0]):
            file_name = os.path.join(saving_folder, f"prediction_{pat_id}_slice{i}.png")
            save_image_utils(image[i], file_name)

        # save 3D images
        nifti_folder = os.path.join(TEST_NIFTI_FOLDER, test_train_or_inference)
        image = image.numpy()
        if prediction is False:
            original_nifti_path = os.path.join(nifti_folder, 'aankleuring', f'{pat_id}_LGE_aankleuring_mask.nrrd')
            file_name = os.path.join(saving_folder, f"LGE_img_{pat_id}.nrrd")
        else:
            original_nifti_path = os.path.join(nifti_folder, 'LGE_niftis', f'{pat_id}_LGE_PSIR.mha')
            file_name = os.path.join(saving_folder, f"prediction_{pat_id}.mha")

        if inference:
            original_nifti_path = os.path.join(INFERENCE_NIFTI_FOLDER, test_train_or_inference, f'{pat_id}_LGE_PSIR.mha')
        original_nifti = sitk.ReadImage(original_nifti_path)
        origin = original_nifti.GetOrigin()
        spacing = original_nifti.GetSpacing()
        image = sitk.GetImageFromArray(image.squeeze())
        image.SetOrigin(origin)
        image.SetSpacing(spacing)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(file_name)
        writer.Execute(image)

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
    parser.add_argument('--upsampling', default='convtrans', type=str,
                        help='What kind of model upsampling we want to use',
                        choices=['upsample', 'convtrans'])
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
    parser.add_argument('--dataset', default='AUMC2D', type=str,
                        help='What dataset to use for the segmentation',
                        choices=['AUMC2D', 'AUMC3D', 'AUMC2D_30', 'AUMC3D_30'])
    parser.add_argument('--inference_dataset', default='classification_AUMC_train', type=str,
                        help='What dataset to use for the segmentation inference',
                        choices=['classification_AUMC_train', 'classification_AUMC_validation', 'classification_AUMC_test'])
                        
    parser.add_argument('--segment_task', default='myocard', type=str,
                        help='What type of tissue to segment',
                        choices=['myocard', 'fibrosis', 'myofib'])
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Minibatch size')
    parser.add_argument('--path_to_inference_data', default='L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis\\labels\\labels_model_testing\\test\LGE_niftis', type=str,
                        help='Path to data to use for inference')
    parser.add_argument('--output_path', default='outputs/segment_output', type=str,
                        help='Path to store the segmented images')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--log_dir', default='segment_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    #write prints to file
    if torch.cuda.is_available():
        version_nr = args.model_path.split('version_')[-1].split('/')[0]
    else:
        version_nr = args.model_path.split('version_')[-1].split('\\')[0]
    print('Segmentation has started!')
    if args.test_train_or_inference in ['test', 'validation']:
        if 'last.ckpt' in args.model_path:
            file_name = f'{args.test_train_or_inference}_last_segmentation_version_{version_nr}.txt'
            args.save_images = False
        else:
            file_name = f'{args.test_train_or_inference}_segmentation_version_{version_nr}.txt'
            args.save_images = True
        first_path = os.path.join(args.output_path, file_name)
        # second_path = os.path.join(args.output_path, f"version_{version_nr}", file_name)
        sys.stdout = open(first_path, "w")
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        print(f"Dataset: {args.dataset} | normalize: {args.normalize} | testing on split: {args.test_train_or_inference} | batch_size: {args.batch_size} | seed: {args.seed} | version_no: {version_nr} | model_path: {args.model_path}")
        version_folder = test(args, version_nr)
        sys.stdout.close()
        os.rename(first_path, os.path.join(version_folder, file_name))
    elif args.test_train_or_inference == 'train':
        train(args, version_nr)
    elif args.test_train_or_inference == 'inference':
        perform_inference(args, version_nr)
    else:
        raise NotImplementedError()
    sys.stdout = open("/dev/stdout", "w")
    print('Segmentation completed')
