import numpy as np
import torch
from tqdm import tqdm
import csv
import os
import imageio
from matplotlib import pyplot as plt
from matplotlib import patches
from skimage.transform import resize
import SimpleITK as sitk
import cv2
from utils_functions.utils import get_data_paths

if torch.cuda.is_available():
    ORIGINAL_DIR_NAME = '/home/flieshout/deep_risk_models/data/AUMC_data'
    ORIGINAL_DIR_NAME_N30 = '/home/flieshout/deep_risk_models/data/AUMC_data_n30'
    CLASSIFICATION_DIR_NAME = '/home/flieshout/deep_risk_models/data/AUMC_classification_data'
    CLASSIFICATION_DIR_NAME_V2 = '/home/flieshout/deep_risk_models/data/AUMC_classification_data_version2'
    CLASSIFICATION_DIR_NAME_V3 = '/home/flieshout/deep_risk_models/data/AUMC_classification_data_version3'
    CLASSIFICATION_DIR_NAME_SUBSET = '/home/flieshout/deep_risk_models/data/AUMC_classification_data_subsample'
    CLASSIFICATION_DIR_NAME_ICM = '/home/flieshout/deep_risk_models/data/AUMC_classification_data_ICM'
    CLASSIFICATION_DIR_NAME_CROSS_VAL = '/home/flieshout/deep_risk_models/data/AUMC_data_classification_5_splits'
    CLASSIFICATION_DIR_NAME_FOLD0 = '/home/flieshout/deep_risk_models/data/AUMC_data_classification_5_splits/fold0'
    CLASSIFICATION_DIR_NAME_FOLD1 = '/home/flieshout/deep_risk_models/data/AUMC_data_classification_5_splits/fold1'
    CLASSIFICATION_DIR_NAME_FOLD2 = '/home/flieshout/deep_risk_models/data/AUMC_data_classification_5_splits/fold2'
    CLASSIFICATION_DIR_NAME_FOLD3 = '/home/flieshout/deep_risk_models/data/AUMC_data_classification_5_splits/fold3'
    CLASSIFICATION_DIR_NAME_FOLD4 = '/home/flieshout/deep_risk_models/data/AUMC_data_classification_5_splits/fold4'
    CLASSIFICATION_DIR_NAME_FOLD0_NEW = '/home/flieshout/deep_risk_models/data/AUMC_data_classification_5_splits_2/fold0'
    CLASSIFICATION_DIR_NAME_FOLD1_NEW = '/home/flieshout/deep_risk_models/data/AUMC_data_classification_5_splits_2/fold1'
    CLASSIFICATION_DIR_NAME_FOLD2_NEW = '/home/flieshout/deep_risk_models/data/AUMC_data_classification_5_splits_2/fold2'
    CLASSIFICATION_DIR_NAME_FOLD3_NEW = '/home/flieshout/deep_risk_models/data/AUMC_data_classification_5_splits_2/fold3'
    CLASSIFICATION_DIR_NAME_FOLD4_NEW = '/home/flieshout/deep_risk_models/data/AUMC_data_classification_5_splits_2/fold4'
else:
    ORIGINAL_DIR_NAME = 'L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis\\labels\\labels_n75'
    ORIGINAL_DIR_NAME_N30 = 'L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis\\labels\\labels_model_testing'
    CLASSIFICATION_DIR_NAME = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_old'
    CLASSIFICATION_DIR_NAME_V2 = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_version2'
    CLASSIFICATION_DIR_NAME_V3 = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_version3'
    CLASSIFICATION_DIR_NAME_SUBSET = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_classification_data_subsample'
    CLASSIFICATION_DIR_NAME_ICM = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_ICM'
    CLASSIFICATION_DIR_NAME_CROSS_VAL = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_5_splits'
    CLASSIFICATION_DIR_NAME_FOLD0 = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_5_splits\fold0'
    CLASSIFICATION_DIR_NAME_FOLD1 = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_5_splits\fold1'
    CLASSIFICATION_DIR_NAME_FOLD2 = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_5_splits\fold2'
    CLASSIFICATION_DIR_NAME_FOLD3 = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_5_splits\fold3'
    CLASSIFICATION_DIR_NAME_FOLD4 = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_5_splits\fold4'
    CLASSIFICATION_DIR_NAME_FOLD0_NEW = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_5_splits_2\fold0'
    CLASSIFICATION_DIR_NAME_FOLD1_NEW = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_5_splits_2\fold1'
    CLASSIFICATION_DIR_NAME_FOLD2_NEW = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_5_splits_2\fold2'
    CLASSIFICATION_DIR_NAME_FOLD3_NEW = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_5_splits_2\fold3'
    CLASSIFICATION_DIR_NAME_FOLD4_NEW = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification_5_splits_2\fold4'
NIFTI_SUFFIX = 'LGE_niftis'
MYO_MASK_SUFFIX = 'myo'
AANKLEURING_MASK_SUFFIX = 'aankleuring'
BOUNDING_BOX_FILE = 'bounding_boxes.csv'
MYO_PREDICTIONS_DIR = 'output'
REMOVE_SLICES_FILE = 'Slices_to_remove.csv'

def read_in_AUMC_data(mode, dataset, resize='crop', size=(132, 132), normalize=['clip']):
    """Reads in the AUMC MRI nifti (.mha) files for a specific split and returns the seperate LGE images, myocardium masks, 
        fibrosis masks and patient id's

    Args:
        mode (str): the split for which the data should be returned. Choices: ['train', 'validation', 'test']
        dataset (str): description of the dataset name
        resize (str, optional): whether to resize or crop the images. Defaults to 'crop'.
        size (tuple, optional): size to which the images are resized or cropped. Defaults to (132, 132).
        normalize (list, optional): normalizations that should be applied to the LGE images. Defaults to ['clip'].

    Raises:
        ValueError: when 'mode' is not in ['train', 'validation', 'test']
        ValueError: when the shapes of the LGE image, myocardium mask and fibrosis mask are inconsistent for a single patient

    Returns:
        tuple: LGE_images (list of numpy arrays), myocardium masks (list of numpy arrays), fibrosis masks (list of numpy arrays), patient_ids (list of strings)
    """    
    if mode not in ['train', 'validation', 'test']:
        raise ValueError("'mode' argument should be either 'train', 'validation' or 'test'")
    if '_30' in dataset: #this uses a subsample of the full dataset
        data_folder_path = os.path.join(ORIGINAL_DIR_NAME_N30, mode, NIFTI_SUFFIX)
        myo_mask_folder_path = os.path.join(ORIGINAL_DIR_NAME_N30, mode, MYO_MASK_SUFFIX)
        aankleuring_mask_folder_path = os.path.join(ORIGINAL_DIR_NAME_N30, mode, AANKLEURING_MASK_SUFFIX)
    else:
        data_folder_path = os.path.join(ORIGINAL_DIR_NAME, mode, NIFTI_SUFFIX)
        myo_mask_folder_path = os.path.join(ORIGINAL_DIR_NAME, mode, MYO_MASK_SUFFIX)
        aankleuring_mask_folder_path = os.path.join(ORIGINAL_DIR_NAME, mode, AANKLEURING_MASK_SUFFIX)

    data_paths = get_data_paths(data_folder_path)
    myo_mask_paths = get_data_paths(myo_mask_folder_path)
    aankleuring_mask_paths = get_data_paths(aankleuring_mask_folder_path)
    no_samples = len(data_paths)

    pat_ids = []
    LGE_imgs = []
    myo_masks = []
    aankleuring_masks = []

    for i, (nifti_path, myo_mask_path, aankleuring_mask_path) in tqdm(enumerate(zip(data_paths, myo_mask_paths, aankleuring_mask_paths)), total=no_samples):
        if '\\' in nifti_path:
            pat_id = nifti_path.split('\\')[-1].split('_')[0]
            pat_ids.append(pat_id)
        elif '/' in nifti_path:
            pat_id = nifti_path.split('/')[-1].split('_')[0]
            pat_ids.append(pat_id)
        LGE_img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_path))
        myo_mask = sitk.GetArrayFromImage(sitk.ReadImage(myo_mask_path))
        aankleuring_mask = sitk.GetArrayFromImage(sitk.ReadImage(aankleuring_mask_path))
        if LGE_img.squeeze().shape != myo_mask.shape or LGE_img.squeeze().shape != aankleuring_mask.shape:
            raise ValueError(f"Inconsistent shapes for {mode}-images {pat_id}. LGE-img: {LGE_img.shape}; myo-mask: {myo_mask.shape}; aankleuring-mask: {aankleuring_mask.shape}. {nifti_path} {myo_mask_path} {aankleuring_mask_path}")
        LGE_img = LGE_img.astype(np.int16)
        myo_mask = myo_mask.astype(np.int16)
        aankleuring_mask = aankleuring_mask.astype(np.int16)
        LGE_img = LGE_img.squeeze()
        myo_mask = myo_mask.squeeze()
        aankleuring_mask = aankleuring_mask.squeeze()

        if resize == 'resize':
            print(f'resizing to {size[0]}x{size[1]}')
            LGE_img, aankleuring_mask, myo_mask = resize_images(LGE_img, aankleuring_mask, myo_mask, size=size)

        LGE_imgs.append(LGE_img)
        myo_masks.append(myo_mask)
        aankleuring_masks.append(aankleuring_mask)
    
    if resize == 'crop':
        if str(size[0]) == 'smallest':
            print('cropping to smallest LGE frames')
            LGE_imgs, aankleuring_masks, myo_masks = crop_imgs(LGE_imgs, aankleuring_masks, myo_masks)
        else:
            print(f'cropping to {size[0]}x{size[1]}')
            LGE_imgs, aankleuring_masks, myo_masks = crop_imgs_size(size[0], size[1], LGE_imgs, aankleuring_masks, myo_masks)

    if dataset in ['AUMC3D', 'AUMC3D_30']: #make sure that every patient has the same number of slices (usually 13)
        LGE_imgs, myo_masks, aankleuring_masks  = cropping_slices(LGE_imgs, pat_ids, myo_masks, aankleuring_masks, dataset=dataset)
        LGE_imgs = normalize_imgs(LGE_imgs, normalize=normalize) #clip values of 1 and 99 percentile and normalize to mean 0 and std 1
        LGE_imgs, myo_masks, aankleuring_masks  = pad_slices(LGE_imgs, myo_masks=myo_masks, aankleuring_masks=aankleuring_masks)
    else:
        LGE_imgs = normalize_imgs(LGE_imgs, normalize=normalize) #clip values of 1 and 99 percentile and normalize to mean 0 and std 1
    
    # plot_bounding_box(LGE_imgs[0], myo_masks[0], [5,6,7], bounding_box_coordinates[0])
    return LGE_imgs, myo_masks, aankleuring_masks, pat_ids

def read_in_fibrosis_AUMC_data(mode, dataset='AUMC2D', myocard_model_version=0, resize='crop', size=(132, 132), crop_slices=True, normalize=['clip']):
    """Reads in and returns the LGE images, the myocardium predictions saved by 'myocard_model_version' and the fibrosis ground truth masks

    Args:
        mode (str): the split for which the data should be returned. Choices: ['train', 'test']
        dataset (str, optional): description of the dataset name. Defaults to 'AUMC2D'.
        myocard_model_version (int, optional): myocard model from which the predictions are obtained. Defaults to 0.
        resize (str, optional): whether to resize or crop the images. Defaults to 'crop'.
        size (tuple, optional): size to which the images are resized or cropped. Defaults to (132, 132).
        crop_slices (boolean, optional): whether to crop the number of slices. Defaults to True.
        normalize (list, optional): normalizations that should be applied to the LGE images. Defaults to ['clip'].

    Raises:
        ValueError: when 'mode' is not in ['train', 'test']
        ValueError: when 'resize' is set to 'resize' but the 'size' parameter does not match the sizes of the myocard prediction from 'myocard_model_version'
        ValueError: when the shapes of the LGE image and fibrosis gt mask are inconsistent for a single patient

    Returns:
        _type_: _description_
    """
    if mode not in ['train', 'test']:
        raise ValueError("'mode' argument should be either 'train' or 'test'")
    if '_30' in dataset:
        data_folder_path = os.path.join(ORIGINAL_DIR_NAME_N30, mode, NIFTI_SUFFIX)
        myo_predictions_folder_path = os.path.join(MYO_PREDICTIONS_DIR, 'AUMC2D_30', 'myocard', f'version_{myocard_model_version}', mode)
        aankleuring_mask_folder_path = os.path.join(ORIGINAL_DIR_NAME_N30, mode, AANKLEURING_MASK_SUFFIX)
    else:
        data_folder_path = os.path.join(ORIGINAL_DIR_NAME, mode, NIFTI_SUFFIX)
        myo_predictions_folder_path = os.path.join(MYO_PREDICTIONS_DIR, 'AUMC2D', 'myocard', f'version_{myocard_model_version}', mode)
    aankleuring_mask_folder_path = os.path.join(ORIGINAL_DIR_NAME, mode, AANKLEURING_MASK_SUFFIX)

    data_paths = get_data_paths(data_folder_path)
    myo_predictions_paths = get_data_paths(myo_predictions_folder_path)
    aankleuring_mask_paths = get_data_paths(aankleuring_mask_folder_path)
    no_samples = len(data_paths)

    shape_myocard = imageio.imread(myo_predictions_paths[0])[:,:,0].shape
    if resize=='resize' and shape_myocard != size:
        raise ValueError(f'The images can only be resized to the shape of the myocard predictions = {shape_myocard}')
    
    myo_predictions_paths = order_myocard_paths(myo_predictions_paths)

    pat_ids = []
    LGE_imgs = []
    myo_predictions = []
    aankleuring_masks = []

    for i, (nifti_path, aankleuring_mask_path) in tqdm(enumerate(zip(data_paths, aankleuring_mask_paths)), total=no_samples):
        if '\\' in nifti_path:
            pat_id = nifti_path.split('\\')[-1].split('_')[0]
            pat_ids.append(pat_id)
        elif '/' in nifti_path:
            pat_id = nifti_path.split('/')[-1].split('_')[0]
            pat_ids.append(pat_id)
        LGE_img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_path))
        aankleuring_mask = sitk.GetArrayFromImage(sitk.ReadImage(aankleuring_mask_path))
        if LGE_img.squeeze().shape != aankleuring_mask.shape:
            raise ValueError(f"Inconsistent shapes for {mode}-images {pat_id}. LGE-img: {LGE_img.shape}; aankleuring-mask: {aankleuring_mask.shape}. {nifti_path} {aankleuring_mask_path}")
        aankleuring_mask = aankleuring_mask.astype(np.int16).squeeze()
        LGE_img = LGE_img.astype(np.int16).squeeze()

        if resize == 'resize':
            LGE_img, aankleuring_mask, _ = resize_images(LGE_img, aankleuring_mask, myo_masks=None, size=size)

        LGE_imgs.append(LGE_img)
        aankleuring_masks.append(aankleuring_mask)
    
    if resize == 'crop':
        if str(size[0]) == 'smallest':
            print('cropping to smallest LGE frames')
            LGE_imgs, aankleuring_masks, _ = crop_imgs(LGE_imgs, aankleuring_masks, myo_masks=None)
        else:
            print(f'cropping to {size[0]}x{size[1]}')
            LGE_imgs, aankleuring_masks, _ = crop_imgs_size(size[0], size[1], LGE_imgs, aankleuring_masks, myo_masks=None)
    
    prev_patid = None
    myo_3d_prediction = []
    for i, path in enumerate(myo_predictions_paths):
        if dataset in ['AUMC2D', 'AUMC2D_30']:
            myocard_prediction = imageio.imread(path)[:,:,0]
            myo_predictions.append(myocard_prediction)
        elif dataset in ['AUMC3D', 'AUMC3D_30']:
            pat_id = path.split('_')[-2]
            if i == len(myo_predictions_paths)-1:
                myocard_prediction = imageio.imread(path)[:,:,0]
                myo_3d_prediction.append(myocard_prediction)
                myo_predictions.append(np.array(myo_3d_prediction))
                break
            elif pat_id != prev_patid and prev_patid is not None:
                myo_predictions.append(np.array(myo_3d_prediction))
                myo_3d_prediction = []
            myocard_prediction = imageio.imread(path)[:,:,0]
            myo_3d_prediction.append(myocard_prediction)
            prev_patid = pat_id

    if dataset in ['AUMC3D', 'AUMC3D_30'] and crop_slices:
        LGE_imgs, myo_masks, aankleuring_masks  = cropping_slices(LGE_imgs, pat_ids, myo_masks, aankleuring_masks, dataset=dataset)
        LGE_imgs = normalize_imgs(LGE_imgs, normalize=normalize) #clip values of 1 and 99 percentile and normalize to mean 0 and std 1
        LGE_imgs, myo_masks, aankleuring_masks  = pad_slices(LGE_imgs, myo_masks=myo_masks, aankleuring_masks=aankleuring_masks)
    else:
        LGE_imgs = normalize_imgs(LGE_imgs, normalize=normalize) #clip values of 1 and 99 percentile and normalize to mean 0 and std 1
    
    return LGE_imgs, myo_predictions, aankleuring_masks, pat_ids

def read_in_AUMC_classification_clinical_data(mode, resize='crop', size=(132, 132), crop_slices=True, normalize=['clip'], mean_values=None, dataset='AUMC3D'):
    if 'fold' in dataset:
        variables_csv_path = os.path.join(CLASSIFICATION_DIR_NAME_CROSS_VAL, 'clinical_features.csv')
    else:
        variables_csv_path = os.path.join(CLASSIFICATION_DIR_NAME, 'clinical_features.csv')

    LGE_imgs, labels, pat_ids, weights = read_in_AUMC_classification_data(mode, resize=resize, size=size, crop_slices=crop_slices, normalize=normalize, dataset=dataset)
    if mode == 'train' and mean_values is None:
        mean_values = get_mean_values(pat_ids, variables_csv_path)
    print('mean values:', mean_values)
    clinical_values = read_in_clinical_values(pat_ids, variables_csv_path, mean_values)
    return LGE_imgs, clinical_values, labels, pat_ids, weights, mean_values

def read_in_AUMC_classification_data(mode, resize='resize', size=(256, 256), crop_slices=(13,11), normalize=['clip'], dataset='AUMC3D'):
    classification_dir_dict = {'AUMC3D' : CLASSIFICATION_DIR_NAME,
                                'AUMC3D_version2' : CLASSIFICATION_DIR_NAME_V2,
                                'AUMC3D_version3' : CLASSIFICATION_DIR_NAME_V3,
                                'AUMC3D_subsample' : CLASSIFICATION_DIR_NAME_SUBSET,
                                'AUMC3D_ICM' : CLASSIFICATION_DIR_NAME_ICM,
                                'AUMC3D_fold0' : CLASSIFICATION_DIR_NAME_FOLD0,
                                'AUMC3D_fold1' : CLASSIFICATION_DIR_NAME_FOLD1,
                                'AUMC3D_fold2' : CLASSIFICATION_DIR_NAME_FOLD2,
                                'AUMC3D_fold3' : CLASSIFICATION_DIR_NAME_FOLD3,
                                'AUMC3D_fold4' : CLASSIFICATION_DIR_NAME_FOLD4,
                                'AUMC3D_fold0_new' : CLASSIFICATION_DIR_NAME_FOLD0_NEW,
                                'AUMC3D_fold1_new' : CLASSIFICATION_DIR_NAME_FOLD1_NEW,
                                'AUMC3D_fold2_new' : CLASSIFICATION_DIR_NAME_FOLD2_NEW,
                                'AUMC3D_fold3_new' : CLASSIFICATION_DIR_NAME_FOLD3_NEW,
                                'AUMC3D_fold4_new' : CLASSIFICATION_DIR_NAME_FOLD4_NEW,
                                }
    if mode not in ['train', 'validation', 'test']:
        raise ValueError("'mode' argument should be either 'train', 'validation' or 'test'")
    classification_dir = classification_dir_dict[dataset]
    print(f'Using datasplit from folder: {classification_dir}')
    data_folder_path = os.path.join(classification_dir, mode)
    if 'fold' in dataset:
        labels_csv_path = os.path.join(CLASSIFICATION_DIR_NAME_CROSS_VAL, 'ICD_therapy_labels.csv')
    else:
        labels_csv_path = os.path.join(classification_dir, 'ICD_therapy_labels.csv')

    data_paths = get_data_paths(data_folder_path)
    no_samples = len(data_paths)

    pat_ids = []
    LGE_imgs = []

    for i, nifti_path in tqdm(enumerate(data_paths), total=no_samples):
        if '\\' in nifti_path:
            pat_id = nifti_path.split('\\')[-1].split('_')[0]
            pat_ids.append(pat_id)
        elif '/' in nifti_path:
            pat_id = nifti_path.split('/')[-1].split('_')[0]
            pat_ids.append(pat_id)
        LGE_img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_path))
        LGE_img = LGE_img.astype(np.int16).squeeze()

        if resize == 'resize':
            print(f'resizing to {size[0]}x{size[1]}')
            raise NotImplementedError()

        LGE_imgs.append(LGE_img)
    
    if resize == 'crop':
        if str(size[0]) == 'smallest':
            print('cropping to smallest LGE frames')
            LGE_imgs, _, _ = crop_imgs(LGE_imgs)
        else:
            print(f'cropping to {size[0]}x{size[1]}')
            LGE_imgs, _, _ = crop_imgs_size(size[0], size[1], LGE_imgs, task='classification')
            
    if crop_slices:
        LGE_imgs, _, _  = cropping_slices(LGE_imgs, pat_ids, task='classification')
        LGE_imgs = normalize_imgs(LGE_imgs, normalize=normalize) #clip values of 1 and 99 percentile and normalize to mean 0 and std 1
        LGE_imgs, _, _ = pad_slices(LGE_imgs)
    else:
        LGE_imgs = normalize_imgs(LGE_imgs, normalize=normalize) #clip values of 1 and 99 percentile and normalize to mean 0 and std 1
    labels = get_therapy_labels(pat_ids, labels_csv_path)
    therapy_labels = [pat_label[0] for pat_label in labels]
    therapy_356days_labels = [pat_label[1] for pat_label in labels]
    mortality_labels = [pat_label[2] for pat_label in labels]
    mortality_365days_labels = [pat_label[3] for pat_label in labels]
    weights_therapy = (len(therapy_labels) - np.sum(therapy_labels)) / np.sum(therapy_labels)
    weights_therapy_365days = (len(therapy_356days_labels) - np.sum(therapy_356days_labels)) / np.sum(therapy_356days_labels)
    weights_mortality = (len(mortality_labels) - np.sum(mortality_labels)) / np.sum(mortality_labels)
    weights_mortality_365days = (len(mortality_365days_labels) - np.sum(mortality_365days_labels)) / np.sum(mortality_365days_labels)
    weights = [weights_therapy, weights_therapy_365days, weights_mortality, weights_mortality_365days]
    return LGE_imgs, labels, pat_ids, weights

def cropping_slices(LGE_imgs, pat_ids, myo_predictions=None, aankleuring_masks=None, dataset='AUMC3D', task='segmentation', height=132, width=132):
    if task == 'segmentation':
        if '_30' in dataset:
            csv_path = os.path.join(ORIGINAL_DIR_NAME_N30, REMOVE_SLICES_FILE)
        else:
            csv_path = os.path.join(ORIGINAL_DIR_NAME, REMOVE_SLICES_FILE)
    elif task=='classification':
        csv_path = os.path.join(CLASSIFICATION_DIR_NAME, REMOVE_SLICES_FILE)
    else:
        raise ValueError(f'Task {task} not recognized')
    remove_slices_dict = {}
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for i, row in enumerate(csv_reader):
            if i == 0:
                patid_index = row.index('SubjectID')
                slice_count_index = row.index('slice_count')
                remove_slices_index = row.index('Slices_to_remove')
            elif row[0].strip() == '':
                break
            else:
                if int(row[slice_count_index]) > 13:
                    remove_slices_string = row[remove_slices_index].split('/')
                    remove_slices_int = [int(s) for s in remove_slices_string]
                    remove_slices_dict[row[patid_index]] = remove_slices_int
    cropped_LGE_imgs, cropped_myo_predictions, cropped_aankleuring_masks = [], [], []
    for i, (LGE_img, pat_id) in enumerate(zip(LGE_imgs, pat_ids)):
        if pat_id in remove_slices_dict:
            LGE_img = np.delete(LGE_img, remove_slices_dict[pat_id], axis=0)
            if myo_predictions is not None:
                myo_mask = np.delete(myo_predictions[i], remove_slices_dict[pat_id], axis=0)
                cropped_myo_predictions.append(myo_mask)
            if aankleuring_masks is not None:
                aankleuring_mask = np.delete(aankleuring_masks[i], remove_slices_dict[pat_id], axis=0)
                cropped_aankleuring_masks.append(aankleuring_mask)
        else:
            if myo_predictions is not None:
                myo_mask = myo_predictions[i]
                cropped_myo_predictions.append(myo_mask)
            if aankleuring_masks is not None:
                aankleuring_mask = aankleuring_masks[i]
                cropped_aankleuring_masks.append(aankleuring_mask)
        if LGE_img.shape[0] > 13:
            raise ValueError(f'LGE stack cannot have more than 13 slices but found {LGE_img.shape[0]}')
        cropped_LGE_imgs.append(LGE_img)
    return cropped_LGE_imgs, cropped_myo_predictions, cropped_aankleuring_masks

def pad_slices(LGE_imgs, height=132, width=132, myo_masks=None, aankleuring_masks=None):
    padded_LGE_imgs, padded_myo_predictions, padded_aankleuring_masks = [], [], []
    for i, LGE_img in enumerate(LGE_imgs):
        slice_padding = 13 - LGE_img.shape[0]
        top_padding = int(slice_padding/2.0)
        bottom_padding = top_padding + LGE_img.shape[0]
        height_padding = height - LGE_img.shape[1]
        top_height_padding = int(height_padding/2.0)
        bottom_height_padding = top_height_padding + LGE_img.shape[1]
        width_padding = height - LGE_img.shape[2]
        left_padding = int(width_padding/2.0)
        right_padding = left_padding + LGE_img.shape[2]
        LGE_img_new = np.zeros((13, width, height), dtype=LGE_img.dtype)
        LGE_img_new[top_padding:bottom_padding, top_height_padding:bottom_height_padding, left_padding:right_padding] = LGE_img
        padded_LGE_imgs.append(LGE_img_new)
        if myo_masks is not None:
            myo_mask = myo_masks[i]
            myo_mask_new = np.zeros((13, LGE_img_new.shape[1], LGE_img_new.shape[2]), dtype=myo_mask.dtype)
            myo_mask_new[top_padding:bottom_padding,:,:] = myo_mask
            padded_myo_predictions.append(myo_mask_new)
        if aankleuring_masks is not None:
            aankleuring_mask = aankleuring_masks[i]
            aankleuring_mask_new = np.zeros((13, LGE_img.shape[1], LGE_img.shape[2]), dtype=aankleuring_mask.dtype)
            aankleuring_mask_new[top_padding:bottom_padding,:,:] = aankleuring_mask
            padded_aankleuring_masks.append(aankleuring_mask_new)
    return padded_LGE_imgs, padded_myo_predictions, padded_aankleuring_masks

def get_mean_values(pat_ids, clinical_variables_path):
    dict_from_csv = {}
    with open(clinical_variables_path, mode='r') as inp:
        reader = csv.reader(inp, delimiter=';')
        for i, rows in enumerate(reader):
            if i != 0:
                #patid: (SubjectID;Geslacht;Age;ICD_EDV_MRI;ICD_ESV_MRI;ICD_LVEF_MRI;ischemisch_aankleuringspatroon;nonischemisch_aankleuringspatroon;overig_aankleuringspatroon)
                dict_from_csv[rows[0]] = [rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7], rows[8]]
    
    # take the mean of the train dataset if value is absent or -99
    train_ECV_MRI_values = [float(values[2]) for pat_id, values in dict_from_csv.items() if (values[2] != '' and str(values[2]) != '-99' and pat_id in pat_ids)]
    train_ESV_MRI_values = [float(values[3]) for pat_id, values in dict_from_csv.items() if (values[3] != '' and str(values[3]) != '-99' and pat_id in pat_ids)]
    train_LVEF_MRI_values = [float(values[4]) for pat_id, values in dict_from_csv.items() if (values[4] != '' and str(values[4]) != '-99' and pat_id in pat_ids)]
    ECV_mean, ESV_mean, LVEF_mean = np.mean(np.array(train_ECV_MRI_values)), np.mean(np.array(train_ESV_MRI_values)), np.mean(np.array(train_LVEF_MRI_values))
    
    return [ECV_mean, ESV_mean, LVEF_mean]

def get_therapy_labels(pat_ids, labels_file_path):
    with open(labels_file_path, mode='r') as inp:
        reader = csv.reader(inp, delimiter=';')
        dict_from_csv = {rows[0]:[rows[1], rows[2], rows[3], rows[4]] for rows in reader} #patid: (therapie, therapie1jaar, mortality, mortality1jaar)
    labels = []
    for pat_id in pat_ids:
        pat_id_labels = dict_from_csv[pat_id]
        labels.append([float(x) for x in pat_id_labels])
    return labels

def read_in_clinical_values(pat_ids, clinical_variables_path, mean_values):
    dict_from_csv = {}
    with open(clinical_variables_path, mode='r') as inp:
        reader = csv.reader(inp, delimiter=';')
        for rows in reader:
            #patid: (SubjectID;Geslacht;Age;ICD_EDV_MRI;ICD_ESV_MRI;ICD_LVEF_MRI;ischemisch_aankleuringspatroon;nonischemisch_aankleuringspatroon;overig_aankleuringspatroon)
            dict_from_csv[rows[0]] = [rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7], rows[8]]
    
    # take the mean of the train dataset if value is absent or -99
    variables = []
    for pat_id in pat_ids:
        pat_id_variables = dict_from_csv[pat_id]
        float_variables = []
        for i, var in enumerate(pat_id_variables):
            if i == 2 and (var == '' or var == '-99'):
                # print(f"replaced ECV '{var}' for patient {pat_id}")
                var = mean_values[0]
            elif i == 3 and (var == '' or var == '-99'):
                # print(f"replaced ESV '{var}' for patient {pat_id}")
                var = mean_values[1]
            elif i == 4 and (var == '' or var == '-99'):
                # print(f"replaced lvef '{var}' for patient {pat_id}")
                var = mean_values[2]
            try:
                float_variables.append(float(var))
            except:
                raise ValueError(f'Problems with {pat_id}, variable {i}: {var}')
        variables.append(float_variables)
    return variables

def normalize_imgs(imgs, normalize):
    normalized_imgs = []
    if 'clip' in normalize and 'scale_before_gamma' in normalize:
        for img in imgs:
            vmin, vmax = np.percentile(img, (1, 99))
            img = img.clip(vmin, vmax)
            mean, std = np.mean(img), np.std(img)
            norm_img = (img-mean)/std
            normalized_imgs.append(norm_img)
    elif 'clip' in normalize:
        for img in imgs:
            vmin, vmax = np.percentile(img, (1, 99))
            img = img.clip(vmin, vmax)
            normalized_imgs.append(img)
    elif 'scale_before_gamma' in normalize:
        for img in imgs:
            mean, std = np.mean(img), np.std(img)
            norm_img = (img-mean)/std
            normalized_imgs.append(norm_img)
    else:
        normalized_imgs = imgs
    return normalized_imgs

def crop_imgs(LGE_images, aankleuring_masks, myo_masks=None):
    smallest_height, smallest_width = 1e6, 1e6
    for LGE_image in LGE_images:
        _, h, w = LGE_image.shape
        if h < smallest_height:
            smallest_height = h
        if w < smallest_width:
            smallest_width = w
    return crop_imgs_size(smallest_height, smallest_width, LGE_images, aankleuring_masks, myo_masks)

def crop_imgs_size(height, width, LGE_images, aankleuring_masks=None, myo_masks=None, task='segmentation'):
    new_LGE_imgs, new_myo_masks, new_aankleuring_masks = [], [], []
    for i in range(len(LGE_images)):
        LGE_image = LGE_images[i]
        if myo_masks is not None:
            myo_mask = myo_masks[i]
        else:
            myo_mask = np.zeros_like(LGE_image)
        if aankleuring_masks is not None:
            aankleuring_mask = aankleuring_masks[i]
        else:
            aankleuring_mask = np.zeros_like(LGE_image)
        _, img_h, img_w = LGE_image.shape
        crop_height = int((img_h - height)/2)
        crop_width = int((img_w - width)/2)
        if task == 'classification' and img_h < height:
            crop_height = 0
        if task == 'classification' and img_w < width:
            crop_width = 0
        if crop_height != 0 and crop_width != 0:
            new_LGE_imgs.append(LGE_image[:, crop_height:-crop_height, crop_width: -crop_width])
            new_myo_masks.append(myo_mask[:, crop_height:-crop_height, crop_width: -crop_width])
            new_aankleuring_masks.append(aankleuring_mask[:, crop_height:-crop_height, crop_width: -crop_width])
        elif crop_height != 0:
            new_LGE_imgs.append(LGE_image[:, crop_height:-crop_height, :])
            new_myo_masks.append(myo_mask[:, crop_height:-crop_height, :])
            new_aankleuring_masks.append(aankleuring_mask[:, crop_height:-crop_height, :])
        elif crop_width != 0:
            new_LGE_imgs.append(LGE_image[:, :, crop_width: -crop_width])
            new_myo_masks.append(myo_mask[:, :, crop_width: -crop_width])
            new_aankleuring_masks.append(aankleuring_mask[:, :, crop_width: -crop_width])
        elif crop_height == 0 and crop_width == 0:
            new_LGE_imgs.append(LGE_image)
            new_myo_masks.append(myo_mask)
            new_aankleuring_masks.append(aankleuring_mask)
        else:
            raise ValueError(f"Invalid calculation for cropping width: {crop_width} and/or cropping heigth: {crop_height}")
        if task=='segmentation':
            if new_LGE_imgs[-1].shape[1] != height or new_LGE_imgs[-1].shape[2] != width:
                raise ValueError(f"LGE image cropping not correct. Should've been cropped to {height}x{width} but got {new_LGE_imgs[-1].shape[1]}x{new_LGE_imgs[-1].shape[2]}")
            if new_myo_masks[-1].shape[1] != height or new_myo_masks[-1].shape[2] != width:
                    raise ValueError(f"LGE image cropping not correct. Should've been cropped to {height}x{width} but got {new_myo_masks[-1].shape[1]}x{new_myo_masks[-1].shape[2]}")
            if new_aankleuring_masks[-1].shape[1] != height or new_aankleuring_masks[-1].shape[2] != width:
                    raise ValueError(f"LGE image cropping not correct. Should've been cropped to {height}x{width} but got {new_aankleuring_masks[-1].shape[1]}x{new_aankleuring_masks[-1].shape[2]}")
    shape_list = [LGE.shape for LGE in new_LGE_imgs]
    # new_LGE_imgs = np.stack(new_LGE_imgs)
    # new_myo_masks = np.stack(new_myo_masks)
    # new_aankleuring_masks = np.stack(new_aankleuring_masks)
    return new_LGE_imgs, new_aankleuring_masks, new_myo_masks

def resize_images(LGE_images, aankleuring_masks, myo_masks=None, size=(256,256)):
    LGE_imgs_new, aankleuring_masks_new, myo_masks_new = np.zeros((LGE_images.shape[0], size[0], size[1])), np.zeros((LGE_images.shape[0], size[0], size[1])), np.zeros((LGE_images.shape[0], size[0], size[1]))
    for i in range(LGE_images.shape[0]):
        LGE_imgs_new[i] = cv2.resize(LGE_images[i], dsize=size, interpolation=cv2.INTER_LINEAR)
        aankleuring_masks_new[i] = cv2.resize(aankleuring_masks[i], dsize=size, interpolation=cv2.INTER_LINEAR)
        if myo_masks is not None:
            myo_masks_new[i] = cv2.resize(myo_masks[i], dsize=size, interpolation=cv2.INTER_LINEAR)
    return LGE_imgs_new, aankleuring_masks_new, myo_masks_new

def order_myocard_paths(myocard_paths):
    myocard_paths.sort()
    ordered_list = []
    list_per_pat = []
    slice_indices_list = []
    prev_pat_id = -1
    for path in myocard_paths:
        pat_id = path.split('prediction_')[-1].split('_')[0]
        if pat_id != prev_pat_id and prev_pat_id != -1:
            list_per_pat = [x for _,x in sorted(zip(slice_indices_list,list_per_pat))]
            ordered_list.extend(list_per_pat)
            list_per_pat = []
            slice_indices_list = []
        prev_pat_id = pat_id
        slice_nr = int(path.split('slice')[-1].split('.png')[0])
        list_per_pat.append(path)
        slice_indices_list.append(slice_nr)
    list_per_pat = [x for _,x in sorted(zip(slice_indices_list,list_per_pat))]
    ordered_list.extend(list_per_pat)
    return ordered_list

if __name__ == '__main__':
    read_in_AUMC_data('train')

