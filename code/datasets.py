import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as tudl
from preprocessing import (get_array_from_nifti, myoseg_to_roi,
                           roi_crop_multiple_images)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# In datasets set CHECK_NSLICES to True to make sure the number of slices in a image noted in the excel sheet is correct.
# This makes it slower to initialize the dataset classes, since every image needs to be loaded.
# Advice is to check once with CHECK_NSLICES=True when a change is made, and then leave it on False.
CHECK_NSLICES = False

# If DETERMINISTIC_SPLIT = True, use the below defined hardcoded train,
# validation and test splits for the Deep Risk dataset.
DETERMINISTIC_SPLIT = True
DETERMINISTIC_VAL = ['DRAUMC0046',            'DRAUMC0051',
                     'DRAUMC0063',            'DRAUMC0331',
                     'DRAUMC0790',            'DRAUMC0805',
                     'DRAUMC0809',            'DRAUMC0810',
                     'DRAUMC0891',            'DRAUMC0949',
                     'DRAUMC1049',            'DRAUMC1059']

DETERMINISTIC_TEST = ['DRAUMC0075',            'DRAUMC0184',
                      'DRAUMC0270',            'DRAUMC0310',
                      'DRAUMC0338',            'DRAUMC0380',
                      'DRAUMC0385',            'DRAUMC0411',
                      'DRAUMC0435',            'DRAUMC0503',
                      'DRAUMC0507',            'DRAUMC0518',
                      'DRAUMC0567',            'DRAUMC0585',
                      'DRAUMC0634',            'DRAUMC0635',
                      'DRAUMC0642',            'DRAUMC0673',
                      'DRAUMC0696',            'DRAUMC1017',
                      'DRAUMC1042',            'DRAUMC1155',
                      'DRAUMC1166',            'DRAUMC1199']

DETERMINISTIC_TRAIN = ['DRAUMC0002',            'DRAUMC0008',            'DRAUMC0056',            'DRAUMC0072',
                       'DRAUMC0149',            'DRAUMC0150',            'DRAUMC0169',            'DRAUMC0219',
                       'DRAUMC0222',            'DRAUMC0235',            'DRAUMC0259',            'DRAUMC0286',
                       'DRAUMC0313',            'DRAUMC0315',            'DRAUMC0317',            'DRAUMC0336',
                       'DRAUMC0339',            'DRAUMC0371',            'DRAUMC0388',            'DRAUMC0403',
                       'DRAUMC0419',            'DRAUMC0431',            'DRAUMC0437',            'DRAUMC0446',
                       'DRAUMC0478',            'DRAUMC0481',            'DRAUMC0485',            'DRAUMC0501',
                       'DRAUMC0508',            'DRAUMC0519',            'DRAUMC0521',            'DRAUMC0527',
                       'DRAUMC0528',            'DRAUMC0532',            'DRAUMC0560',            'DRAUMC0583',
                       'DRAUMC0601',            'DRAUMC0632',            'DRAUMC0643',
                       'DRAUMC0656',            'DRAUMC0667',            'DRAUMC0698',
                       'DRAUMC0701',            'DRAUMC0703',            'DRAUMC0725',            'DRAUMC0743',
                       'DRAUMC0749',            'DRAUMC0751',            'DRAUMC0759',            'DRAUMC0768',
                       'DRAUMC0781',            'DRAUMC0787',            'DRAUMC0797',            'DRAUMC0804',
                       'DRAUMC0812',            'DRAUMC0847',            'DRAUMC0868',
                       'DRAUMC0873',            'DRAUMC0881',            'DRAUMC0918',            'DRAUMC0923',
                       'DRAUMC0941',            'DRAUMC0954',            'DRAUMC0985',            'DRAUMC1037',
                       'DRAUMC1065',            'DRAUMC1079',            'DRAUMC1080',            'DRAUMC1082',
                       'DRAUMC1084',            'DRAUMC1085',            'DRAUMC1092',            'DRAUMC1094',
                       'DRAUMC1104',            'DRAUMC1122',            'DRAUMC1159',            'DRAUMC1164',
                       'DRAUMC1217',            'DRAUMC1224',            'DRAUMC1226',            'DRAUMC1253']


class MyopsDataset3D(Dataset):
    """ Dataset class for Myops dataset for stack (3D MRI) data.
    Didn't end up being used, so some changes may need to be made to make it useful,
    for example segmentations are not returned in getitem.
    Probably needs to be (partially) rewritten if we want to use Myops.
    """

    def __init__(self, image_dir, ground_truth_dir, transform=None):
        """ Inputs:
            -   image_dir:          directory with myops images.
            -   ground_truth_dir:   directory with ground truth segmentations.
            -   transform:          (torchvision) transforms for data augmentation.
        """
        # select only LGE images from dataset (denoted by "DE", other modalities are ignored)
        self.image_pathlist = [os.path.join(image_dir, fname)
                               for fname in os.listdir(image_dir) if "DE" in fname]
        self.gd_pathlist = [os.path.join(image_dir, fname)
                            for fname in os.listdir(ground_truth_dir)]
        self.transform = transform

        # create per-slice classification labels,
        # i.e. a binary label whether there is scar in a slice
        labels = []
        for gd_fname in os.listdir(ground_truth_dir):
            gd_path = os.path.join(ground_truth_dir, gd_fname)
            gd_img = sitk.ReadImage(gd_path)
            gd_array = sitk.GetArrayFromImage(gd_img)
            # Myops uses the following values for different tissues in its ground truth:
            #   edema: 1220
            #   scar: 2221
            #   Left Ventricle bloodpool: 500
            #   Right Ventricle bloodpool: 600
            #   Left Ventricle myocardium: 200
            labels.append(list(np.amax((gd_array >= 2221), (1, 2))))

        self.labels = labels
        print("positive labels:", sum([sum(label) for label in labels]))
        print("total labels:", sum([len(label) for label in labels]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """ Returns:
            - img_tensor:   A (transformed) 3D image tensor
            - label:        A list of length-(number of slices) with binary fibrosis labels."""
        img_path = self.image_pathlist[idx]
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)
        img_tensor = torch.Tensor(img_array)
        label = torch.Tensor(self.labels[idx])
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


class MyopsDataset2D(Dataset):
    """Dataset class for Myops dataset for slice (2D MRI) data.
    Didn't end up being used, so some changes may need to be made to make it useful,
    for example segmentations are not returned in getitem.
    Probably needs to be (partially) rewritten if we want to use Myops."""

    def __init__(self, image_dir, ground_truth_dir, transform=None):
        """ Inputs:
            -   image_dir:          directory with myops images.
            -   ground_truth_dir:   directory with ground truth segmentations.
            -   transform:          (torchvision) transforms for data augmentation.
        """
        # select only LGE images from dataset (denoted by "DE", other modalities are ignored)
        self.image_pathlist = [os.path.join(
            image_dir, fname) for fname in os.listdir(image_dir) if "DE" in fname]
        self.gd_pathlist = [os.path.join(image_dir, fname)
                            for fname in os.listdir(ground_truth_dir)]
        self.transform = transform

        # dataset statistics (used to set images to [0, 1] range)
        self.maxvalue = 5060
        self.minvalue = 0

        # create classification labels, a per slice label of whether there are scars present

        # We want to retrieve data per 2D image slice,
        # so store both image path and slice number as tuple.
        self.slice_pathlist = []
        self.labels = []
        for gd_fname, image_path in zip(os.listdir(ground_truth_dir), self.image_pathlist):
            gd_path = os.path.join(ground_truth_dir, gd_fname)
            gd_img = sitk.ReadImage(gd_path)
            gd_array = sitk.GetArrayFromImage(gd_img)
            # Myops uses the following values for different tissues in its ground truth:
            #   edema: 1220
            #   scar: 2221
            #   Left Ventricle bloodpool: 500
            #   Right Ventricle bloodpool: 600
            #   Left Ventricle myocardium: 200
            for slice_idx, gd_slice in enumerate(gd_array):
                self.labels.append(int(np.max((gd_slice >= 2221))))
                self.slice_pathlist.append((image_path, slice_idx))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """ Returns:
            - img_tensor:   A (transformed) 2D image tensor.
            - label:        A binary fibrosis label.
        """
        img_path, slice_idx = self.slice_pathlist[idx]

        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)
        img_tensor = torch.Tensor(
            img_array[slice_idx] / self.maxvalue)[None, :]

        label = self.labels[idx]

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


class EmidecDataset3D(Dataset):
    """Dataset class for Emidec dataset for stack (3D MRI) data."""

    def __init__(self, base_dir, image_mapname="Images", gt_mapname="Contours",
                 transform=None, pred_myo_dir=None, roi_crop="fixed", pred_fib_dir=None):
        """Input:
            -   base_dir:       Directory where Emidec data is located.
            -   image_mapname:  Subfolder containing images, default="Images"
            -   gt_mapname:     Subfolder containing ground truth segmentations,
                                default="Contours"
            -   transform:      (torchvision) transforms for data augmentation, resizing etc.
            - pred_myo_dir:     directory containing myocardium predictions. Required for
                                doing an region of interest crop.
            - roi_crop:         Which type of Region-of-Interest crop to use.
                                Options: 
                                    * "fixed" -> size 100 crop around predicted myocardium center
                                    * "fitted" -> Smallest square crop around predicted myocardium
                                    * anything else -> No cropping
            - pred_fib_dir:     Directory containing fibrosis predictions.    
        """
        self.roi_crop = roi_crop
        # select images and ground truth filenames
        # lists are sorted, so images/gt/pred_myo are in same patient order)
        self.image_pathlist = sorted([str(x) for x in
                                      Path(base_dir).glob(f'**/{image_mapname}/*.nii.gz')])
        self.gt_pathlist = sorted([str(x) for x in
                                   Path(base_dir).glob(f'**/{gt_mapname}/*.nii.gz')])
        if self.gt_pathlist == []:
            self.split = "test"
        else:
            # emidec training set is used, since no ground truth is available for test set
            self.split = "train"

        # search for myocardium prediction files if specified
        if pred_myo_dir != None:
            self.pred_myo_pathlist = sorted(
                [str(x) for x in Path(pred_myo_dir).glob(f'*.nrrd')])
            assert len(self.pred_myo_pathlist) == len(
                self.image_pathlist), f"Number of myocardium predictions and images don't match: {len(self.pred_myo_pathlist)=}, {len(self.image_pathlist)=}"
        else:
            self.pred_myo_pathlist = []

        # search for fibrosis predictions if specified 
        if pred_fib_dir != None:
            self.pred_fib_pathlist = sorted(
                [str(x) for x in Path(pred_fib_dir).glob(f'*.nrrd')])
            assert len(self.pred_fib_pathlist) == len(self.image_pathlist)
        else:
            self.pred_fib_pathlist = []

        self.transform = transform
        self.id_to_idxs = {str(Path(x.stem).stem) for x in Path(
            base_dir).glob(f'**/{image_mapname}/*.nii.gz')}

    def __len__(self):
        return len(self.image_pathlist)

    def __getitem__(self, idx):
        """Returns a dictionary with keys:
            -   img -> transformed image tensor
            -   gt_fib ->  ground truth fibrosis segmentation (binary)
            -   gt_myo  -> ground truth myocardium segmentation (binary)
            -   (optional) pred_fib -> Prediction fibrosis segmentation (confidence between 0 and 1)
            -   (optional) pred_myo -> Prediction myocardium segmentation (confidence between 0 and 1)
            -   crop_corners -> corner location where the crop is made. nan if no crop is done.
            -   original_shape -> shape of image before resizing/cropping.
            -   spacing -> spacing between voxels in original image.
            -   origin -> Origin of the image. Not really necessary for us, but sitk requires
                          an origin. We set the origin to (0, 0, 0, 0). Might be more relevant
                          if you try to combine short axis images with long axis images.
            -   img_path -> path where the image is saved, so you can check it with e.g. mitk.
        """

        # collect image arrays and corresponding names in lists for process_images_3d function
        images, image_names = [], []
        # first, normal image
        img_array, spacing, origin = get_array_from_nifti(
            self.image_pathlist[idx], with_spacing=True, with_origin=True)
        original_shape = img_array.shape
        images.append(img_array)
        image_names.append('img')

        # next, different ground truth segmentations
        gt_array = get_array_from_nifti(self.gt_pathlist[idx])
        # ground truth legend: 
        #   background (0)
        #   cavity(1)
        #   normal myocardium (2)
        #   myocardial infarction (3) 
        #   no-reflow (4))
        #   all_myocardium = union(2, 3, 4)
        gt_no_reflow = gt_array == 4
        gt_fib = gt_array == 3
        images.append(gt_fib)
        image_names.append('gt_fib')
        gt_myo = (gt_array == 2) + gt_fib + gt_no_reflow
        images.append(gt_myo)
        image_names.append('gt_myo')

        # next, different prediction images
        if self.pred_fib_pathlist != []:
            pred_fib = get_array_from_nifti(self.pred_fib_pathlist[idx])
            images.append(pred_fib)
            image_names.append('pred_fib')

        if self.pred_myo_pathlist != []:
            pred_myo = get_array_from_nifti(self.pred_myo_pathlist[idx])
            images.append(pred_myo)
            image_names.append('pred_myo')

        # transform images and add batch information
        data = process_images_3d(
            images, image_names, self.transform, self.roi_crop)
        data['original_shape'] = original_shape
        data['spacing'] = spacing
        data['origin'] = origin
        data['img_path'] = self.image_pathlist[idx]

        return data


def process_images_3d(images, image_names, transform, roi_crop, depth_to_batch=False):
    """ Inputs:
            - images:           list of numpy array images, may include segmentations
            - image_names:      list of strings to describe the images
            - transform:        a transformation on the images
            - roi_crop:         Which type of Region-of-Interest crop to use.
                                Options: 
                                    * "fixed" -> size 100 crop around predicted myocardium center
                                    * "fitted" -> Smallest square crop around predicted myocardium
                                    * anything else -> No cropping
            - depth_to_batch:   If True, move depth (slice) dimension to batch dimension,
                                so it can be given as input to 2D models.

        Returns:
            -   Dictionary with  (1) image_names as keys, with transformed image tensors as values
                                 (2) data['crop_corners'] contains the corner locations of the crop,
                                     which is for example needed to reverse the crop during inference.

        Important for the ordering of images :
        - MRI image must be first (will get normalization, colorjitter augmentation etc)
        - pred_myo must be last, so we know which image to use for region of interest cropping
        - Other predictions/segmentations can go anywhere in between, e.g. 
          fibrosis ground truth/pseudo-ground truth/predictions.
    """
    assert image_names[0] == 'img'

    # perform region of interest crop around myocardium prediction
    if "pred_myo" in image_names and roi_crop in ["fitted", "fixed"]:
        # require pred_myo last for convenience
        assert image_names[-1] == "pred_myo"
        pred_myo = images.pop(-1)
        if roi_crop == "fitted":
            fixed_size = None
        elif roi_crop == "fixed":
            fixed_size = 100
        crop_results = roi_crop_multiple_images(
            pred_myo, images, fixed_size=fixed_size)
        crop_corners = crop_results[-1]
        # cropped pred_myo has been put back in images at the end of the list
        images = list(crop_results)[:-1]
    else:
        crop_corners = float('nan')

    # put depth dimension in batch dimension, which (using torchvision transforms)
    # guarantees that the same 2D transform is applied to each slice
    D, H, W = images[0].shape[-3:]
    images = [torch.from_numpy(x).float().view(D, 1, H, W) for x in images]

    # apply other transformations (e.g. resizing, data augmentations)
    if transform:
        images = transform(images)

    D, _, new_H, new_W = images[0].shape
    if depth_to_batch == False:
        # for 3D models, recover depth dimension from batch dimension and add channel dimension
        images = [x.view(1, D, new_H, new_W) for x in images]
    else:
        images = [x.view(D, 1, new_H, new_W) for x in images]

    data = {name: image for name, image in zip(image_names, images)}
    data['crop_corners'] = crop_corners
    return data


class DeepRiskDataset2D(Dataset):
    """Dataset class for Deep Risk dataset for 2D slices with weak per-slice fibrosis labels."""

    def __init__(self, image_dir, labels_file, seg_labels_dir=None, transform=None,
                 split="train", train_frac=1.0, split_seed=42, myoseg_dir=None,
                 include_no_myo=False, roi_crop="fixed",
                 img_type='PSIR'):
        """Input:
            - image_dir:        Directory where Deep Risk images are located.
            - labels_file:      Excel sheet with (1) Patient ID,
                                and for up to three sequences per patient
                                (2) which slices have no myocardium
                                (3) weak fibrosis labels (which slices have fibrosis)
                                (4) total number of slices per sequence.
            - seg_labels_dir:   (Optional) Directory containing ground truth fibrosis segmentations.
            - transform:        (torchvision) transforms for data augmentation, resizing etc.
            - split:            Data split. Default="train". Options: "train", "val", "test".
            - train_frac:       Proportion of the data to use as training data.
                                Is only used when global variable DETERMINISTIC_SPLIT == False.
            - split_seed:       seed to use when creating a random split.
                                Is only used when global variable DETERMINISTIC_SPLIT == False.
            - myoseg_dir:       Directory containing (predicted) myocardium segmentations.
                                Required for doing a region of interest crop.
            - include_no_myo:   Whether to keep slices without myocardium in the dataset.
                                Default=False. Uses (manually labelled) information from labels_file,
                                since myocardium predictions are probably not accurate enough for this.     
            - roi_crop:         Which type of Region-of-Interest crop to use.
                                Options: 
                                    * "fixed" -> size 100 crop around predicted myocardium center
                                    * "fitted" -> Smallest square crop around predicted myocardium
            - img_type:         Type of image to use. For Deep Risk we have both MAG and PSIR, which
                                should contain the same information (with some different scaling/mapping).
                                We choose one of these to train and evaluate on. Default="PSIR"
        """
        random.seed(split_seed)
        self.transform = transform
        assert split in ["train", "val", "test"]
        if roi_crop not in ["fitted", "fixed"]:
            raise NotImplementedError
        assert img_type in ['PSIR', 'MAG']
        self.roi_crop = roi_crop
        # dataset statistics (can be used for setting images between [0, 1] range)
        self.maxvalue = 4095.0
        self.minvalue = 0
        # load labels
        label_df = pd.read_excel(labels_file)
        # split train val and test data based on patients (rather than slices/mri's)
        PIXEL_LABEL_IDS = ['DRAUMC0008', 'DRAUMC0051', 'DRAUMC0072', 'DRAUMC0075',
                           'DRAUMC0172', 'DRAUMC0219', 'DRAUMC0235',
                           'DRAUMC0286', 'DRAUMC0315',
                           'DRAUMC0331', 'DRAUMC0336', 'DRAUMC0338',
                           'DRAUMC0371', 'DRAUMC0380', 'DRAUMC0411', 'DRAUMC0419', 'DRAUMC0431',
                           'DRAUMC0435', 'DRAUMC0437',
                           'DRAUMC0478', 'DRAUMC0481', 'DRAUMC0485', 'DRAUMC0501',
                           'DRAUMC0507', 'DRAUMC0508', 'DRAUMC0527', 'DRAUMC0528',
                           'DRAUMC0532', 'DRAUMC0560', 'DRAUMC0567', 'DRAUMC0583',
                           'DRAUMC0601',
                           'DRAUMC0634', 'DRAUMC0642', 'DRAUMC0656', 'DRAUMC0667',
                           'DRAUMC0673',
                           'DRAUMC0725', 'DRAUMC0743', 'DRAUMC0749', 'DRAUMC0751', 'DRAUMC0759',
                           'DRAUMC0768', 'DRAUMC0797', 'DRAUMC0804', 'DRAUMC0805',
                           'DRAUMC0810', 'DRAUMC0847',
                           'DRAUMC0868', 'DRAUMC0891', 'DRAUMC0923',
                           'DRAUMC0941', 'DRAUMC0949',
                           'DRAUMC0985', 'DRAUMC1017',
                           'DRAUMC1037', 'DRAUMC1042',
                           'DRAUMC1049', 'DRAUMC1059', 'DRAUMC1065', 'DRAUMC1080',
                           'DRAUMC1082', 'DRAUMC1084', 'DRAUMC1085',
                           'DRAUMC1092', 'DRAUMC1094', 'DRAUMC1122',
                           'DRAUMC1159', 'DRAUMC1164', 'DRAUMC1166', 'DRAUMC1199',
                           'DRAUMC1217', 'DRAUMC1224', 'DRAUMC1226', 'DRAUMC1253']
        # always choose the same test set, 20%
        MYO_TEST_IDS = ["DRAUMC0075", "DRAUMC0172", "DRAUMC0338", "DRAUMC0380", "DRAUMC0411", "DRAUMC0435",
                        "DRAUMC0507", "DRAUMC0567", "DRAUMC0634", "DRAUMC0642", "DRAUMC0673", "DRAUMC1017",
                        "DRAUMC1042", "DRAUMC1166", "DRAUMC1199"]  # "DRAUMC1199" onduidelijk
        if DETERMINISTIC_SPLIT == False:
            TEST_IDS = MYO_TEST_IDS
            TEST_IDS.extend(["DRAUMC0051", "DRAUMC0805", "DRAUMC0891",
                            "DRAUMC1049", "DRAUMC0008", "DRAUMC0949"])
            TRAIN_VAL_PIXEL_LABEL_IDS = [
                id for id in PIXEL_LABEL_IDS if id not in TEST_IDS]
            VAL_IDS = random.sample(TRAIN_VAL_PIXEL_LABEL_IDS, int(
                len(TRAIN_VAL_PIXEL_LABEL_IDS) * (1 - train_frac)))
        else:
            TEST_IDS = DETERMINISTIC_TEST
            VAL_IDS = DETERMINISTIC_VAL

        if split == "test":
            label_df = label_df[label_df['SubjectID'].isin(TEST_IDS)]
            print(f"Test patients: {len(label_df)}")
            print(f"{TEST_IDS=}")
        elif split == "val":
            label_df = label_df[label_df['SubjectID'].isin(VAL_IDS)]
            print(f"Validation patients: {len(label_df)}")
            print(f"{VAL_IDS=}")
        elif split == "train":
            label_df = label_df[~label_df['SubjectID'].isin(TEST_IDS)]
            label_df = label_df[~label_df['SubjectID'].isin(VAL_IDS)]
            print(f"Train patients: {len(label_df)}")

        # select only labeled patients
        label_df = label_df[label_df.Gelabeled == 'Ja']
        print(f'{len(label_df)=}')

        self.slice_pathlist = []
        self.labels = []
        self.myoseg_pathlist = []
        self.fibrosis_seg_pathlist = []
        if seg_labels_dir != None:
            all_fibrosis_seg_files = [x for x in seg_labels_dir.glob('*.nrrd')]
        else:
            all_fibrosis_seg_files = []

        for _, row in tqdm(label_df.iterrows()):
            n_slices = row.n_slices.split('|')
            assert len(n_slices) == 3
            for i, n_slices_seq in enumerate(n_slices):
                # skip empty/non-existent sequences
                if n_slices_seq == '-':
                    continue
                else:
                    n_slices_seq = int(n_slices_seq)

                subjectID = row["SubjectID"]
                seq_labels = row["ICD_LGE_slices_seq" + str(i)].split(';')
                seq_no_myo = row["Geen_myo_slices_seq" + str(i)].split(';')

                image_path = [x for x in image_dir.glob(
                    f"{subjectID}*seq{i}*{img_type}*.mha") if not "old" in str(x)]
                assert len(
                    image_path) != 0, f"Found no image file for sequence \n {subjectID}:seq{i}"
                assert len(
                    image_path) == 1, f"Found multiple files for a sequence \n {image_path}"
                image_path = image_path[0]
                sequence_identifier = str(
                    Path(image_path).stem).split(img_type)[0]
                # add fibrosis segmentation if it exists
                fibrosis_seg_file = [
                    fibrosis_seg for fibrosis_seg in all_fibrosis_seg_files if sequence_identifier in fibrosis_seg.stem]

                if myoseg_dir != None:
                    myoseg_f = [f for f in myoseg_dir.glob(
                        f"{sequence_identifier}*.nrrd")]
                    assert len(
                        myoseg_f) != 0, f"Found no myocard segmentation file for sequence \n {sequence_identifier}"
                    assert len(
                        myoseg_f) == 1, f"Found multiple myocard segmentations for a sequence \n {image_path}"
                    myoseg_f = myoseg_f[0]

                if CHECK_NSLICES:
                    img = sitk.ReadImage(str(image_path))
                    img_array = sitk.GetArrayFromImage(img)
                    if n_slices_seq != len(img_array[0]):
                        print(f"{image_path=}")
                        print(
                            f"Error: Number of slices {n_slices_seq} and {len(img_array[0])} does not match")
                
                # label each slice
                for slice_idx in range(n_slices_seq):
                    if str(slice_idx) in seq_no_myo and include_no_myo == False:
                        pass
                    else:
                        # add path to fibrosis segmentation list if available
                        if len(fibrosis_seg_file) > 0:
                            assert len(fibrosis_seg_file) == 1
                            self.fibrosis_seg_pathlist.append(
                                str(fibrosis_seg_file[0]))
                        else:
                            self.fibrosis_seg_pathlist.append(None)

                        # add weak label and per-slice (image path, slice index) tuple
                        if str(slice_idx) in seq_labels:
                            self.labels.append(1)
                            self.slice_pathlist.append(
                                (str(image_path), slice_idx))
                        else:
                            self.labels.append(0)
                            self.slice_pathlist.append(
                                (str(image_path), slice_idx))

                        # add filenames of myocard segmentations
                        if myoseg_dir != None:
                            self.myoseg_pathlist.append(str(myoseg_f))

    def __len__(self):
        return len(self.labels)

    def get_n_fibrotic_gt_slices(self):
        """ Returns total number of slices with fibrosis label and ground truth fibrosis segmentations."""
        n_fibrotic_slices = 0
        for idx, (_) in enumerate(self.slice_pathlist):
            if self.fibrosis_seg_pathlist[idx] != None and self.labels[idx] == 1:
                n_fibrotic_slices += 1
        return n_fibrotic_slices

    def get_n_gt_slices(self):
        """ Returns total number of slices with ground truth fibrosis segmentations (also slices without fibrosis)."""
        n_slices = 0
        for f in self.fibrosis_seg_pathlist:
            if f != None:
                n_slices += 1
        return n_slices

    def __getitem__(self, idx):
        """Returns a dictionary with keys:
            - img                   ->  transformed image tensor
            - label                 ->  Binary fibrosis label (over entire slice)
            - fibrosis_seg_label    ->  Ground truth fibrosis segmentation (originally binary, but with interpolation).
                                        Values set to -1 if no fibrosis segmentation exists.
            - (optional) myo_seg    ->  Prediction myocardium segmentation (confidence between 0 and 1).
                                        Values set to -1 if no myocardium segmentation exists.
            - crop_corners          ->  corner location where the crop is made. nan if no crop is done.
            - original_shape        ->  shape of image before resizing/cropping.
            - spacing               ->  spacing between voxels in original image.
            - origin                ->  Origin of the image. Not really necessary for us, but sitk requires
                                        an origin. We set the origin to (0, 0, 0, 0). Might be more relevant
                                        if you try to combine short axis images with long axis images.
            - img_path              ->  path where the image is saved, so you can check it with e.g. mitk.
            - slice_idx             ->  slice within the image at img_path.
        """
        # select slice from sitk image
        image_path, slice_idx = self.slice_pathlist[idx]
        img = sitk.ReadImage(image_path)
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        img_array = sitk.GetArrayFromImage(img)
        img_array = img_array.astype(np.float32)[:, slice_idx]
        original_shape = img_array.shape

        # select slice from fibrosis labels if available
        if self.fibrosis_seg_pathlist[idx] != None:
            fibrosis_seg_file = self.fibrosis_seg_pathlist[idx]
            fibrosis_seg_img = sitk.ReadImage(fibrosis_seg_file)
            fibrosis_seg_array = sitk.GetArrayFromImage(fibrosis_seg_img)
            fibrosis_seg_array = fibrosis_seg_array.astype('float32')
            fibrosis_seg_array = fibrosis_seg_array[slice_idx][None, ...]
        else:
            fibrosis_seg_array = -1 * np.ones_like(img_array)  # "unavailable"

        # take roi cropping if specified
        if self.myoseg_pathlist != []:
            myoseg_f = self.myoseg_pathlist[idx]
            myoseg = sitk.ReadImage(str(myoseg_f))
            myoseg_arr_3d = sitk.GetArrayFromImage(myoseg)[0]
            myoseg_arr = myoseg_arr_3d.astype(np.float32)[slice_idx][None, ...]
            if self.roi_crop == "fitted":
                img_array, crop_corners = myoseg_to_roi(
                    img_array, np.copy(myoseg_arr_3d), fixed_size=None)
                myoseg_arr, crop_corners2 = myoseg_to_roi(
                    myoseg_arr, np.copy(myoseg_arr_3d), fixed_size=None)
                fibrosis_seg_array, crop_corners3 = myoseg_to_roi(
                    fibrosis_seg_array, np.copy(myoseg_arr_3d), fixed_size=None)
            elif self.roi_crop == "fixed":
                img_array, crop_corners = myoseg_to_roi(
                    img_array, np.copy(myoseg_arr_3d), fixed_size=100)
                myoseg_arr, crop_corners2 = myoseg_to_roi(
                    myoseg_arr, np.copy(myoseg_arr_3d), fixed_size=100)
                fibrosis_seg_array, crop_corners3 = myoseg_to_roi(
                    fibrosis_seg_array, np.copy(myoseg_arr_3d), fixed_size=100)
            assert crop_corners == crop_corners2 == crop_corners3, f"{crop_corners=}{crop_corners2=}{crop_corners3=}"
        else:
            myoseg_arr = -1.0 * np.ones_like(img_array)  # "unavailable"
            crop_corners = float('nan')

        # convert to tensor
        img_tensor = torch.from_numpy(img_array / self.maxvalue)
        fibrosis_seg_tensor = torch.from_numpy(fibrosis_seg_array)
        myoseg_tensor = torch.from_numpy(myoseg_arr)

        pre_transform = [img_tensor, fibrosis_seg_tensor, myoseg_tensor]

        label = self.labels[idx]

        # apply other transformations (e.g. resampling, data augs)
        if self.transform:
            post_transform = self.transform(pre_transform)
            img_tensor, fibrosis_seg_tensor, myoseg_tensor = post_transform[
                0], post_transform[1], post_transform[2]

        data = {'img_path': image_path, 'slice_idx': slice_idx,
                'fibrosis_seg_label': fibrosis_seg_tensor, 'myo_seg': myoseg_tensor,
                'img': img_tensor, 'label': label,
                'crop_corners': crop_corners, 'original_shape': original_shape,
                'origin': origin, 'spacing': spacing}
        return data


class DeepRiskDataset3D(Dataset):
    """Dataset class for Deep Risk dataset for 3D stacks with weak per-slice fibrosis labels."""

    def __init__(self, image_dir, labels_file,
                 seg_labels_dir=None, transform=None,
                 split="train", train_frac=1.0, split_seed=42,
                 myoseg_dir=None, include_no_myo=False, roi_crop="fixed",
                 min_slices=7, max_slices=15, img_type='PSIR'):
        """Input:
            - image_dir:        Directory where Deep Risk images are located.
            - labels_file:      Excel sheet with (1) Patient ID,
                                and for up to three sequences per patient
                                (2) which slices have no myocardium
                                (3) weak fibrosis labels (which slices have fibrosis)
                                (4) total number of slices per sequence.
            - seg_labels_dir:   (Optional) Directory containing ground truth fibrosis segmentations.
            - transform:        (torchvision) transforms for data augmentation, resizing etc.
            - split:            Data split. Default="train". Options: "train", "val", "test".
            - train_frac:       Proportion of the data to use as training data.
                                Is only used when global variable DETERMINISTIC_SPLIT == False.
            - split_seed:       seed to use when creating a random split.
                                Is only used when global variable DETERMINISTIC_SPLIT == False.
            - myoseg_dir:       Directory containing (predicted) myocardium segmentations.
                                Required for doing a region of interest crop.
            - include_no_myo:   Whether to keep slices without myocardium in the dataset.
                                Default=False. Uses (manually labelled) information from labels_file,
                                since myocardium predictions are probably not accurate enough for this.     
            - roi_crop:         Which type of Region-of-Interest crop to use.
                                Options: 
                                    * "fixed" -> size 100 crop around predicted myocardium center
                                    * "fitted" -> Smallest square crop around predicted myocardium
            - img_type:         Type of image to use. For Deep Risk we have both MAG and PSIR, which
                                should contain the same information (with some different scaling/mapping).
                                We choose one of these to train and evaluate on. Default="PSIR"
            - min_slices:       Mimimum number of slices for a sequence/image to be included in the
                                dataset. 3D models probably don't work well for sequence with few slices
                                or large(r) distance between slices. Default=7.
            - max_slices:       Maximum number of slices for a sequence/image to be included in the
                                dataset. Prevents running into out-of-memory errors.   
        """
        random.seed(split_seed)
        self.transform = transform
        assert split in ["train", "val", "test"]
        self.split = split
        if roi_crop not in ["fitted", "fixed"]:
            raise NotImplementedError
        assert img_type in ['PSIR', 'MAG']
        self.roi_crop = roi_crop
        # dataset statistics (can be used for setting images between [0, 1] range)
        self.maxvalue = 4095.0
        self.minvalue = 0
        # load labels
        label_df = pd.read_excel(labels_file)
        # split train val and test data based on patients (rather than slices/mri's)
        PIXEL_LABEL_IDS = ['DRAUMC0008', 'DRAUMC0051', 'DRAUMC0072', 'DRAUMC0075',
                           'DRAUMC0172', 'DRAUMC0219', 'DRAUMC0235',
                           'DRAUMC0286', 'DRAUMC0315',
                           'DRAUMC0331', 'DRAUMC0336', 'DRAUMC0338',
                           'DRAUMC0371', 'DRAUMC0380', 'DRAUMC0411', 'DRAUMC0419', 'DRAUMC0431',
                           'DRAUMC0435', 'DRAUMC0437',
                           'DRAUMC0478', 'DRAUMC0481', 'DRAUMC0485', 'DRAUMC0501',
                           'DRAUMC0507', 'DRAUMC0508', 'DRAUMC0527', 'DRAUMC0528',
                           'DRAUMC0532', 'DRAUMC0560', 'DRAUMC0567', 'DRAUMC0583',
                           'DRAUMC0601',
                           'DRAUMC0634', 'DRAUMC0642', 'DRAUMC0656', 'DRAUMC0667',
                           'DRAUMC0673',
                           'DRAUMC0725', 'DRAUMC0743', 'DRAUMC0749', 'DRAUMC0751', 'DRAUMC0759',
                           'DRAUMC0768', 'DRAUMC0797', 'DRAUMC0804', 'DRAUMC0805',
                           'DRAUMC0810', 'DRAUMC0847',
                           'DRAUMC0868', 'DRAUMC0891', 'DRAUMC0923',
                           'DRAUMC0941', 'DRAUMC0949',
                           'DRAUMC0985', 'DRAUMC1017',
                           'DRAUMC1037', 'DRAUMC1042',
                           'DRAUMC1049', 'DRAUMC1059', 'DRAUMC1065', 'DRAUMC1080',
                           'DRAUMC1082', 'DRAUMC1084', 'DRAUMC1085',
                           'DRAUMC1092', 'DRAUMC1094', 'DRAUMC1122',
                           'DRAUMC1159', 'DRAUMC1164', 'DRAUMC1166', 'DRAUMC1199',
                           'DRAUMC1217', 'DRAUMC1224', 'DRAUMC1226', 'DRAUMC1253']

        MYO_TEST_IDS = ["DRAUMC0075", "DRAUMC0172", "DRAUMC0338", "DRAUMC0380", "DRAUMC0411", "DRAUMC0435",
                        "DRAUMC0507", "DRAUMC0567", "DRAUMC0634", "DRAUMC0642", "DRAUMC0673", "DRAUMC1017",
                        "DRAUMC1042", "DRAUMC1166", "DRAUMC1199"]  # "DRAUMC1199" onduidelijk

        if DETERMINISTIC_SPLIT == False:
            TEST_IDS = MYO_TEST_IDS
            TEST_IDS.extend(["DRAUMC0051", "DRAUMC0805", "DRAUMC0891",
                            "DRAUMC1049", "DRAUMC0008", "DRAUMC0949"])
            TRAIN_VAL_PIXEL_LABEL_IDS = [
                id for id in PIXEL_LABEL_IDS if id not in TEST_IDS]
            VAL_IDS = random.sample(TRAIN_VAL_PIXEL_LABEL_IDS, int(
                len(TRAIN_VAL_PIXEL_LABEL_IDS) * (1 - train_frac)))
        else:
            TEST_IDS = DETERMINISTIC_TEST
            VAL_IDS = DETERMINISTIC_VAL

        if split == "test":
            label_df = label_df[label_df['SubjectID'].isin(TEST_IDS)]
            print(f"Test patients: {len(label_df)}")
            print(f"{TEST_IDS=}")
        elif split == "val":
            label_df = label_df[label_df['SubjectID'].isin(VAL_IDS)]
            print(
                f"Validation patients: {len(label_df)} {label_df['SubjectID']}")
            print(f"{VAL_IDS=}")
        elif split == "train":
            label_df = label_df[~label_df['SubjectID'].isin(TEST_IDS)]
            label_df = label_df[~label_df['SubjectID'].isin(VAL_IDS)]
            print(f"Train patients: {len(label_df)}")

        # select only labeled patients
        label_df = label_df[label_df.Gelabeled == 'Ja']
        print(f'{len(label_df)=}')

        self.image_pathlist = []
        self.labels = []
        self.myoseg_pathlist = []
        self.fibrosis_seg_pathlist = []
        self.max_num_slices = max_slices
        self.no_myo_list = []

        if seg_labels_dir != None:
            all_fibrosis_seg_files = [x for x in seg_labels_dir.glob('*.nrrd')]
        else:
            all_fibrosis_seg_files = []

        if myoseg_dir != None:
            all_myoseg_files = [x for x in myoseg_dir.glob("*.nrrd")]
        else:
            all_myoseg_files = []

        for _, row in tqdm(label_df.iterrows()):
            n_slices = row.n_slices.split('|')
            assert len(n_slices) == 3
            for i, n_slices_seq in enumerate(n_slices):
                # skip empty/non-existent/too small/too large sequences
                if n_slices_seq == '-' or int(n_slices_seq) < min_slices or int(n_slices_seq) > max_slices:
                    continue
                else:
                    n_slices_seq = int(n_slices_seq)

                subjectID = row["SubjectID"]
                seq_pos_labels = row["ICD_LGE_slices_seq" + str(i)].split(';')
                seq_no_myo = row["Geen_myo_slices_seq" + str(i)].split(';')

                # add image path
                image_path = [x for x in image_dir.glob(
                    f"{subjectID}*seq{i}*{img_type}*.mha") if not "old" in str(x)]
                assert len(image_path) != 0, "Found no image file for sequence"
                assert len(image_path) == 1, f"Found multiple files for a sequence \n {image_path}"
                image_path = image_path[0]
                sequence_identifier = str(Path(image_path).stem).split(img_type)[0]

                if CHECK_NSLICES:
                    img = sitk.ReadImage(str(image_path))
                    img_array = sitk.GetArrayFromImage(img)
                    if n_slices_seq != len(img_array[0]):
                        print(f"{image_path=}")
                        print(f"Warning: Number of slices {n_slices_seq} and {len(img_array[0])} does not match")
                self.image_pathlist.append(str(image_path))

                # add slice-level classification labels
                seq_labels = []
                for i in range(n_slices_seq):
                    if include_no_myo == False and str(i) in seq_no_myo:
                        seq_labels.append("TO_BE_REMOVED")
                    elif str(i) in seq_pos_labels:
                        seq_labels.append(1)
                    else:
                        seq_labels.append(0)
                self.labels.append(seq_labels)

                # add fibrosis segmentation path if it exists
                fibrosis_seg_file = [
                    fibrosis_seg for fibrosis_seg in all_fibrosis_seg_files if sequence_identifier in fibrosis_seg.stem]
                if len(fibrosis_seg_file) > 0:
                    assert len(fibrosis_seg_file) == 1
                    self.fibrosis_seg_pathlist.append(
                        str(fibrosis_seg_file[0]))
                else:
                    self.fibrosis_seg_pathlist.append(None)

                # add myoseg path if specified
                if myoseg_dir != None:
                    myoseg_f = [
                        myoseg for myoseg in all_myoseg_files if sequence_identifier in myoseg.stem]
                    assert len(
                        myoseg_f) != 0, "Found no myocard segmentation file for sequence"
                    assert len(
                        myoseg_f) == 1, f"Found multiple myocard segmentations for a sequence \n {image_path}"
                    myoseg_f = myoseg_f[0]
                    self.myoseg_pathlist.append(str(myoseg_f))

    def __len__(self):
        return len(self.labels)

    def add_transform(self, transform):
        """Function to add/change the transform after initialization."""
        self.transform = transform

    def __getitem__(self, idx):
        """Returns a dictionary with keys:
            - img                   ->  transformed image tensor.
                                        Padded with zero values along slice dimension.
            - label                 ->  Binary fibrosis label (over entire slice).
                                        Padded with nan along slice dimension.
            - fibrosis_seg_label    ->  Ground truth fibrosis segmentation (originally binary, but with interpolation).
                                        Values set to -1 if no fibrosis segmentation exists.
                                        Padded with zero values along slice dimension.
            - (optional) myo_seg    ->  Prediction myocardium segmentation (confidence between 0 and 1).
                                        Values set to -1 if no myocardium segmentation exists.
                                        Padded with zero values along slice dimension.
            - crop_corners          ->  corner location where the crop is made. nan if no crop is done.
            - original_shape        ->  shape of image before resizing/cropping.
            - spacing               ->  spacing between voxels in original image.
            - origin                ->  Origin of the image. Not really necessary for us, but sitk requires
                                        an origin. We set the origin to (0, 0, 0, 0). Might be more relevant
                                        if you try to combine short axis images with long axis images.
            - img_path              ->  path where the image is saved, so you can check it with e.g. mitk.
        """
        # load image stack
        image_path = self.image_pathlist[idx]
        img = sitk.ReadImage(image_path)
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        img_array = sitk.GetArrayFromImage(img)
        img_array = img_array.astype(np.float32)
        original_shape = img_array.shape
        # load fibrosis labels if available
        if self.fibrosis_seg_pathlist[idx] != None:
            fibrosis_seg_file = self.fibrosis_seg_pathlist[idx]
            fibrosis_seg_img = sitk.ReadImage(fibrosis_seg_file)
            fibrosis_seg_array = sitk.GetArrayFromImage(fibrosis_seg_img)
            fibrosis_seg_array = fibrosis_seg_array.astype('float32')
            fibrosis_seg_array = fibrosis_seg_array[None, ...]
        else:
            fibrosis_seg_array = -1 * np.ones_like(img_array)  # "unavailable"

        # load myo segmentation if specified
        if self.myoseg_pathlist != []:
            myoseg_f = self.myoseg_pathlist[idx]
            myoseg = sitk.ReadImage(str(myoseg_f))
            myoseg_array = sitk.GetArrayFromImage(myoseg).astype(np.float32)

            assert myoseg_array.shape[-2] == img_array.shape[-2] == fibrosis_seg_array.shape[
                -2], f"shapes {myoseg_array.shape}, {img_array.shape}, {fibrosis_seg_array.shape} don't match for sequence {image_path}"
            assert myoseg_array.shape[-1] == img_array.shape[-1] == fibrosis_seg_array.shape[
                -1], f"shapes {myoseg_array.shape}, {img_array.shape}, {fibrosis_seg_array.shape} don't match for sequence {image_path}"
            # take roi cropping if specified
            if self.roi_crop == "fitted":
                img_array, crop_corners = myoseg_to_roi(
                    img_array, myoseg_array, fixed_size=None)
                fibrosis_seg_array, crop_corners2 = myoseg_to_roi(
                    fibrosis_seg_array, myoseg_array, fixed_size=None)
                myoseg_array, crop_corners3 = myoseg_to_roi(
                    myoseg_array, myoseg_array, fixed_size=None)
            elif self.roi_crop == "fixed":
                img_array, crop_corners = myoseg_to_roi(
                    img_array, myoseg_array, fixed_size=100)
                fibrosis_seg_array, crop_corners2 = myoseg_to_roi(
                    fibrosis_seg_array, myoseg_array, fixed_size=100)
                myoseg_array, crop_corners3 = myoseg_to_roi(
                    myoseg_array, myoseg_array, fixed_size=100)
            assert crop_corners == crop_corners2 == crop_corners3, f"{crop_corners=}{crop_corners2=}{crop_corners3=}"
        else:
            myoseg_array = -1.0 * np.ones_like(img_array)  # "unavailable"
            crop_corners = float('nan')

        img_tensor = torch.from_numpy(img_array / self.maxvalue)
        fibrosis_seg_tensor = torch.from_numpy(fibrosis_seg_array)
        myoseg_tensor = torch.from_numpy(myoseg_array)

        label = self.labels[idx]
        # remove slices without myocard if specified
        remove_no_myo = np.array([l == "TO_BE_REMOVED" for l in label])
        if any(remove_no_myo):
            img_tensor, fibrosis_seg_tensor, myoseg_tensor = img_tensor[..., ~remove_no_myo, :,
                                                                        :], fibrosis_seg_tensor[..., ~remove_no_myo, :, :], myoseg_tensor[..., ~remove_no_myo, :, :]
            label = [l for l in label if l != "TO_BE_REMOVED"]

        # pad images along slice dimension for equal dataloader size within batch
        # padding classification labels with nan -> ignore slices in loss
        padding = self.max_num_slices - img_tensor.shape[1]
        img_tensor = F.pad(img_tensor, pad=(
            0, 0, 0, 0, 0, padding), mode='constant', value=0.0)
        fibrosis_seg_tensor = F.pad(fibrosis_seg_tensor, pad=(
            0, 0, 0, 0, 0, padding), mode='constant', value=0.0)
        myoseg_tensor = F.pad(myoseg_tensor, pad=(
            0, 0, 0, 0, 0, padding), mode='constant', value=0.0)
        padded_label = label + [float('nan')]*padding
        padded_label = torch.Tensor(padded_label)

        # apply other transformations (e.g. resizing, data augs)
        if self.transform:
            post_transform = self.transform(
                [img_tensor, fibrosis_seg_tensor, myoseg_tensor])
            img_tensor, fibrosis_seg_tensor, myoseg_tensor = post_transform[
                0], post_transform[1], post_transform[2]

        data = {'img_path': image_path,
                'fibrosis_seg_label': fibrosis_seg_tensor, 'myo_seg': myoseg_tensor,
                'img': img_tensor, 'label': padded_label,
                'crop_corners': crop_corners, 'original_shape': original_shape,
                'origin': origin, 'spacing': spacing}
        return data


class DeepRiskDatasetSegmentation2D(Dataset):
    """Dataset class for Deep Risk dataset for 2D slice with (pseudo) fibrosis segmentations).
    Used for training the weakly-supervised fibrosis segmentation network.
    """

    def __init__(self, image_dir, labels_file, pseudoseg_dir,
                 gt_seg_dir=None, transform=None, split="train",
                 train_frac=1.0, split_seed=42, myoseg_dir=None,
                 gt_myoseg_dir=None,
                 include_no_myo=False, roi_crop="fixed",
                 pixel_gt_only=False, img_type='PSIR'):
        """Input:
            - image_dir:        Directory where Deep Risk images are located.
            - labels_file:      Excel sheet with (1) Patient ID,
                                and for up to three sequences per patient
                                (2) which slices have no myocardium
                                (3) weak fibrosis labels (which slices have fibrosis)
                                (4) total number of slices per sequence.
            - pseudoseg_dir:    Directory containing pseudo ground truth fibrosis segmentations
                                for every image.
            - gt_seg_dir:       (Optional) Directory containing ground truth fibrosis segmentations,
                                not necessarily for every image.
            - transform:        (torchvision) transforms for data augmentation, resizing etc.
            - split:            Data split. Default="train". Options: "train", "val", "test".
            - train_frac:       Proportion of the data to use as training data.
                                Is only used when global variable DETERMINISTIC_SPLIT == False.
            - split_seed:       seed to use when creating a random split.
                                Is only used when global variable DETERMINISTIC_SPLIT == False.
            - myoseg_dir:       (Optional) Directory containing predicted myocardium segmentations.
                                Required for doing a region of interest crop.
            - gt_myoseg_dir:    (Optional) Directory containing ground truth myocardium segmentations,
                                not necessarily for every image.
            - pixel_gt_only:    Whether to only include images that have ground truth segmentations.
            - include_no_myo:   Whether to keep slices without myocardium in the dataset.
                                Default=False. Uses (manually labelled) information from labels_file,
                                since myocardium predictions are probably not accurate enough for this.     
            - roi_crop:         Which type of Region-of-Interest crop to use.
                                Options: 
                                    * "fixed" -> size 100 crop around predicted myocardium center
                                    * "fitted" -> Smallest square crop around predicted myocardium
            - img_type:         Type of image to use. For Deep Risk we have both MAG and PSIR, which
                                should contain the same information (with some different scaling/mapping).
                                We choose one of these to train and evaluate on. Default="PSIR"
        """
        random.seed(split_seed)
        self.transform = transform
        assert split in ["train", "val", "test"]
        if roi_crop not in ["fitted", "fixed"]:
            raise NotImplementedError
        assert img_type in ['PSIR', 'MAG']
        self.roi_crop = roi_crop
        # dataset statistics (can be used for setting images between [0, 1] range)
        self.maxvalue = 4095.0
        self.minvalue = 0
        # load labels
        label_df = pd.read_excel(labels_file)
        # split train val and test data based on patients (rather than slices/mri's)
        PIXEL_LABEL_IDS = ['DRAUMC0008', 'DRAUMC0051', 'DRAUMC0072', 'DRAUMC0075',
                           'DRAUMC0172', 'DRAUMC0219', 'DRAUMC0235',
                           'DRAUMC0286', 'DRAUMC0315',
                           'DRAUMC0331', 'DRAUMC0336', 'DRAUMC0338',
                           'DRAUMC0371', 'DRAUMC0380', 'DRAUMC0411', 'DRAUMC0419', 'DRAUMC0431',
                           'DRAUMC0435', 'DRAUMC0437',
                           'DRAUMC0478', 'DRAUMC0481', 'DRAUMC0485', 'DRAUMC0501',
                           'DRAUMC0507', 'DRAUMC0508', 'DRAUMC0527', 'DRAUMC0528',
                           'DRAUMC0532', 'DRAUMC0560', 'DRAUMC0567', 'DRAUMC0583',
                           'DRAUMC0601',
                           'DRAUMC0634', 'DRAUMC0642', 'DRAUMC0656', 'DRAUMC0667',
                           'DRAUMC0673',
                           'DRAUMC0725', 'DRAUMC0743', 'DRAUMC0749', 'DRAUMC0751', 'DRAUMC0759',
                           'DRAUMC0768', 'DRAUMC0797', 'DRAUMC0804', 'DRAUMC0805',
                           'DRAUMC0810', 'DRAUMC0847',
                           'DRAUMC0868', 'DRAUMC0891', 'DRAUMC0923',
                           'DRAUMC0941', 'DRAUMC0949',
                           'DRAUMC0985', 'DRAUMC1017',
                           'DRAUMC1037', 'DRAUMC1042',
                           'DRAUMC1049', 'DRAUMC1059', 'DRAUMC1065', 'DRAUMC1080',
                           'DRAUMC1082', 'DRAUMC1084', 'DRAUMC1085',
                           'DRAUMC1092', 'DRAUMC1094', 'DRAUMC1122',
                           'DRAUMC1159', 'DRAUMC1164', 'DRAUMC1166', 'DRAUMC1199',
                           'DRAUMC1217', 'DRAUMC1224', 'DRAUMC1226', 'DRAUMC1253']

        MYO_TEST_IDS = ["DRAUMC0075", "DRAUMC0172", "DRAUMC0338", "DRAUMC0380", "DRAUMC0411", "DRAUMC0435",
                        "DRAUMC0507", "DRAUMC0567", "DRAUMC0634", "DRAUMC0642", "DRAUMC0673", "DRAUMC1017",
                        "DRAUMC1042", "DRAUMC1166", "DRAUMC1199"]  # "DRAUMC1199" onduidelijk

        if DETERMINISTIC_SPLIT == False:
            TEST_IDS = MYO_TEST_IDS
            TEST_IDS.extend(["DRAUMC0051", "DRAUMC0805", "DRAUMC0891",
                            "DRAUMC1049", "DRAUMC0008", "DRAUMC0949"])
            TRAIN_VAL_PIXEL_LABEL_IDS = [
                id for id in PIXEL_LABEL_IDS if id not in TEST_IDS]
            VAL_IDS = random.sample(TRAIN_VAL_PIXEL_LABEL_IDS, int(
                len(TRAIN_VAL_PIXEL_LABEL_IDS) * (1 - train_frac)))
        else:
            TEST_IDS = DETERMINISTIC_TEST
            VAL_IDS = DETERMINISTIC_VAL
            PIXEL_LABEL_IDS = DETERMINISTIC_TRAIN + DETERMINISTIC_VAL + DETERMINISTIC_TEST

        if split == "test":
            label_df = label_df[label_df['SubjectID'].isin(TEST_IDS)]
            print(f"Test patients: {len(label_df)}")
            print(f"{TEST_IDS=}")
        elif split == "val":
            label_df = label_df[label_df['SubjectID'].isin(VAL_IDS)]
            print(f"Validation patients: {len(label_df)}")
            print(f"{VAL_IDS=}")
        elif split == "train":
            if pixel_gt_only:
                label_df = label_df[label_df['SubjectID'].isin(
                    PIXEL_LABEL_IDS)]
            label_df = label_df[~label_df['SubjectID'].isin(TEST_IDS)]
            label_df = label_df[~label_df['SubjectID'].isin(VAL_IDS)]
            print(f"Train patients: {len(label_df)}")

        # select only labeled patients
        label_df = label_df[label_df.Gelabeled == 'Ja']
        print(f'{len(label_df)=}')

        self.slice_pathlist = []
        self.pseudoseg_pathlist = []
        self.myoseg_pathlist = []
        self.gt_myoseg_pathlist = []
        self.gt_seg_pathlist = []
        self.id_to_idxs = defaultdict(list)
        if gt_seg_dir != None:
            all_gt_seg_files = [x for x in gt_seg_dir.glob('*.nrrd')]
        else:
            all_gt_seg_files = []

        all_pseudoseg_files = [x for x in pseudoseg_dir.glob('*.nrrd')]
        print(f"{len(all_pseudoseg_files)=}")

        for _, row in tqdm(label_df.iterrows()):
            n_slices = row.n_slices.split('|')
            assert len(n_slices) == 3
            for i, n_slices_seq in enumerate(n_slices):
                # skip empty/non-existent sequences
                if n_slices_seq == '-':
                    continue
                else:
                    n_slices_seq = int(n_slices_seq)

                subjectID = row["SubjectID"]
                seq_labels = row["ICD_LGE_slices_seq" + str(i)].split(';')
                seq_no_myo = row["Geen_myo_slices_seq" + str(i)].split(';')

                image_path = [x for x in image_dir.glob(
                    f"{subjectID}*seq{i}*{img_type}*.mha") if not "old" in str(x)]
                assert len(image_path) != 0, "Found no image file for sequence"
                assert len(
                    image_path) == 1, f"Found multiple files for a sequence \n {image_path}"
                image_path = image_path[0]
                sequence_identifier = str(
                    Path(image_path).stem).split(img_type)[0]

                # add fibrosis segmentation if it exists
                gt_seg_file = [f for f in all_gt_seg_files if sequence_identifier in f.stem]
                if pixel_gt_only and len(gt_seg_file) == 0:
                    continue

                pseudoseg_file = [f for f in all_pseudoseg_files if sequence_identifier in f.stem]
                if len(pseudoseg_file) == 0:
                    print(f"Skipped sequence without pseudolabel: {image_path=} {n_slices_seq=}")
                    continue
                assert len(pseudoseg_file) != 0, f"Found no pseudolabel file for sequence {image_path=} {all_pseudoseg_files=}"
                assert len(pseudoseg_file) == 1, f"Found multiple pseudolabel files for a sequence \n {pseudoseg_file}"
                pseudoseg_file = pseudoseg_file[0]

                if myoseg_dir != None:
                    myoseg_f = [f for f in myoseg_dir.glob(
                        f"{sequence_identifier}*.nrrd")]
                    assert len(myoseg_f) != 0, f"Found no myocard segmentation file for sequence"
                    assert len(myoseg_f) == 1, f"Found multiple myocard segmentations for a sequence \n {image_path}"
                    myoseg_f = myoseg_f[0]

                if gt_myoseg_dir != None:
                    gt_myoseg_file = [f for f in gt_myoseg_dir.glob(
                        f"{sequence_identifier}*.nrrd")]
                else:
                    gt_myoseg_file = []

                if CHECK_NSLICES:
                    img = sitk.ReadImage(str(image_path))
                    img_array = sitk.GetArrayFromImage(img)
                    if n_slices_seq != len(img_array[0]):
                        print(f"{image_path=}")
                        print(f"Error: Number of slices {n_slices_seq} and {len(img_array[0])} does not match")
                # setup paths for each slice
                for slice_idx in range(n_slices_seq):
                    if str(slice_idx) in seq_no_myo and include_no_myo == False:
                        pass
                    else:
                        # add path to ground truth segmentation if available
                        if len(gt_seg_file) > 0:
                            assert len(gt_seg_file) == 1, f"Multiple ground truth files: {gt_seg_file}"
                            self.gt_seg_pathlist.append(str(gt_seg_file[0]))
                        else:
                            self.gt_seg_pathlist.append(None)
                            if pixel_gt_only == True or not split == "train":
                                print(f"File {image_path} has no ground truth")
                        # add ground truth myocardium segmentation if available
                        if len(gt_myoseg_file) > 0:
                            assert len(gt_myoseg_file) == 1, f"Multiple ground truth files: {gt_myoseg_file}"
                            self.gt_myoseg_pathlist.append(str(gt_myoseg_file[0]))
                        else:
                            self.gt_myoseg_pathlist.append(None)

                        # add pseudo seg and path to image
                        self.slice_pathlist.append((str(image_path), slice_idx))
                        self.pseudoseg_pathlist.append(str(pseudoseg_file))
                        # find filenames of myocard segmentations -> for roi cropping
                        if myoseg_dir != None:
                            self.myoseg_pathlist.append(str(myoseg_f))
                        # create mapping ID : idxs, in order to find all data for a patient in get_patient
                        self.id_to_idxs[subjectID].append(len(self.slice_pathlist)-1)

    def __len__(self):
        return len(self.slice_pathlist)

    def get_n_fibrotic_gt_slices(self):
        n_fibrotic_slices = 0
        for pid in self.id_to_idxs:
            gt_seg = self.get_patient_batch(pid)['gt_fib']
            for gt_seg_slice in gt_seg:
                if (gt_seg_slice.max() > 0.001):
                    n_fibrotic_slices += 1
        return n_fibrotic_slices

    def __getitem__(self, idx):
        """Returns a dictionary with keys:
            - img                   ->  transformed image tensor
            - pseudo_fib            ->  pseudo ground truth fibrosis segmentation.
            - gt_fib                ->  Ground truth fibrosis segmentation (originally binary, but with interpolation).
                                        Values set to -1 if no fibrosis segmentation exists.
            - gt_myo                ->  Ground truth myocardium segmentation.
                                        Values set to -1 if it doesn't exist.
            - pred_myo              ->  Prediction myocardium segmentation (confidence between 0 and 1).
                                        Values set to -1 if no myocardium segmentation exists.
            - crop_corners          ->  corner location where the crop is made. nan if no crop is done.
            - original_shape        ->  shape of image before resizing/cropping.
            - spacing               ->  spacing between voxels in original image.
            - origin                ->  Origin of the image. Not really necessary for us, but sitk requires
                                        an origin. We set the origin to (0, 0, 0, 0). Might be more relevant
                                        if you try to combine short axis images with long axis images.
            - img_path              ->  path where the image is saved, so you can check it with e.g. mitk.
            - slice_idx             ->  slice within the image at img_path.
        """
        # select slice from sitk image
        image_path, slice_idx = self.slice_pathlist[idx]

        img = sitk.ReadImage(image_path)
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        img_array = sitk.GetArrayFromImage(img)
        img_array = img_array.astype(np.float32)[:, slice_idx]
        original_shape = img_array.shape
        # select slice from pseudolabel
        pseudoseg_file = self.pseudoseg_pathlist[idx]
        pseudoseg_img = sitk.ReadImage(pseudoseg_file)
        pseudoseg_array = sitk.GetArrayFromImage(pseudoseg_img)
        pseudoseg_array = pseudoseg_array.astype(
            np.float32)[:, slice_idx]  # pseudoseg_array.astype('float32')
        # pseudoseg_array = #pseudoseg_array[slice_idx][None, ...]

        # select slice from fibrosis labels if available
        if self.gt_seg_pathlist[idx] != None:
            gt_seg_file = self.gt_seg_pathlist[idx]
            gt_seg_img = sitk.ReadImage(gt_seg_file)
            gt_seg_array = sitk.GetArrayFromImage(gt_seg_img)
            gt_seg_array = gt_seg_array.astype('float32')
            gt_seg_array = gt_seg_array[slice_idx][None, ...]
        else:
            gt_seg_array = -1 * np.ones_like(img_array)  # "unavailable"
        # select slice from gt myoseg if available
        if self.gt_myoseg_pathlist[idx] != None:
            gt_myoseg_file = self.gt_myoseg_pathlist[idx]
            gt_myoseg_img = sitk.ReadImage(gt_myoseg_file)
            gt_myoseg_array = sitk.GetArrayFromImage(gt_myoseg_img)
            gt_myoseg_array = gt_myoseg_array.astype('float32')
            gt_myoseg_array = gt_myoseg_array[slice_idx][None, ...]
        else:
            gt_myoseg_array = -1 * np.ones_like(img_array)

        assert pseudoseg_array.shape == img_array.shape == gt_seg_array.shape == gt_myoseg_array.shape, f"Shapes {pseudoseg_array.shape}, {img_array.shape}, {gt_seg_array.shape}, {gt_myoseg_array.shape} don't match"
        # take roi cropping if specified
        if self.myoseg_pathlist != []:
            myoseg_f = self.myoseg_pathlist[idx]
            myoseg = sitk.ReadImage(str(myoseg_f))
            myoseg_arr_3d = sitk.GetArrayFromImage(myoseg)[0]
            myoseg_arr = myoseg_arr_3d.astype(np.float32)[slice_idx][None, ...]
            if self.roi_crop == "fitted":
                img_array, crop_corners = myoseg_to_roi(
                    img_array, np.copy(myoseg_arr_3d), fixed_size=None)
                myoseg_arr, crop_corners2 = myoseg_to_roi(
                    myoseg_arr, np.copy(myoseg_arr_3d), fixed_size=None)
                pseudoseg_array, crop_corners3 = myoseg_to_roi(
                    pseudoseg_array, np.copy(myoseg_arr_3d), fixed_size=None)
                gt_seg_array, crop_corners4 = myoseg_to_roi(
                    gt_seg_array, np.copy(myoseg_arr_3d), fixed_size=None)
                gt_myoseg_array, crop_corners5 = myoseg_to_roi(
                    gt_myoseg_array, np.copy(myoseg_arr_3d), fixed_size=None)
            elif self.roi_crop == "fixed":
                img_array, crop_corners = myoseg_to_roi(
                    img_array, np.copy(myoseg_arr_3d), fixed_size=100)
                myoseg_arr, crop_corners2 = myoseg_to_roi(
                    myoseg_arr, np.copy(myoseg_arr_3d), fixed_size=100)
                pseudoseg_array, crop_corners3 = myoseg_to_roi(
                    pseudoseg_array, np.copy(myoseg_arr_3d), fixed_size=100)
                gt_seg_array, crop_corners4 = myoseg_to_roi(
                    gt_seg_array, np.copy(myoseg_arr_3d), fixed_size=100)
                gt_myoseg_array, crop_corners5 = myoseg_to_roi(
                    gt_myoseg_array, np.copy(myoseg_arr_3d), fixed_size=100)
            assert crop_corners == crop_corners2 == crop_corners3 == crop_corners4 == crop_corners5, f"{crop_corners=}{crop_corners2=}{crop_corners3=}{crop_corners4=}{crop_corners5=}"
        else:
            myoseg_arr = -1.0 * np.ones_like(img_array)  # "unavailable"
            crop_corners = float('nan')

        img_tensor = torch.from_numpy(img_array / self.maxvalue)
        pseudoseg_tensor = torch.from_numpy(pseudoseg_array)
        gt_seg_tensor = torch.from_numpy(gt_seg_array)
        myoseg_tensor = torch.from_numpy(myoseg_arr)
        gt_myoseg_tensor = torch.from_numpy(gt_myoseg_array)

        pre_transform = [img_tensor, pseudoseg_tensor,
                         gt_seg_tensor, myoseg_tensor, gt_myoseg_tensor]
        # apply other transformations (e.g. resampling, data augs)
        if self.transform:
            post_transform = self.transform(pre_transform)
            img_tensor, pseudoseg_tensor, gt_seg_tensor, myoseg_tensor, gt_myoseg_tensor = post_transform[
                0], post_transform[1], post_transform[2], post_transform[3], post_transform[4]

        data = {'img_path': image_path, 'slice_idx': slice_idx,
                'pseudo_fib': pseudoseg_tensor, 'gt_fib': gt_seg_tensor,
                'pred_myo': myoseg_tensor, 'gt_myo': gt_myoseg_tensor,
                'img': img_tensor,
                'crop_corners': crop_corners, 'original_shape': original_shape,
                'origin': origin, 'spacing': spacing}
        return data

    def get_patient_batch(self, pid):
        """Returns batch with slices from a specific patient id."""
        assert pid in self.id_to_idxs
        idxs = self.id_to_idxs[pid]
        data_list = [self.__getitem__(idx) for idx in idxs]
        keys = data_list[0].keys()

        batch = tudl.default_collate(data_list)
        return batch


class DeepRiskDatasetMyoSegmentation2D(Dataset):
    """Dataset class for Deep Risk dataset for 2D slices with myocardium segmentations."""
    def __init__(self, image_dir, gt_seg_dir,
                 transform=None, split="train",
                 train_frac=1.0, split_seed=42,
                 include_no_myo=False, img_type='PSIR', gt_type='PSIR'):
        """Input:
            - image_dir:        Directory where Deep Risk images are located.
            - gt_seg_dir:       Directory containing ground truth myocardium segmentations.
            - transform:        (torchvision) transforms for data augmentation, resizing etc.
            - split:            Data split. Default="train". Options: "train", "val", "test".
            - train_frac:       Proportion of the data to use as training data.
                                Is only used when global variable DETERMINISTIC_SPLIT == False.
            - split_seed:       seed to use when creating a random split.
                                Is only used when global variable DETERMINISTIC_SPLIT == False.
            - include_no_myo:   Whether to keep slices without myocardium in the dataset.
                                Default=False. Uses (manually labelled) information from
                                ground truth segmentations.
            - img_type:         Type of image to use. For Deep Risk we have both MAG and PSIR, which
                                should contain the same information (with some different scaling/mapping).
                                We choose one of these to train and evaluate on. Default="PSIR"
            - gt_type:          On which type the ground truth segmentation are drawn.
                                Is used to match ground truth filenames with image filenames.
        """
        random.seed(split_seed)
        self.transform = transform
        assert split in ["train", "val", "test"]
        assert img_type in ['PSIR', 'MAG']
        assert gt_type in ['PSIR', 'MAG']
        # dataset statistics (can be used for setting images between [0, 1] range)
        self.maxvalue = 4095.0
        self.minvalue = 0
        # split train val and test data based on patients (rather than slices/mri's)
        PIXEL_LABEL_IDS = ['DRAUMC0008', 'DRAUMC0051', 'DRAUMC0072', 'DRAUMC0075',
                           'DRAUMC0172', 'DRAUMC0219', 'DRAUMC0235',
                           'DRAUMC0286', 'DRAUMC0315',
                           'DRAUMC0331', 'DRAUMC0336', 'DRAUMC0338',
                           'DRAUMC0371', 'DRAUMC0380', 'DRAUMC0411', 'DRAUMC0419', 'DRAUMC0431',
                           'DRAUMC0435', 'DRAUMC0437',
                           'DRAUMC0478', 'DRAUMC0481', 'DRAUMC0485', 'DRAUMC0501',
                           'DRAUMC0507', 'DRAUMC0508', 'DRAUMC0527', 'DRAUMC0528',
                           'DRAUMC0532', 'DRAUMC0560', 'DRAUMC0567', 'DRAUMC0583',
                           'DRAUMC0601',
                           'DRAUMC0634', 'DRAUMC0642', 'DRAUMC0656', 'DRAUMC0667',
                           'DRAUMC0673',
                           'DRAUMC0725', 'DRAUMC0743', 'DRAUMC0749', 'DRAUMC0751', 'DRAUMC0759',
                           'DRAUMC0768', 'DRAUMC0797', 'DRAUMC0804', 'DRAUMC0805',
                           'DRAUMC0810', 'DRAUMC0847',
                           'DRAUMC0868', 'DRAUMC0891', 'DRAUMC0923',
                           'DRAUMC0941', 'DRAUMC0949',
                           'DRAUMC0985', 'DRAUMC1017',
                           'DRAUMC1037', 'DRAUMC1042',
                           'DRAUMC1049', 'DRAUMC1059', 'DRAUMC1065', 'DRAUMC1080',
                           'DRAUMC1082', 'DRAUMC1084', 'DRAUMC1085',
                           'DRAUMC1092', 'DRAUMC1094', 'DRAUMC1122',
                           'DRAUMC1159', 'DRAUMC1164', 'DRAUMC1166', 'DRAUMC1199',
                           'DRAUMC1217', 'DRAUMC1224', 'DRAUMC1226', 'DRAUMC1253']
        # always choose the same test set, 20%
        MYO_TEST_IDS = ["DRAUMC0075", "DRAUMC0172", "DRAUMC0338", "DRAUMC0380", "DRAUMC0411", "DRAUMC0435",
                        "DRAUMC0507", "DRAUMC0567", "DRAUMC0634", "DRAUMC0642", "DRAUMC0673", "DRAUMC1017",
                        "DRAUMC1042", "DRAUMC1166", "DRAUMC1199"]  # "DRAUMC1199" onduidelijk

        if DETERMINISTIC_SPLIT == False:
            TEST_IDS = MYO_TEST_IDS
            TEST_IDS.extend(["DRAUMC0051", "DRAUMC0805", "DRAUMC0891",
                            "DRAUMC1049", "DRAUMC0008", "DRAUMC0949"])
            TRAIN_VAL_PIXEL_LABEL_IDS = [
                id for id in PIXEL_LABEL_IDS if id not in TEST_IDS]
            VAL_IDS = random.sample(TRAIN_VAL_PIXEL_LABEL_IDS, int(
                len(TRAIN_VAL_PIXEL_LABEL_IDS) * (1 - train_frac)))
            TRAIN_IDS = [
                id for id in TRAIN_VAL_PIXEL_LABEL_IDS if id not in VAL_IDS]
        else:
            TEST_IDS = DETERMINISTIC_TEST
            VAL_IDS = DETERMINISTIC_VAL
            TRAIN_IDS = DETERMINISTIC_TRAIN

        if split == "test":
            ids = TEST_IDS
            print(f"Test patients: {len(TEST_IDS)}")
        elif split == "val":
            ids = VAL_IDS
            print(f"Validation patients: {len(VAL_IDS)}")
        elif split == "train":
            ids = TRAIN_IDS
            print(f"Train patients: {len(TRAIN_IDS)}")

        self.gt_seg_pathlist = []
        self.image_pathlist = []
        self.id_to_idxs = defaultdict(list)

        for subjectID in ids:
            gt_seg_path = [x for x in gt_seg_dir.glob(f'*{subjectID}*.nrrd')]
            assert len(
                gt_seg_path) != 0, f"Found no segmentation file for patient {subjectID}"
            assert len(
                gt_seg_path) == 1, f'Found multiple files for patient\n {gt_seg_path}'
            gt_seg_path = gt_seg_path[0]
            sequence_identifier = str(Path(gt_seg_path).stem).split(gt_type)[0]

            image_path = [x for x in image_dir.glob(
                f"{sequence_identifier}*{img_type}*.mha") if not "old" in str(x)]
            assert len(
                image_path) != 0, f"Found no image file for sequence \n {sequence_identifier}"
            assert len(
                image_path) == 1, f"Found multiple files for a sequence \n {image_path}"
            image_path = image_path[0]

            gt_seg = sitk.ReadImage(str(gt_seg_path))
            gt_seg_array = sitk.GetArrayFromImage(gt_seg)
            n_slices_seq = len(gt_seg_array)

            if CHECK_NSLICES == True:
                img = sitk.ReadImage(str(image_path))
                img_array = sitk.GetArrayFromImage(img)
                assert len(img_array[0]) == n_slices_seq

            # setup paths for each slice
            for slice_idx in range(n_slices_seq):
                if gt_seg_array[slice_idx].sum() <= 0 and include_no_myo == False:
                    pass
                else:
                    # add path to ground truth segmentation
                    self.gt_seg_pathlist.append(str(gt_seg_path))
                    # add pseudo seg and path to image
                    self.image_pathlist.append((str(image_path), slice_idx))
                    # create mapping ID : idxs, in order to find all data for a patient in get_patient
                    self.id_to_idxs[subjectID].append(
                        len(self.image_pathlist)-1)

    def __len__(self):
        return len(self.image_pathlist)

    def __getitem__(self, idx):
        """Returns a dictionary with keys:
            - img                   ->  transformed image tensor
            - gt_myo                ->  Ground truth myocardium segmentation.
            - original_shape        ->  shape of image before resizing/cropping.
            - spacing               ->  spacing between voxels in original image.
            - origin                ->  Origin of the image. Not really necessary for us, but sitk requires
                                        an origin. We set the origin to (0, 0, 0, 0). Might be more relevant
                                        if you try to combine short axis images with long axis images.
            - img_path              ->  path where the image is saved, so you can check it with e.g. mitk.
            - slice_idx             ->  slice within the image at img_path.
        """
        # select slice from sitk image
        image_path, slice_idx = self.image_pathlist[idx]
        img = sitk.ReadImage(image_path)
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        img_array = sitk.GetArrayFromImage(img)
        img_array = img_array.astype(np.float32)[:, slice_idx]
        original_shape = img_array.shape

        gt_seg_path = self.gt_seg_pathlist[idx]
        gt_seg_img = sitk.ReadImage(gt_seg_path)
        gt_seg_array = sitk.GetArrayFromImage(gt_seg_img)
        gt_seg_array = gt_seg_array.astype('float32')
        gt_seg_array = gt_seg_array[slice_idx][None, ...]

        assert img_array.shape == gt_seg_array.shape, f"Shapes {img_array.shape} and {gt_seg_array.shape} don't match"

        img_tensor = torch.from_numpy(img_array / self.maxvalue)
        gt_seg_tensor = torch.from_numpy(gt_seg_array)

        pre_transform = [img_tensor, gt_seg_tensor]
        # apply other transformations (e.g. resampling, data augs)
        if self.transform:
            post_transform = self.transform(pre_transform)
            img_tensor, gt_seg_tensor = post_transform[0], post_transform[1]

        data = {'img_path': image_path, 'slice_idx': slice_idx,
                'gt_myo': gt_seg_tensor, 'img': img_tensor,
                'original_shape': original_shape,
                'origin': origin, 'spacing': spacing}
        return data

    def get_patient_batch(self, pid):
        """Returns batch with slices from a specific patient id."""
        assert pid in self.id_to_idxs
        idxs = self.id_to_idxs[pid]
        data_list = [self.__getitem__(idx) for idx in idxs]
        batch = tudl.default_collate(data_list)
        return batch


class DeepRiskDatasetSegmentation3D(Dataset):
    """Returns 3D stacks with (pseudo) segmentations and weak labels.
    Most flexible dataset class for 3D, since all types of extra
    segmentations are optional (ground truth fibrosis, ground truth
    myocardium, predictions etc.). Only the images and weak labels
    are stricly required. """

    def __init__(self, image_dir, labels_file, pseudo_fib_dir=None,
                 gt_fib_dir=None, pred_fib_dir=None,
                 gt_myo_dir=None, pred_myo_dir=None,
                 transform=None, split="train",
                 roi_crop="fixed", img_type='PSIR',
                 pixel_gt_only=False):
        """Input:
            - image_dir:        Directory where Deep Risk images are located.
            - labels_file:      Excel sheet with (1) Patient ID,
                                and for up to three sequences per patient
                                (2) which slices have no myocardium
                                (3) weak fibrosis labels (which slices have fibrosis)
                                (4) total number of slices per sequence.
            - pseudo_fib_dir:   (Optional) Directory containing pseudo ground truth
                                fibrosis segmentations.
            - pred_fib_dir:     (Optional) Directory containing predicted fibrosis segmentations.
            - gt_fib_dir:       (Optional) Directory containing ground truth fibrosis segmentations.
            - pred_myo_dir:     (Optional) Directory containing predicted myocardium segmentations.
                                Required for doing a region of interest crop.
            - gt_myoseg_dir:    (Optional) Directory containing ground truth myocardium segmentations.
            - transform:        (torchvision) transforms for data augmentation, resizing etc.
            - split:            Data split. Default="train". Options: "train", "val", "test".
            - pixel_gt_only:    Whether to only include images that have ground truth segmentations.
                                Default=False.     
            - roi_crop:         Which type of Region-of-Interest crop to use.
                                Options: 
                                    * "fixed" -> size 100 crop around predicted myocardium center
                                    * "fitted" -> Smallest square crop around predicted myocardium
            - img_type:         Type of image to use. For Deep Risk we have both MAG and PSIR, which
                                should contain the same information (with some different scaling/mapping).
                                We choose one of these to train and evaluate on. Default="PSIR"
        """
        self.transform = transform
        assert split in ["train", "val", "test", "all"]
        self.split = split
        if roi_crop not in ["fitted", "fixed", "none"]:
            raise NotImplementedError
        assert img_type in ['PSIR', 'MAG']
        self.roi_crop = roi_crop
        # dataset statistics (can be used for setting images between [0, 1] range)
        self.maxvalue = 4095.0
        self.minvalue = 0
        # load labels
        label_df = pd.read_excel(labels_file)
        # split train val and test data based on patients (rather than slices/mri's)
        TEST_IDS = DETERMINISTIC_TEST
        VAL_IDS = DETERMINISTIC_VAL
        PIXEL_LABEL_IDS = DETERMINISTIC_TRAIN + DETERMINISTIC_VAL + DETERMINISTIC_TEST

        if split == "test":
            label_df = label_df[label_df['SubjectID'].isin(TEST_IDS)]
            print(f"Test patients: {len(label_df)}")
            print(f"{TEST_IDS=}")
        elif split == "val":
            label_df = label_df[label_df['SubjectID'].isin(VAL_IDS)]
            print(f"Validation patients: {len(label_df)}")
            print(f"{VAL_IDS=}")
        elif split == "train":
            if pixel_gt_only:
                label_df = label_df[label_df['SubjectID'].isin(
                    PIXEL_LABEL_IDS)]
            label_df = label_df[~label_df['SubjectID'].isin(TEST_IDS)]
            label_df = label_df[~label_df['SubjectID'].isin(VAL_IDS)]
            print(f"Train patients: {len(label_df)}")
        elif split == "all":
            if pixel_gt_only:
                label_df = label_df[label_df['SubjectID'].isin(
                    PIXEL_LABEL_IDS)]
        # select only labeled patients
        label_df = label_df[label_df.Gelabeled == 'Ja']
        print(f'{len(label_df)=}')

        dirs = {'img': image_dir, 'gt_fib': gt_fib_dir, 'pseudo_fib': pseudo_fib_dir,
                'pred_fib': pred_fib_dir, 'gt_myo': gt_myo_dir, 'pred_myo': pred_myo_dir}
        # remove image types that have no directory
        dirs = {x: dirs[x] for x in dirs if dirs[x] != None}
        self.image_names = [x for x in dirs]
        self.pathlists = {k: [] for k in self.image_names}
        self.labels = []
        self.id_to_idxs = defaultdict(list)

        all_files = {k: [] for k in self.image_names}
        for image_name, dir in dirs.items():
            if image_name == 'img':
                all_files[image_name] = [
                    str(x) for x in dir.glob(f'*{img_type}*.mha')]
            else:
                all_files[image_name] = [str(x) for x in dir.glob(f'*.nrrd')]

        for _, row in tqdm(label_df.iterrows()):
            n_slices = row.n_slices.split('|')
            assert len(n_slices) == 3
            for i, n_slices_seq in enumerate(n_slices):
                # skip empty/non-existent sequences
                if n_slices_seq == '-':
                    continue
                else:
                    n_slices_seq = int(n_slices_seq)

                subjectID = row["SubjectID"]
                seq_pos_labels = row["ICD_LGE_slices_seq" + str(i)].split(';')
                import re

                r = re.compile(f".*{subjectID}.*seq{i}.*")
                for image_name, files in all_files.items():
                    path = list(filter(r.match, files))
                    if CHECK_NSLICES:
                        img = get_array_from_nifti(path)
                        # there is inconsistency between slice and channel dimension across image types:
                        # for  ground truth/pseudo_fib (C,D,H,W) and ground truths / myo_preds (D,C,H,W)
                        # this is dumb
                        # but since C=1, number of slices is first two dimension shapes multiplied
                        img_slices = img.shape[0] * img.shape[1]
                        if n_slices_seq != img_slices:
                            print(f"{path=}")
                            print(f"Error: Number of slices {n_slices_seq} and {img_slices} does not match")
                    if len(path) == 0:
                        if not 'gt' in image_name:
                            # we expect that not every image has a ground truth,
                            # but we should have predictions and pseudo ground truth for every image
                            print(f"Warning: Found no {image_name} file for sequence {subjectID} seq {i}")
                        self.pathlists[image_name].append(None)
                    else:
                        assert len(path) == 1, f"Found multiple files for a sequence \n {path}"
                        path = path[0]
                        self.pathlists[image_name].append(path)

                # if a sequence does not have a required ground truth,
                # retroactively get rid of all other image types for that sequence.
                has_gt = True
                if pixel_gt_only == True:
                    for image_name, paths in self.pathlists.items():
                        if 'gt' in image_name and paths[-1] == None:
                            has_gt = False
                            for image_name in self.pathlists:
                                del self.pathlists[image_name][-1]
                            continue

                if pixel_gt_only == False or has_gt == True:
                    # add slice-level classification labels
                    seq_labels = []
                    for i in range(n_slices_seq):
                        if str(i) in seq_pos_labels:
                            seq_labels.append(1)
                        else:
                            seq_labels.append(0)
                    self.labels.append(seq_labels)

                    # create mapping ID : idxs, in order to find all data for a patient in get_patient
                    # (which can be multiple sequences per patient)
                    self.id_to_idxs[subjectID].append(len(self.labels)-1)

    def add_transform(self, transform):
        """Function to add/change the (torchvision) transform after initialization."""
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def get_n_fibrotic_gt_slices(self):
        """Returns total number of fibrotic slices with a ground truth fibrosis segmentation label."""
        return sum([sum(l) for i, l in enumerate(self.labels) if self.pathlists['gt_fib'][i] != None])

    def __getitem__(self, idx):
        """Returns a dictionary with keys:
            - img                   ->  transformed image tensor
            - label                 ->  List of per-slice binary fibrosis labels.
            - (Optional) pseudo_fib ->  pseudo ground truth fibrosis segmentation.
            - (Optional) gt_fib     ->  Ground truth fibrosis segmentation (originally binary, but with interpolation).
                                        Values set to -1 if no fibrosis segmentation exists.
            - (Optional) gt_myo     ->  Ground truth myocardium segmentation.
                                        Values set to -1 if it doesn't exist.
            - (Optional) pred_myo   ->  Prediction myocardium segmentation (confidence between 0 and 1).
                                        Values set to -1 if no myocardium segmentation exists.
            - (Optional) pred_fib   ->  Prediction fibrosis segmentation (confidence between 0 and 1).
                                    	Values set to -1 if no fibrosis segmentation exists.
            - crop_corners          ->  corner location where the crop is made. nan if no crop is done.
            - original_shape        ->  shape of image before resizing/cropping.
            - spacing               ->  spacing between voxels in original image.
            - origin                ->  Origin of the image. Not really necessary for us, but sitk requires
                                        an origin. We set the origin to (0, 0, 0, 0). Might be more relevant
                                        if you try to combine short axis images with long axis images.
            - img_path              ->  path where the image is saved, so you can check it with e.g. mitk.
            - slice_idx             ->  slice within the image at img_path.
        """
        images, image_names = [], []
        img_array, spacing, origin = get_array_from_nifti(
            self.pathlists['img'][idx], with_spacing=True, with_origin=True)
        original_shape = img_array.shape
        images.append(img_array)
        image_names.append('img')

        for image_name in self.pathlists:
            if image_name != 'img':
                if self.pathlists[image_name][idx] != None:
                    array = get_array_from_nifti(
                        self.pathlists[image_name][idx])
                else:
                    array = -1.0 * np.ones_like(img_array)
                images.append(array)
                image_names.append(image_name)

        data = process_images_3d(images, image_names, self.transform, self.roi_crop)
        data['original_shape'] = original_shape
        data['spacing'] = spacing
        data['origin'] = origin
        data['label'] = self.labels[idx]
        data['img_path'] = self.pathlists['img'][idx]
        return data

    def get_patient_batch(self, pid):
        """Returns batch with 3D stacks for a specific patient id."""
        assert pid in self.id_to_idxs
        idxs = self.id_to_idxs[pid]
        data_list = [self.__getitem__(idx) for idx in idxs]
        keys = data_list[0].keys()

        batch = tudl.default_collate(data_list)
        return batch


if __name__ == '__main__':
    """# test out MyoPS

    DATA_DIR_NAME = 'MyoPS 2020 Dataset'
    TRAIN_DIR = os.path.join(DATA_DIR_NAME, 'train25//train25')
    GD_DIR = os.path.join(DATA_DIR_NAME, 'train25_myops_gd//train25_myops_gd')

    transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset2D = MyopsDataset2D(TRAIN_DIR, GD_DIR, transform=transforms)
    dataloader2D = DataLoader(dataset2D, batch_size=150, shuffle=True)

    # dataset statistics for normalization
    print(len(dataset2D))
    for i, (img, label) in enumerate(dataloader2D):
        print(img.mean())
        print(img.std())
        print(img.max())
        print(img.min())"""

    # test out deeprisk
    DATA_DIR = Path(r"\\amc.intra\users\R\rcklein\home\deeprisk\weakly_supervised\data")
    IMG_DIR = DATA_DIR.joinpath("all_niftis_n=657")
    LABELS_FILE = DATA_DIR.joinpath("weak_labels_n=657.xlsx")
    MYOSEG_DIR = DATA_DIR.joinpath(r"nnUnet_results\nnUNet\2d\Task500_MyocardSegmentation\predictions")
    SEG_LABELS_DIR = None
    # make images equal size
    data_transforms = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Normalize(mean=[.57], std=[.06])
    ])
    if MYOSEG_DIR == None:
        data_transforms = transforms.Compose([
            transforms.CenterCrop((224, 224)),
            data_transforms
        ])

    deeprisk_train = DeepRiskDataset2D(IMG_DIR, LABELS_FILE, seg_labels_dir=SEG_LABELS_DIR,
                                       transform=data_transforms, split="train", train_frac=0.75, split_seed=42, myoseg_dir=MYOSEG_DIR)
    deeprisk_val = DeepRiskDataset2D(IMG_DIR, LABELS_FILE, seg_labels_dir=SEG_LABELS_DIR,
                                     transform=data_transforms, split="val", train_frac=0.75, split_seed=42, myoseg_dir=MYOSEG_DIR)
    deeprisk_test = DeepRiskDataset2D(IMG_DIR, LABELS_FILE, transform=data_transforms,
                                      split="test", train_frac=0.75, split_seed=42, myoseg_dir=MYOSEG_DIR)
    print("train", len(deeprisk_train), "positive", sum(deeprisk_train.labels))
    print("val", len(deeprisk_val), "positive", sum(deeprisk_val.labels))
    print("test", len(deeprisk_test), "positive", sum(deeprisk_test.labels))

    train_dataloader = DataLoader(
        deeprisk_train, batch_size=128, shuffle=False)
    val_dataloader = DataLoader(deeprisk_train, batch_size=128, shuffle=False)
    test_dataloader = DataLoader(deeprisk_test, batch_size=128, shuffle=False)

    # dataset statistics, check normalization
    # for i, (path, img, label) in enumerate(train_dataloader):
    #    print("mean:", img.mean().item(), ", std:", img.std().item(), ", max:", img.max().item(), ", min:", img.min().item())
