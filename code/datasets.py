import numpy as np
import torch
import torch.nn.functional as F
import os
from matplotlib import pyplot as plt
from matplotlib import cm
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import pandas as pd
import random
from tqdm import tqdm
from collections import defaultdict
import torch.utils.data.dataloader as tudl

from preprocessing import myoseg_to_roi, normalize_image_func, normalize_image, set_image_range, roi_crop_multiple_images, get_array_from_nifti

CHECK_NSLICES = False
NEW_SPLIT = True
NEW_VAL =  ['DRAUMC0046',            'DRAUMC0051',
            'DRAUMC0063',            'DRAUMC0331',
            'DRAUMC0790',            'DRAUMC0805',
            'DRAUMC0809',            'DRAUMC0810',
            'DRAUMC0891',            'DRAUMC0949',
            'DRAUMC1049',            'DRAUMC1059']

NEW_TEST = ['DRAUMC0075',            'DRAUMC0184',
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

NEW_TRAIN =['DRAUMC0002',            'DRAUMC0008',            'DRAUMC0056',            'DRAUMC0072',
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
    def __init__(self, image_dir, ground_truth_dir, transform=None):
        # select only LGE images from dataset (for now)
        self.image_pathlist = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if "DE" in fname]
        self.gd_pathlist = [os.path.join(image_dir, fname) for fname in os.listdir(ground_truth_dir)]
        self.transform = transform

        # create classification labels, a per slice label of whether there is scars or edema present
        labels = []
        for gd_fname in os.listdir(ground_truth_dir):
            gd_path = os.path.join(ground_truth_dir, gd_fname)
            gd_img = sitk.ReadImage(gd_path)
            gd_array = sitk.GetArrayFromImage(gd_img)
            # edema and scar have ground truth values of 1220 and 2221
            # LV blood, RV blood, LV myo have 500, 600 and 200
            labels.append(list(np.amax((gd_array>=2221), (1, 2))))

        self.labels = labels
        print("positive labels:", sum([sum(label) for label in labels]))
        print("total labels:", sum([len(label) for label in labels]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_pathlist[idx]
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)
        img_tensor = torch.Tensor(img_array)
        label = torch.Tensor(self.labels[idx])
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label




class MyopsDataset2D(Dataset):
    """Returns 2D slices rather than the 3D images"""
    def __init__(self, image_dir, ground_truth_dir, transform=None):
        # select only LGE images from dataset (for now)
        self.image_pathlist = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if "DE" in fname]
        self.gd_pathlist = [os.path.join(image_dir, fname) for fname in os.listdir(ground_truth_dir)]
        self.transform = transform
        # dataset statistics (used to set images to [0, 1] range)
        self.maxvalue = 5060
        self.minvalue = 0


        # create classification labels, a per slice label of whether there are scars present
        self.slice_pathlist = []
        self.labels = []
        for gd_fname, image_path in zip(os.listdir(ground_truth_dir), self.image_pathlist):
            gd_path = os.path.join(ground_truth_dir, gd_fname)
            gd_img = sitk.ReadImage(gd_path)
            gd_array = sitk.GetArrayFromImage(gd_img)
            # edema and scar have ground truth values of 1220 and 2221
            # LV blood, RV blood, LV myo have 500, 600 and 200
            for slice_idx, gd_slice in enumerate(gd_array):
                self.labels.append(int(np.max((gd_slice>=2221))))
                self.slice_pathlist.append((image_path, slice_idx))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path, slice_idx = self.slice_pathlist[idx]

        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)
        img_tensor = torch.Tensor(img_array[slice_idx]  / self.maxvalue )[None, :]
        # paste grayscale to RGB
        # img_tensor = img_tensor.repeat(3, 1, 1)

        label = self.labels[idx]

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


class EmidecDataset3D(Dataset):
    def __init__(self, base_dir, image_mapname="Images", gt_mapname="Contours",
                    transform=None, pred_myo_dir=None, roi_crop="fixed", pred_fib_dir=None):
        self.roi_crop = roi_crop
        # select only LGE images from dataset / (sorting so images/gt/pred_myo is in same patient order)
        self.image_pathlist = sorted([str(x) for x in Path(base_dir).glob(f'**/{image_mapname}/*.nii.gz')])
        self.gt_pathlist = sorted([str(x) for x in Path(base_dir).glob(f'**/{gt_mapname}/*.nii.gz')])
        if self.gt_pathlist == []:
            self.split = "test"
        else:
            self.split = "train" # emidec training set is used, since no ground truth is available for test set


        if pred_myo_dir != None:
            self.pred_myo_pathlist = sorted([str(x) for x in Path(pred_myo_dir).glob(f'*.nrrd')])
            assert len(self.pred_myo_pathlist) == len(self.image_pathlist), f"{len(self.pred_myo_pathlist)=}, {len(self.image_pathlist)=}"
        else:
            self.pred_myo_pathlist = []

        if pred_fib_dir != None:
            self.pred_fib_pathlist = sorted([str(x) for x in Path(pred_fib_dir).glob(f'*.nrrd')])
            assert len(self.pred_fib_pathlist) == len(self.image_pathlist)
        else:
            self.pred_fib_pathlist = []

        self.transform = transform
        self.id_to_idxs = {str(Path(x.stem).stem) for x in Path(base_dir).glob(f'**/{image_mapname}/*.nii.gz')}



    def __len__(self):
        return len(self.image_pathlist)

    def __getitem__(self, idx):
        images, image_names = [], []
        img_array, spacing, origin = get_array_from_nifti(self.image_pathlist[idx], with_spacing=True, with_origin=True)
        original_shape = img_array.shape
        images.append(img_array)
        image_names.append('img')

        gt_array = get_array_from_nifti(self.gt_pathlist[idx])
        # legend ground truth: (background (0), cavity(1), normal myocardium (2), myocardial infarction (3) and no-reflow (4)), all_myocardium = union(2, 3, 4)
        gt_no_reflow = gt_array == 4
        gt_fib = gt_array == 3
        images.append(gt_fib)
        image_names.append('gt_fib')
        gt_myo = (gt_array == 2) + gt_fib + gt_no_reflow
        images.append(gt_myo)
        image_names.append('gt_myo')

        if self.pred_fib_pathlist != []:
            pred_fib = get_array_from_nifti(self.pred_fib_pathlist[idx])
            images.append(pred_fib)
            image_names.append('pred_fib')

        if self.pred_myo_pathlist != []:
            pred_myo = get_array_from_nifti(self.pred_myo_pathlist[idx])
            images.append(pred_myo)
            image_names.append('pred_myo')

        data = process_images_3d(images, image_names, self.transform, self.roi_crop)
        data['original_shape'] = original_shape
        data['spacing'] = spacing
        data['origin'] = origin
        data['img_path'] = self.image_pathlist[idx]

        return data
        """# load myo segmentation if specified
        if self.pred_myo_pathlist != []:
            pred_myo_array = get_array_from_nifti(self.pred_myo_pathlist[idx])
            # take roi cropping if specified
            if self.roi_crop == "fitted" or self.roi_crop == "fixed":
                if self.roi_crop == "fitted":
                    fixed_size = None
                elif self.roi_crop == "fixed":
                    fixed_size = 100
                img_array, gt_fib, gt_myo, pred_myo_array, crop_corners = roi_crop_multiple_images([img_array, gt_fib, gt_myo], pred_myo_array, fixed_size=fixed_size)

            cropped_myo_pixels = gt_myo.sum()
            assert cropped_myo_pixels == myo_pixels, f"Lost part of ground truth label during cropping {myo_pixels=}, {cropped_myo_pixels=}}"
        else:
            pred_myo_array = -1.0 * np.ones_like(img_array) #"unavailable"
            crop_corners = float('nan')


        pre_transform = [torch.from_numpy(x) for x in [img_array, gt_fib, pred_myo_array]]

        # apply other transformations (e.g. resizing, data augs)
        if self.transform:
            post_transform = self.transform([pre_transform])
            img_tensor, gt_fib, gt_myo_tensor, pred_myo_tensor = post_transform[0], post_transform[1], post_transform[2]

        data = {'img_path' : image_path,
                'fibrosis_seg_label' : fibrosis_seg_tensor, 'myo_seg' : myoseg_tensor,
                'img' : img_tensor, 'label' : padded_label,
                'crop_corners' : crop_corners, 'original_shape' : original_shape,
                'origin' : origin, 'spacing' : spacing}
        return data"""


def process_images_3d(images, image_names, transform, roi_crop, depth_to_batch=False):
    """ Import ordering of images :
            - LGE image must be first (will get normalization, colorjitter augmentation etc),
            - pred_myo last for the roi cropping,
            - Segmentations in between
    """
    assert image_names[0] == 'img'
    if "pred_myo" in image_names and roi_crop in ["fitted", "fixed"]:
        assert image_names[-1] == "pred_myo" # put pred_myo last for convenience
        pred_myo = images.pop(-1)
        if roi_crop == "fitted":
            fixed_size = None
        elif roi_crop == "fixed":
            fixed_size = 100
        crop_results = roi_crop_multiple_images(pred_myo, images, fixed_size=fixed_size)
        crop_corners = crop_results[-1]
        images = list(crop_results)[:-1] # cropped pred_myo is back in images at the end of the list
    else:
        crop_corners = float('nan')

    # put depth dimension in batch dimension for transforms
    #print(f"{images[0].shape=}")
    D, H, W = images[0].shape[-3:]
    images = [torch.from_numpy(x).float().view(D, 1, H, W) for x in images]

    # apply other transformations (e.g. resizing, data augs)
    if transform:
        images = transform(images)

    D, _, new_H, new_W = images[0].shape
    if depth_to_batch == False:
        # for 3D models, recover depth dimension from batch dimension and add channel dimension
        images = [x.view(1, D, new_H, new_W) for x in images]
    else:
        images = [x.view(D, 1, new_H, new_W) for x in images]

    data = {name : image for name, image in zip(image_names, images)}
    data['crop_corners'] = crop_corners
    return data








class DeepRiskDataset2D(Dataset):
    """Returns 2D slices with weak labels rather than the 3D images"""
    def __init__(self, image_dir, labels_file, seg_labels_dir=None, transform=None,
                 split="train", train_frac = 1.0, split_seed=42, myoseg_dir=None,
                 include_no_myo=False, roi_crop="fixed",
                 img_type='PSIR'):
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
                        "DRAUMC1042", "DRAUMC1166", "DRAUMC1199"] #"DRAUMC1199" onduidelijk
        if NEW_SPLIT == False:
            TEST_IDS = MYO_TEST_IDS
            TEST_IDS.extend(["DRAUMC0051", "DRAUMC0805", "DRAUMC0891", "DRAUMC1049", "DRAUMC0008", "DRAUMC0949"])
            TRAIN_VAL_PIXEL_LABEL_IDS = [id for id in PIXEL_LABEL_IDS if id not in TEST_IDS]
            VAL_IDS = random.sample(TRAIN_VAL_PIXEL_LABEL_IDS, int(len(TRAIN_VAL_PIXEL_LABEL_IDS) * (1 - train_frac)))
        else:
            TEST_IDS = NEW_TEST
            VAL_IDS = NEW_VAL




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

                image_path = [x for x in image_dir.glob(f"{subjectID}*seq{i}*{img_type}*.mha") if not "old" in str(x)]
                assert len(image_path) != 0, f"Found no image file for sequence \n {subjectID}:seq{i}"
                assert len(image_path) == 1, f"Found multiple files for a sequence \n {image_path}"
                image_path = image_path[0]
                sequence_identifier = str(Path(image_path).stem).split(img_type)[0]
                # add fibrosis segmentation if it exists
                fibrosis_seg_file = [fibrosis_seg for fibrosis_seg  in all_fibrosis_seg_files if sequence_identifier in fibrosis_seg.stem]

                if myoseg_dir != None:
                    myoseg_f = [f for f in myoseg_dir.glob(f"{sequence_identifier}*.nrrd")]
                    assert len(myoseg_f) != 0, f"Found no myocard segmentation file for sequence \n {sequence_identifier}"
                    assert len(myoseg_f) == 1, f"Found multiple myocard segmentations for a sequence \n {image_path}"
                    myoseg_f = myoseg_f[0]


                # label each slice
                if CHECK_NSLICES:
                    img = sitk.ReadImage(str(image_path))
                    img_array = sitk.GetArrayFromImage(img)
                    if n_slices_seq != len(img_array[0]):
                        print(f"{image_path=}")
                        print(f"Error: Number of slices {n_slices_seq} and {len(img_array[0])} does not match")

                for slice_idx in range(n_slices_seq):
                    if str(slice_idx) in seq_no_myo and include_no_myo == False:
                        pass
                    else:
                        # add path to fibrosis segmentation if available
                        if len(fibrosis_seg_file) > 0:
                            assert len(fibrosis_seg_file) == 1
                            self.fibrosis_seg_pathlist.append(str(fibrosis_seg_file[0]))
                        else:
                            self.fibrosis_seg_pathlist.append(None)
                        # add weak label and path to image
                        if str(slice_idx) in seq_labels:
                            self.labels.append(1)
                            self.slice_pathlist.append((str(image_path), slice_idx))
                        else:
                            self.labels.append(0)
                            self.slice_pathlist.append((str(image_path), slice_idx))

                        # find filenames of myocard segmentations -> for roi cropping
                        if myoseg_dir != None:
                            self.myoseg_pathlist.append(str(myoseg_f))


    def __len__(self):
        return len(self.labels)

    def get_n_fibrotic_gt_slices(self):
        n_fibrotic_slices = 0
        for idx, (_) in enumerate(self.slice_pathlist):
            if self.fibrosis_seg_pathlist[idx] != None and self.labels[idx] == 1:
                n_fibrotic_slices += 1
        return n_fibrotic_slices

    def get_n_gt_slices(self):
        n_slices = 0
        for f in self.fibrosis_seg_pathlist:
            if f != None:
                n_slices += 1
        return n_slices

    def __getitem__(self, idx):
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
            fibrosis_seg_array = -1 * np.ones_like(img_array) #"unavailable"

        # take roi cropping if specified
        if self.myoseg_pathlist != []:
            myoseg_f = self.myoseg_pathlist[idx]
            myoseg = sitk.ReadImage(str(myoseg_f))
            myoseg_arr_3d = sitk.GetArrayFromImage(myoseg)[0]
            myoseg_arr = myoseg_arr_3d.astype(np.float32)[slice_idx][None, ...]
            if self.roi_crop == "fitted":
                img_array, crop_corners = myoseg_to_roi(img_array, np.copy(myoseg_arr_3d), fixed_size=None)
                myoseg_arr, crop_corners2 = myoseg_to_roi(myoseg_arr, np.copy(myoseg_arr_3d), fixed_size=None)
                fibrosis_seg_array, crop_corners3 = myoseg_to_roi(fibrosis_seg_array, np.copy(myoseg_arr_3d), fixed_size=None)
            elif self.roi_crop == "fixed":
                img_array, crop_corners = myoseg_to_roi(img_array, np.copy(myoseg_arr_3d), fixed_size=100)
                myoseg_arr, crop_corners2 = myoseg_to_roi(myoseg_arr, np.copy(myoseg_arr_3d), fixed_size=100)
                fibrosis_seg_array, crop_corners3 = myoseg_to_roi(fibrosis_seg_array, np.copy(myoseg_arr_3d), fixed_size=100)
            assert crop_corners == crop_corners2 == crop_corners3, f"{crop_corners=}{crop_corners2=}{crop_corners3=}"
        else:
            myoseg_arr = -1.0 * np.ones_like(img_array) #"unavailable"
            crop_corners = float('nan')


        img_tensor = torch.from_numpy(img_array / self.maxvalue)
        fibrosis_seg_tensor = torch.from_numpy(fibrosis_seg_array)
        myoseg_tensor = torch.from_numpy(myoseg_arr)





        pre_transform = [img_tensor, fibrosis_seg_tensor, myoseg_tensor]

        label = self.labels[idx]

        # apply other transformations (e.g. resampling, data augs)
        #print(f'{img_tensor.shape=}')
        if self.transform:
            #img_tensor = self.transform(img_tensor)
            post_transform = self.transform(pre_transform)
            img_tensor, fibrosis_seg_tensor, myoseg_tensor = post_transform[0], post_transform[1], post_transform[2]

        data = {'img_path' : image_path, 'slice_idx' : slice_idx,
                'fibrosis_seg_label' : fibrosis_seg_tensor, 'myo_seg' : myoseg_tensor,
                'img' : img_tensor, 'label' : label,
                'crop_corners' : crop_corners, 'original_shape' : original_shape,
                'origin' : origin, 'spacing' : spacing}
        return data



class DeepRiskDataset3D(Dataset):
    """Returns stacks of 2D slices with per-slice weak labels"""
    def __init__(self, image_dir, labels_file,
                 seg_labels_dir=None, transform=None,
                 split="train", train_frac = 1.0, split_seed=42,
                 myoseg_dir=None, include_no_myo=False, roi_crop="fixed",
                 min_slices=7, max_slices=15, img_type='PSIR'):
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
                        "DRAUMC1042", "DRAUMC1166", "DRAUMC1199"] #"DRAUMC1199" onduidelijk

        if NEW_SPLIT == False:
            TEST_IDS = MYO_TEST_IDS
            TEST_IDS.extend(["DRAUMC0051", "DRAUMC0805", "DRAUMC0891", "DRAUMC1049", "DRAUMC0008", "DRAUMC0949"])
            TRAIN_VAL_PIXEL_LABEL_IDS = [id for id in PIXEL_LABEL_IDS if id not in TEST_IDS]
            VAL_IDS = random.sample(TRAIN_VAL_PIXEL_LABEL_IDS, int(len(TRAIN_VAL_PIXEL_LABEL_IDS) * (1 - train_frac)))
        else:
            TEST_IDS = NEW_TEST
            VAL_IDS = NEW_VAL

        if split == "test":
            label_df = label_df[label_df['SubjectID'].isin(TEST_IDS)]
            print(f"Test patients: {len(label_df)}")
            print(f"{TEST_IDS=}")
        elif split == "val":
            label_df = label_df[label_df['SubjectID'].isin(VAL_IDS)]
            print(f"Validation patients: {len(label_df)} {label_df['SubjectID']}")
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
        self.max_num_slices = 15
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
                # skip empty/non-existent/incomplete sequences
                if n_slices_seq == '-' or int(n_slices_seq) < min_slices or int(n_slices_seq) > max_slices:
                    continue
                else:
                    n_slices_seq = int(n_slices_seq)

                subjectID = row["SubjectID"]
                seq_pos_labels = row["ICD_LGE_slices_seq" + str(i)].split(';')
                seq_no_myo = row["Geen_myo_slices_seq" + str(i)].split(';')

                # add image path
                image_path = [x for x in image_dir.glob(f"{subjectID}*seq{i}*{img_type}*.mha") if not "old" in str(x)]
                assert len(image_path) != 0, "Found no image file for sequence"
                assert len(image_path) == 1, f"Found multiple files for a sequence \n {image_path}"
                image_path = image_path[0]
                sequence_identifier = str(Path(image_path).stem).split(img_type)[0]

                if CHECK_NSLICES:
                    img = sitk.ReadImage(str(image_path))
                    img_array = sitk.GetArrayFromImage(img)
                    if n_slices_seq != len(img_array[0]):
                        print(f"{image_path=}")
                        print(f"Error: Number of slices {n_slices_seq} and {len(img_array[0])} does not match")
                self.image_pathlist.append(str(image_path))
                if n_slices_seq > self.max_num_slices:
                    self.max_num_slices = n_slices_seq

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
                #print(f"{seq_labels=} {seq_no_myo=} {seq_pos_labels=}")
                #self.labels.append([1 if str(i) in seq_labels else 0 for i in range(n_slices_seq)])


                # add fibrosis segmentation path if it exists
                fibrosis_seg_file = [fibrosis_seg for fibrosis_seg  in all_fibrosis_seg_files if sequence_identifier in fibrosis_seg.stem]
                if len(fibrosis_seg_file) > 0:
                    assert len(fibrosis_seg_file) == 1
                    self.fibrosis_seg_pathlist.append(str(fibrosis_seg_file[0]))
                else:
                    self.fibrosis_seg_pathlist.append(None)

                # add myoseg path if specified
                #if myoseg_dir != None:
                #    myoseg_f = [f for f in myoseg_dir.glob(f"{subjectID}*seq{i}*PSIR*.nrrd")]
                #    assert len(myoseg_f) != 0, "Found no myocard segmentation file for sequence"
                #    assert len(myoseg_f) == 1, f"Found multiple myocard segmentations for a sequence \n {image_path}"
                #    myoseg_f = myoseg_f[0]
                #    self.myoseg_pathlist.append(str(myoseg_f))
                if myoseg_dir != None:
                    myoseg_f = [myoseg for myoseg in all_myoseg_files if sequence_identifier in myoseg.stem]
                    assert len(myoseg_f) != 0, "Found no myocard segmentation file for sequence"
                    assert len(myoseg_f) == 1, f"Found multiple myocard segmentations for a sequence \n {image_path}"
                    myoseg_f = myoseg_f[0]
                    self.myoseg_pathlist.append(str(myoseg_f))


    def __len__(self):
        return len(self.labels)

    def add_transform(self, transform):
        self.transform = transform


    def __getitem__(self, idx):
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
            fibrosis_seg_array = -1 * np.ones_like(img_array) #"unavailable"

        # load myo segmentation if specified
        if self.myoseg_pathlist != []:
            myoseg_f = self.myoseg_pathlist[idx]
            myoseg = sitk.ReadImage(str(myoseg_f))
            myoseg_array = sitk.GetArrayFromImage(myoseg).astype(np.float32)

            assert myoseg_array.shape[-2] == img_array.shape[-2] == fibrosis_seg_array.shape[-2], f"shapes {myoseg_array.shape}, {img_array.shape}, {fibrosis_seg_array.shape} don't match for sequence {image_path}"
            assert myoseg_array.shape[-1] == img_array.shape[-1] == fibrosis_seg_array.shape[-1], f"shapes {myoseg_array.shape}, {img_array.shape}, {fibrosis_seg_array.shape} don't match for sequence {image_path}"
            # take roi cropping if specified
            if self.roi_crop == "fitted":
                img_array, crop_corners = myoseg_to_roi(img_array, myoseg_array, fixed_size=None)
                fibrosis_seg_array, crop_corners2 = myoseg_to_roi(fibrosis_seg_array, myoseg_array, fixed_size=None)
                myoseg_array, crop_corners3 = myoseg_to_roi(myoseg_array, myoseg_array, fixed_size=None)
            elif self.roi_crop == "fixed":
                img_array, crop_corners = myoseg_to_roi(img_array, myoseg_array, fixed_size=100)
                fibrosis_seg_array, crop_corners2 = myoseg_to_roi(fibrosis_seg_array, myoseg_array, fixed_size=100)
                myoseg_array, crop_corners3 = myoseg_to_roi(myoseg_array, myoseg_array, fixed_size=100)
            assert crop_corners == crop_corners2 == crop_corners3, f"{crop_corners=}{crop_corners2=}{crop_corners3=}"
        else:
            myoseg_array = -1.0 * np.ones_like(img_array) #"unavailable"
            crop_corners = float('nan')


        img_tensor = torch.from_numpy(img_array / self.maxvalue)
        fibrosis_seg_tensor = torch.from_numpy(fibrosis_seg_array)
        myoseg_tensor = torch.from_numpy(myoseg_array)



        label = self.labels[idx]
        # remove slices without myocard if specified
        remove_no_myo = np.array([l == "TO_BE_REMOVED" for l in label])
        if any(remove_no_myo):
            img_tensor, fibrosis_seg_tensor, myoseg_tensor = img_tensor[..., ~remove_no_myo, :, :], fibrosis_seg_tensor[..., ~remove_no_myo, :, :], myoseg_tensor[..., ~remove_no_myo, :, :]
            label = [l for l in label if l != "TO_BE_REMOVED"]

        pre_transform = [img_tensor, fibrosis_seg_tensor, myoseg_tensor]
        # pad images for equal dataloader size within batch
        # padding with nan -> ignore slices in loss
        padding = self.max_num_slices - img_tensor.shape[1]
        img_tensor = F.pad(img_tensor, pad=(0, 0, 0, 0, 0, padding), mode='constant', value=0.0)
        fibrosis_seg_tensor = F.pad(fibrosis_seg_tensor, pad=(0, 0, 0, 0, 0, padding), mode='constant', value=0.0)
        myoseg_tensor = F.pad(myoseg_tensor, pad=(0, 0, 0, 0, 0, padding), mode='constant', value=0.0)
        padded_label = label + [float('nan')]*padding
        padded_label = torch.Tensor(padded_label)

        # apply other transformations (e.g. resizing, data augs)
        if self.transform:
            post_transform = self.transform([img_tensor, fibrosis_seg_tensor, myoseg_tensor])
            img_tensor, fibrosis_seg_tensor, myoseg_tensor = post_transform[0], post_transform[1], post_transform[2]

        data = {'img_path' : image_path,
                'fibrosis_seg_label' : fibrosis_seg_tensor, 'myo_seg' : myoseg_tensor,
                'img' : img_tensor, 'label' : padded_label,
                'crop_corners' : crop_corners, 'original_shape' : original_shape,
                'origin' : origin, 'spacing' : spacing}
        return data




class DeepRiskDatasetSegmentation2D(Dataset):
    """Returns 2D slices with (pseudo) segmentations """
    def __init__(self, image_dir, labels_file, pseudoseg_dir,
                 gt_seg_dir=None, transform=None, split="train",
                 train_frac = 1.0, split_seed=42, myoseg_dir=None,
                 gt_myoseg_dir=None,
                 include_no_myo=False, roi_crop="fixed",
                 pixel_gt_only=False, img_type='PSIR'):
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
                        "DRAUMC1042", "DRAUMC1166", "DRAUMC1199"] #"DRAUMC1199" onduidelijk

        if NEW_SPLIT == False:
            TEST_IDS = MYO_TEST_IDS
            TEST_IDS.extend(["DRAUMC0051", "DRAUMC0805", "DRAUMC0891", "DRAUMC1049", "DRAUMC0008", "DRAUMC0949"])
            TRAIN_VAL_PIXEL_LABEL_IDS = [id for id in PIXEL_LABEL_IDS if id not in TEST_IDS]
            VAL_IDS = random.sample(TRAIN_VAL_PIXEL_LABEL_IDS, int(len(TRAIN_VAL_PIXEL_LABEL_IDS) * (1 - train_frac)))
        else:
            TEST_IDS = NEW_TEST
            VAL_IDS = NEW_VAL
            PIXEL_LABEL_IDS = NEW_TRAIN + NEW_VAL + NEW_TEST

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
                label_df = label_df[label_df['SubjectID'].isin(PIXEL_LABEL_IDS)]
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

                image_path = [x for x in image_dir.glob(f"{subjectID}*seq{i}*{img_type}*.mha") if not "old" in str(x)]
                assert len(image_path) != 0, "Found no image file for sequence"
                assert len(image_path) == 1, f"Found multiple files for a sequence \n {image_path}"
                image_path = image_path[0]
                sequence_identifier = str(Path(image_path).stem).split(img_type)[0]

                # add fibrosis segmentation if it exists
                gt_seg_file = [f for f in all_gt_seg_files if sequence_identifier in f.stem]
                if subjectID == "DRAUMC1155":
                    print(f"{gt_seg_file=}")
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
                    myoseg_f = [f for f in myoseg_dir.glob(f"{sequence_identifier}*.nrrd")]
                    assert len(myoseg_f) != 0, f"Found no myocard segmentation file for sequence"
                    assert len(myoseg_f) == 1, f"Found multiple myocard segmentations for a sequence \n {image_path}"
                    myoseg_f = myoseg_f[0]

                if gt_myoseg_dir != None:
                    gt_myoseg_file = [f for f in gt_myoseg_dir.glob(f"{sequence_identifier}*.nrrd")]
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
        pseudoseg_array = pseudoseg_array.astype(np.float32)[:, slice_idx]#pseudoseg_array.astype('float32')
        #pseudoseg_array = #pseudoseg_array[slice_idx][None, ...]

        # select slice from fibrosis labels if available
        if self.gt_seg_pathlist[idx] != None:
            gt_seg_file = self.gt_seg_pathlist[idx]
            gt_seg_img = sitk.ReadImage(gt_seg_file)
            gt_seg_array = sitk.GetArrayFromImage(gt_seg_img)
            gt_seg_array = gt_seg_array.astype('float32')
            gt_seg_array = gt_seg_array[slice_idx][None, ...]
        else:
            gt_seg_array = -1 * np.ones_like(img_array) #"unavailable"
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
                img_array, crop_corners = myoseg_to_roi(img_array, np.copy(myoseg_arr_3d), fixed_size=None)
                myoseg_arr, crop_corners2 = myoseg_to_roi(myoseg_arr, np.copy(myoseg_arr_3d), fixed_size=None)
                pseudoseg_array, crop_corners3 = myoseg_to_roi(pseudoseg_array, np.copy(myoseg_arr_3d), fixed_size=None)
                gt_seg_array, crop_corners4 = myoseg_to_roi(gt_seg_array, np.copy(myoseg_arr_3d), fixed_size=None)
                gt_myoseg_array, crop_corners5 = myoseg_to_roi(gt_myoseg_array, np.copy(myoseg_arr_3d), fixed_size=None)
            elif self.roi_crop == "fixed":
                img_array, crop_corners = myoseg_to_roi(img_array, np.copy(myoseg_arr_3d), fixed_size=100)
                myoseg_arr, crop_corners2 = myoseg_to_roi(myoseg_arr, np.copy(myoseg_arr_3d), fixed_size=100)
                pseudoseg_array, crop_corners3 = myoseg_to_roi(pseudoseg_array, np.copy(myoseg_arr_3d), fixed_size=100)
                gt_seg_array, crop_corners4 = myoseg_to_roi(gt_seg_array, np.copy(myoseg_arr_3d), fixed_size=100)
                gt_myoseg_array, crop_corners5 = myoseg_to_roi(gt_myoseg_array, np.copy(myoseg_arr_3d), fixed_size=100)
            assert crop_corners == crop_corners2 == crop_corners3 == crop_corners4 == crop_corners5, f"{crop_corners=}{crop_corners2=}{crop_corners3=}{crop_corners4=}{crop_corners5=}"
        else:
            myoseg_arr = -1.0 * np.ones_like(img_array) #"unavailable"
            crop_corners = float('nan')


        img_tensor = torch.from_numpy(img_array / self.maxvalue)
        pseudoseg_tensor = torch.from_numpy(pseudoseg_array)
        gt_seg_tensor = torch.from_numpy(gt_seg_array)
        myoseg_tensor = torch.from_numpy(myoseg_arr)
        gt_myoseg_tensor = torch.from_numpy(gt_myoseg_array)


        pre_transform = [img_tensor, pseudoseg_tensor, gt_seg_tensor, myoseg_tensor, gt_myoseg_tensor]
        # apply other transformations (e.g. resampling, data augs)
        if self.transform:
            post_transform = self.transform(pre_transform)
            img_tensor, pseudoseg_tensor, gt_seg_tensor, myoseg_tensor, gt_myoseg_tensor = post_transform[0], post_transform[1], post_transform[2], post_transform[3], post_transform[4]

        data = {'img_path' : image_path, 'slice_idx' : slice_idx,
                'pseudo_fib' : pseudoseg_tensor, 'gt_fib' : gt_seg_tensor, # pseudoseg, gt_seg
                'pred_myo' : myoseg_tensor, 'gt_myo' : gt_myoseg_tensor, #myo_seg, gt_myo_seg
                 'img' : img_tensor,
                'crop_corners' : crop_corners, 'original_shape' : original_shape,
                'origin' : origin, 'spacing' : spacing}
        return data

    def get_patient_batch(self, pid):
        assert pid in self.id_to_idxs
        idxs = self.id_to_idxs[pid]
        data_list = [self.__getitem__(idx) for idx in idxs]
        keys = data_list[0].keys()

        batch = tudl.default_collate(data_list)
        return batch


class DeepRiskDatasetMyoSegmentation2D(Dataset):
    def __init__(self, image_dir, gt_seg_dir,
                 transform=None, split="train",
                 train_frac = 1.0, split_seed=42,
                 include_no_myo=False, img_type='PSIR', gt_type='PSIR'):
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
                        "DRAUMC1042", "DRAUMC1166", "DRAUMC1199"] #"DRAUMC1199" onduidelijk

        if NEW_SPLIT == False:
            TEST_IDS = MYO_TEST_IDS
            TEST_IDS.extend(["DRAUMC0051", "DRAUMC0805", "DRAUMC0891", "DRAUMC1049", "DRAUMC0008", "DRAUMC0949"])
            TRAIN_VAL_PIXEL_LABEL_IDS = [id for id in PIXEL_LABEL_IDS if id not in TEST_IDS]
            VAL_IDS = random.sample(TRAIN_VAL_PIXEL_LABEL_IDS, int(len(TRAIN_VAL_PIXEL_LABEL_IDS) * (1 - train_frac)))
            TRAIN_IDS = [id for id in TRAIN_VAL_PIXEL_LABEL_IDS if id not in VAL_IDS]
        else:
            TEST_IDS = NEW_TEST
            VAL_IDS = NEW_VAL
            TRAIN_IDS = NEW_TRAIN


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
            assert len(gt_seg_path) != 0, f"Found no segmentation file for patient {subjectID}"
            assert len(gt_seg_path) == 1, f'Found multiple files for patient\n {gt_seg_path}'
            gt_seg_path = gt_seg_path[0]
            sequence_identifier = str(Path(gt_seg_path).stem).split(gt_type)[0]

            image_path = [x for x in image_dir.glob(f"{sequence_identifier}*{img_type}*.mha") if not "old" in str(x)]
            assert len(image_path) != 0, f"Found no image file for sequence \n {sequence_identifier}"
            assert len(image_path) == 1, f"Found multiple files for a sequence \n {image_path}"
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
                    self.id_to_idxs[subjectID].append(len(self.image_pathlist)-1)


    def __len__(self):
        return len(self.image_pathlist)

    def __getitem__(self, idx):
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

        data = {'img_path' : image_path, 'slice_idx' : slice_idx,
                'gt_myo' : gt_seg_tensor, 'img' : img_tensor, # gt_seg, img
                'original_shape' : original_shape,
                'origin' : origin, 'spacing' : spacing}
                #{'img' : image_dir, 'gt_fib' : gt_fib_dir, 'pseudo_fib' : pseudo_fib_dir,
                #        'pred_fib' : pred_fib_dir, 'gt_myo' : gt_myo_dir, 'pred_myo' : pred_myo_dir}
        return data

    def get_patient_batch(self, pid):
        assert pid in self.id_to_idxs
        idxs = self.id_to_idxs[pid]
        data_list = [self.__getitem__(idx) for idx in idxs]
        batch = tudl.default_collate(data_list)
        return batch




class DeepRiskDatasetSegmentation3D(Dataset):
    """Returns 3D stacks with (pseudo) segmentations """
    def __init__(self, image_dir, labels_file, pseudo_fib_dir=None,
                 gt_fib_dir=None, pred_fib_dir=None,
                 gt_myo_dir=None, pred_myo_dir=None,
                 transform=None, split="train",
                 roi_crop="fixed", img_type='PSIR',
                 pixel_gt_only=False):
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
        TEST_IDS = NEW_TEST
        VAL_IDS = NEW_VAL
        PIXEL_LABEL_IDS = NEW_TRAIN + NEW_VAL + NEW_TEST

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
                label_df = label_df[label_df['SubjectID'].isin(PIXEL_LABEL_IDS)]
            label_df = label_df[~label_df['SubjectID'].isin(TEST_IDS)]
            label_df = label_df[~label_df['SubjectID'].isin(VAL_IDS)]
            print(f"Train patients: {len(label_df)}")
        elif split == "all":
            if pixel_gt_only:
                label_df = label_df[label_df['SubjectID'].isin(PIXEL_LABEL_IDS)]
        # select only labeled patients
        label_df = label_df[label_df.Gelabeled == 'Ja']
        print(f'{len(label_df)=}')

        dirs = {'img' : image_dir, 'gt_fib' : gt_fib_dir, 'pseudo_fib' : pseudo_fib_dir,
                'pred_fib' : pred_fib_dir, 'gt_myo' : gt_myo_dir, 'pred_myo' : pred_myo_dir}
        # remove image types that have no directory
        dirs = {x : dirs[x] for x in dirs if dirs[x] != None}
        self.image_names = [x for x in dirs]
        self.pathlists = {k: [] for k in self.image_names}
        self.labels = []
        self.id_to_idxs = defaultdict(list)

        all_files = {k: [] for k in self.image_names}
        for image_name, dir in dirs.items():
            if image_name == 'img':
                all_files[image_name] = [str(x) for x in dir.glob(f'*{img_type}*.mha')]
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
                        img_slices = img.shape[0] * img.shape[1] # inconsistency between slice and channel dimension for  ground truth/pseudo_fib (C,D,H,W) and ground truths / myo_preds (D,C,H,W)
                        if n_slices_seq != img_slices:
                            print(f"{path=}")
                            print(f"Error: Number of slices {n_slices_seq} and {img_slices} does not match")
                    if len(path) == 0:
                        if not 'gt' in image_name:
                            print(f"Found no {image_name} file for sequence {subjectID} seq {i}")
                        self.pathlists[image_name].append(None)
                    else:
                        assert len(path) == 1, f"Found multiple files for a sequence \n {image_path}"
                        path = path[0]
                        self.pathlists[image_name].append(path)

                # retroactively get rid of secundary sequences if they don't have ground truth
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
                    self.id_to_idxs[subjectID].append(len(self.labels)-1)


    def add_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def get_n_fibrotic_gt_slices(self):
        return sum([sum(l) for i, l in enumerate(self.labels) if self.pathlists['gt_fib'][i] != None])

    def __getitem__(self, idx):
        images, image_names = [], []
        img_array, spacing, origin = get_array_from_nifti(self.pathlists['img'][idx], with_spacing=True, with_origin=True)
        original_shape = img_array.shape
        images.append(img_array)
        image_names.append('img')

        for image_name in self.pathlists:
            if image_name != 'img':
                if self.pathlists[image_name][idx] != None:
                    array = get_array_from_nifti(self.pathlists[image_name][idx])
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
        #transforms.CenterCrop((256, 256)),
        transforms.Normalize(mean=[.57], std=[.06])
    ])
    if MYOSEG_DIR == None:
        data_transforms = transforms.Compose([
            transforms.CenterCrop((224, 224)),
            data_transforms
        ])

    deeprisk_train = DeepRiskDataset2D(IMG_DIR, LABELS_FILE, seg_labels_dir=SEG_LABELS_DIR, transform=data_transforms, split="train", train_frac = 0.75, split_seed=42, myoseg_dir=MYOSEG_DIR)
    deeprisk_val = DeepRiskDataset2D(IMG_DIR, LABELS_FILE, seg_labels_dir=SEG_LABELS_DIR, transform=data_transforms, split="val", train_frac = 0.75, split_seed=42, myoseg_dir=MYOSEG_DIR)
    deeprisk_test = DeepRiskDataset2D(IMG_DIR, LABELS_FILE, transform=data_transforms, split="test", train_frac = 0.75, split_seed=42, myoseg_dir=MYOSEG_DIR)
    print("train", len(deeprisk_train), "positive", sum(deeprisk_train.labels))
    print("val", len(deeprisk_val), "positive", sum(deeprisk_val.labels))
    print("test", len(deeprisk_test), "positive", sum(deeprisk_test.labels))


    train_dataloader = DataLoader(deeprisk_train, batch_size=128, shuffle=False)
    val_dataloader = DataLoader(deeprisk_train, batch_size=128, shuffle=False)
    test_dataloader = DataLoader(deeprisk_test, batch_size=128, shuffle=False)

    # dataset statistics, check normalization
    #for i, (path, img, label) in enumerate(train_dataloader):
    #    print("mean:", img.mean().item(), ", std:", img.std().item(), ", max:", img.max().item(), ", min:", img.min().item())
