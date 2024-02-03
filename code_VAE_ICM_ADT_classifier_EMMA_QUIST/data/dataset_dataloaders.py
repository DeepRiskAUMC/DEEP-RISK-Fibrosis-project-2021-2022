import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import SimpleITK as sitk
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from monai.transforms import (
        EnsureChannelFirst, Compose,  RandRotate90, 
        Resize,ToTensor, RandFlip
        )

from utils.preprocessing import *

class AUMCDatsetLGE_MLP(Dataset):
        """ Dataset class for 3D LGE + MLP"""
        def __init__(self, base_dir, IDS, image_name=None, label_name = None, predmyo=True, 
                 transforms=None, roi_crop="fitted", masked=False, img_ch=1, train=False, win_size=(64,64,64)):
                """Input:
                -   base_dir:       Base directory to patient mri id folders with lge, cine, myo pred and clinical info.
                -   image_name:     Filename starts with either LGE or CINE followed by an number id.
                -   predmyo_name:   Filename starts with MYO followed by an number id if not exist None
                                        Required for doing an region of interest crop.
                -   transform:      (torchvision) transforms for data augmentation, resizing etc.
                -   split:          Data split. Default="train". Options: "train", "val", "test".
                -   train_frac:     Proportion of the data to use as training data.
                                Is only used when global variable DETERMINISTIC_SPLIT == False.
                -   split_seed:     Seed to use when creating a random split.
                                Is only used when global variable DETERMINISTIC_SPLIT == False.
                -   roi_crop:       Which type of Region-of-Interest crop to use.
                                Options: 
                                        * "fixed" -> size 100 crop around predicted myocardium center
                                        * "fitted" -> Smallest square crop around predicted myocardium
                                        * anything else -> No cropping
                """

                self.transforms = transforms
                self.win_size = win_size
                
                if roi_crop not in ["fitted", "fixed"]:
                        raise NotImplementedError
                
                self.mlp = False
                self.img_name = image_name
                self.roi_crop = roi_crop
                self.masked = masked
                self.img_ch = img_ch
                self.train = train

                self.image_pathlist = sorted([str(x) for x in base_dir.glob(f"**/{image_name}*.nii.gz")])
                self.image_pathlist = filter(lambda path: any(id in path for id in IDS), self.image_pathlist)
                self.image_pathlist = list(self.image_pathlist) 


                if not label_name == None:
                        self.mlp = True
                        self.label_name = label_name
                        self.clin_pathlist = sorted([str(x) for x in Path(base_dir).glob(f'**/*.csv')])
                        self.clin_pathlist = filter(lambda path: any(id in path for id in IDS), self.clin_pathlist)
                        self.clin_pathlist = list(self.clin_pathlist)

                if masked:
                        self.mask_path_list = sorted([str(x) for x in base_dir.glob(f'**/MASK*.nii.gz')])
                        self.mask_path_list = filter(lambda path: any(id in path for id in IDS), self.mask_path_list)
                        self.mask_path_list = list(self.mask_path_list) 
                else:
                        self.mask_path_list = []

                if predmyo:
                        self.pred_myo_path_list = sorted([str(x) for x in base_dir.glob(f'**/MYO*.nrrd')])
                        self.pred_myo_path_list = filter(lambda path: any(id in path for id in IDS), self.pred_myo_path_list)
                        self.pred_myo_path_list = list(self.pred_myo_path_list) 
                else:
                        self.myo_path_list = []

                self.id_to_idxs = {(str(Path(x.stem).stem)).split('_')[1] for x in base_dir.glob(f'**/{image_name}*.nii.gz')}
                self.id_to_idxs = filter(lambda idx: any(int(id.split('_')[0]) == int(idx) for id in IDS), self.id_to_idxs)
                self.id_to_idxs = set(self.id_to_idxs) 

        def __len__(self):
                return len(self.image_pathlist)
        
        def compute_class_weights(self):

                labels = []
                for idx in range(len(self)):
                        _, label = self[idx]
                        labels.append(label)

                unique_labels = np.unique(labels)

                class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
                class_weights = torch.tensor(class_weights, dtype=torch.float32)

                return class_weights
        
        def IMG_transform(self, img):
                if self.train:
                        augment_transforms = Compose(
                                [
                                        EnsureChannelFirst(channel_dim=0),
                                        RandFlip(spatial_axis=0),
                                        RandFlip(spatial_axis=1),
                                        RandRotate90(),
                                        Resize(self.win_size, mode = "area"),
                                        ToTensor(),
                                ]
                                )
                        img = augment_transforms(img)
                else:
                        augment_transforms = Compose(
                                [
                                        EnsureChannelFirst(channel_dim=0),
                                        Resize(self.win_size, mode = "area"),
                                        ToTensor(),
                                ]
                                )
                        img = augment_transforms(img)
                return img
        
        def __getitem__(self, index):
                
                # load image stack
                image_path = self.image_pathlist[index]
                img = sitk.ReadImage(image_path)

                label = []
                if self.mlp:
                        clin_path = self.clin_pathlist[index]
                        clin_df = pd.read_csv(clin_path, header=None, index_col=0)
                        label = int(clin_df.loc[self.label_name][1])
                        
                        # primary prev. == 1 sec. prev == 2
                        if label == 2:
                                label = 0

                myoseg_f = self.pred_myo_path_list[index]
                myoseg = sitk.ReadImage(str(myoseg_f))
                mask_array = None
                if self.masked:
                        mask_f = self.mask_path_list[index]
                        mask = sitk.ReadImage(str(mask_f))
                        if self.img_ch > 1:
                                binary_myo = sitk.BinaryThreshold(myoseg, lowerThreshold=0.5, insideValue=1, outsideValue=0)
                                img_array = sitk.GetArrayFromImage(img).astype(np.float32)
                                mask_array = sitk.GetArrayFromImage(binary_myo).astype(np.float32)
                        else:
                                try:
                                        masked_image = sitk.Mask(img, mask)
                                except:
                                        masked_image = img * mask
                                img = masked_image
                                mask_array = sitk.GetArrayFromImage(mask).astype(np.float32)
                        

                if self.img_ch == 1:
                        img_array = sitk.GetArrayFromImage(img).astype(np.float32)

                myoseg_array = sitk.GetArrayFromImage(myoseg).astype(np.float32)

                
                assert myoseg_array.shape[-2] == img_array.shape[-2] , f"shapes {myoseg_array.shape}, {img_array.shape}  don't match for sequence {image_path}"
                assert myoseg_array.shape[-1] == img_array.shape[-1] , f"shapes {myoseg_array.shape}, {img_array.shape}, don't match for sequence {image_path}"

                # take roi cropping if specified
                if self.roi_crop == "fitted":
                        img_array, myoseg_array, mask_array, _ = myoseg_to_roi(img_array, myoseg_array, mask_array, fixed_size=None, masked=self.masked)
                        
                elif self.roi_crop == "fixed":
                        img_array, myoseg_array, mask_array, _ = myoseg_to_roi(img_array, myoseg_array, mask_array, fixed_size=100, masked=self.masked)

                if self.transforms:
                        img_array = self.transforms(img_array)
                       
                if self.img_ch > 1:
                        img_array = np.stack((img_array, mask_array), axis=0) 
                        img_array = self.IMG_transform(img_array)
                # plt.figure()        
                # plt.imshow(img_array[0,30, ...], cmap='gray')
                # plt.axis('off')
                # plt.savefig('TEST.png')
                # quit()

                return img_array, label
        

class EmidecDataset(torch.utils.data.Dataset):
        def __init__(self, image_files,  labels, transforms):
                self.image_files = image_files
                self.labels = labels
                self.transforms = transforms

        def __len__(self):
                return len(self.image_files)

        def compute_class_weights(self):

                labels = []
                for idx in range(len(self)):
                        _, label = self[idx]
                        labels.append(label)

                unique_labels = np.unique(labels)

                class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
                class_weights = torch.tensor(class_weights, dtype=torch.float32)

                return class_weights

        def __getitem__(self, index):
                return self.transforms(self.image_files[index]), self.labels[index]