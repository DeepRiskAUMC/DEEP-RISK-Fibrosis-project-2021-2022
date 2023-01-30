from configparser import Interpolation
from torch.utils.data.dataloader import DataLoader
from data_loading.import_AUMC_dataset import read_in_AUMC_data, read_in_fibrosis_AUMC_data, read_in_AUMC_classification_data
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import random
import cv2
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt

class AUMCDataset3D(Dataset):
    def __init__(self, LGE_images, myo_masks, aankleuring_masks, pat_ids, transform=None, max_value=4095) -> None:
        super().__init__()
        self.LGE_images = LGE_images
        self.myo_masks = myo_masks
        self.aankleuring_masks = aankleuring_masks
        self.pat_ids = pat_ids
        self.transform = transform
        self.max_value = max_value
        print(f"Maximum value of dataset: {self.max_value}. Transforms: {transform}")
    
    def __len__(self):
        return len(self.LGE_images)

    def __getitem__(self, index):
        LGE_image = torch.from_numpy(self.LGE_images[index])
        myo_mask = torch.from_numpy(self.myo_masks[index])
        aankleuring_mask = torch.from_numpy(self.aankleuring_masks[index])
        if LGE_image.dim() < 4:
            LGE_image = LGE_image.unsqueeze(dim=0)
        if myo_mask.dim() < 4:
            myo_mask = myo_mask.unsqueeze(dim=0)
        if aankleuring_mask.dim() < 4:
            aankleuring_mask = aankleuring_mask.unsqueeze(dim=0)
        pat_id = self.pat_ids[index]

        if self.transform is not None:
            LGE_image, myo_mask, aankleuring_mask = perform_transformations(self.transform, LGE_image, myo_mask, aankleuring_mask, self.max_value, dims='3D')

        return LGE_image, myo_mask, aankleuring_mask, pat_id
    
class AUMCDataset2D(Dataset):
    def __init__(self, LGE_images, myo_masks, aankleuring_masks, pat_ids, slice_indices, transform=None, max_value=4095) -> None:
        super().__init__()
        self.LGE_images = LGE_images
        self.myo_masks = myo_masks
        self.aankleuring_masks = aankleuring_masks
        self.pat_ids = pat_ids
        self.slice_indices = slice_indices
        # self.slice_indices = get_slice_indices(pat_ids)
        self.transform = transform
        self.max_value = max_value
        print(f"Maximum value of dataset: {self.max_value}. Transforms: {transform}")
    
    def __len__(self):
        return len(self.LGE_images)

    def __getitem__(self, index):
        LGE_image = torch.from_numpy(self.LGE_images[index])
        myo_mask = torch.from_numpy(self.myo_masks[index])
        aankleuring_mask = torch.from_numpy(self.aankleuring_masks[index])
        if LGE_image.dim() < 3:
            LGE_image = LGE_image.unsqueeze(dim=0)
        if myo_mask.dim() < 3:
            myo_mask = myo_mask.unsqueeze(dim=0)
        if aankleuring_mask.dim() < 3:
            aankleuring_mask = aankleuring_mask.unsqueeze(dim=0)
        pat_id = self.pat_ids[index]
        slice_nr = self.slice_indices[index]
        
        # save_image(myo_mask, f"test_myocard_mask_{pat_id}_slice{slice_nr}.png")
        # save_image(aankleuring_mask, f"test_fibrosis_mask_{pat_id}_slice{slice_nr}.png")

        if self.transform is not None:
            LGE_image, myo_mask, aankleuring_mask = perform_transformations(self.transform, LGE_image, myo_mask, aankleuring_mask, self.max_value, dims='2D')
        return LGE_image, myo_mask, aankleuring_mask, pat_id, slice_nr

def load_data(dataset, batch_size=8, num_workers=1, only_test=False, transformations=[], fibrosis_model=None, myocard_model_version=None, use_only_fib='no', resize='resize', size=(256, 256), normalize=['clip']):
    if dataset in ['AUMC2D', 'AUMC3D', 'AUMC2D_30', 'AUMC3D_30']:    
        train_data, val_data, test_data = get_data(dataset, only_test=only_test, transforms=transformations, 
                                            fibrosis_model=fibrosis_model, myocard_model_version=myocard_model_version, 
                                            use_only_fib=use_only_fib, resize=resize, size=size, normalize=normalize)

        if 'scale_after_gamma' in normalize or 'scale_after_gamma' in transformations:
            test_transform = ['scale_after_gamma']
        else:
            test_transform = []
        
        test_LGE_imgs, test_myo_masks, test_aankleuring_masks, test_pat_ids = test_data[:4]
        if only_test:
            max_value = np.max(test_LGE_imgs)
            if '2D' in dataset:
                test_slice_indices = test_data[4]
                test_dataset = AUMCDataset2D(test_LGE_imgs, test_myo_masks, test_aankleuring_masks, test_pat_ids, test_slice_indices, transform=test_transform, max_value=max_value)
            else:
                test_dataset = AUMCDataset3D(test_LGE_imgs, test_myo_masks, test_aankleuring_masks, test_pat_ids, transform=test_transform, max_value=max_value)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            return test_loader
        else:
            train_LGE_imgs, train_myo_masks, train_aankleuring_masks, train_pat_ids = train_data[:4]
            val_LGE_imgs, val_myo_masks, val_aankleuring_masks, val_pat_ids = val_data[:4]
            max_value = max([np.max(train_LGE_imgs), np.max(val_LGE_imgs), np.max(test_LGE_imgs)])
            if '2D' in dataset:
                train_slice_indices, val_slice_indices, test_slice_indices = train_data[4], val_data[4], test_data[4]
                train_dataset = AUMCDataset2D(train_LGE_imgs, train_myo_masks, train_aankleuring_masks, train_pat_ids, train_slice_indices, transform=transformations, max_value=max_value)
                val_dataset = AUMCDataset2D(val_LGE_imgs, val_myo_masks, val_aankleuring_masks, val_pat_ids, val_slice_indices, transform=test_transform, max_value=max_value)
                test_dataset = AUMCDataset2D(test_LGE_imgs, test_myo_masks, test_aankleuring_masks, test_pat_ids, test_slice_indices, transform=test_transform, max_value=max_value)
            else:
                train_dataset = AUMCDataset3D(train_LGE_imgs, train_myo_masks, train_aankleuring_masks, train_pat_ids, transform=transformations, max_value=max_value)
                val_dataset = AUMCDataset3D(val_LGE_imgs, val_myo_masks, val_aankleuring_masks, val_pat_ids, transform=test_transform, max_value=max_value)
                test_dataset = AUMCDataset3D(test_LGE_imgs, test_myo_masks, test_aankleuring_masks, test_pat_ids, transform=test_transform, max_value=max_value)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            loss_weights = get_loss_weights(train_dataset)
            return train_loader, val_loader, test_loader, loss_weights

def load_data_cross_validation(folds, dataset, batch_size=8, num_workers=1, only_test=False, transformations=[], fibrosis_model=None, myocard_model_version=None, use_only_fib='no', resize='resize', size=(256, 256), normalize=['clip'], seed=42):
    train_data, val_data, _ = get_data(dataset, only_test, transformations, fibrosis_model, myocard_model_version, use_only_fib, resize, size, normalize)

    # merge train and test data
    LGE_imgs = train_data[0] + val_data[0]
    myo_masks = train_data[1] + val_data[1]
    aankleuring_masks = train_data[2] + val_data[2]
    pat_ids = train_data[3] + val_data[3]

    unique_pat_ids = list(set(pat_ids))
    random.seed(seed)
    s = list(range(len(unique_pat_ids)))
    random.shuffle(unique_pat_ids)
    s = [unique_pat_ids[x::folds] for x in range(folds)]
    split_indices = []
    for pat_id_list in s:
        single_split_indices = [i for (i,patid) in enumerate(pat_ids) if patid in pat_id_list]
        split_indices.append(single_split_indices) 
    
    # turn into datasets and dataloaders
    if 'scale_after_gamma' in normalize or 'scale_after_gamma' in transformations:
        test_transform = ['scale_after_gamma']
    else:
        test_transform = []
    max_value = np.max(LGE_imgs)
    dataloader_splits = []
    for val_indices in split_indices:
        train_indices = [x for x in range(len(LGE_imgs)) if x not in val_indices]
        train_LGE_imgs, train_myo_masks, train_aankleuring_masks, train_pat_ids = [LGE_imgs[i] for i in train_indices], [myo_masks[i] for i in train_indices], [aankleuring_masks[i] for i in train_indices], [pat_ids[i] for i in train_indices]
        val_LGE_imgs, val_myo_masks, val_aankleuring_masks, val_pat_ids = [LGE_imgs[i] for i in val_indices], [myo_masks[i] for i in val_indices], [aankleuring_masks[i] for i in val_indices], [pat_ids[i] for i in val_indices]
        if '2D' in dataset:
            slice_indices = train_data[4] + val_data[4]
            train_slice_indices = [slice_indices[i] for i in train_indices]
            val_slice_indices = [slice_indices[i] for i in val_indices]
            train_dataset = AUMCDataset2D(train_LGE_imgs, train_myo_masks, train_aankleuring_masks, train_pat_ids, train_slice_indices, transform=transformations, max_value=max_value)
            val_dataset = AUMCDataset2D(val_LGE_imgs, val_myo_masks, val_aankleuring_masks, val_pat_ids, val_slice_indices, transform=test_transform, max_value=max_value)
        else:
            train_dataset = AUMCDataset3D(train_LGE_imgs, train_myo_masks, train_aankleuring_masks, train_pat_ids, transform=transformations, max_value=max_value)
            val_dataset = AUMCDataset3D(val_LGE_imgs, val_myo_masks, val_aankleuring_masks, val_pat_ids, transform=test_transform, max_value=max_value)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        dataloader_splits.append([train_loader, val_loader])

    return dataloader_splits


def get_data(dataset, only_test=False, transforms=[], fibrosis_model=None, myocard_model_version=None, use_only_fib='no', resize='resize', size=(256, 256), normalize=['clip'], seed=42):
    if resize == 'resize':
        size = (256,256)
    elif resize == 'none':
        size = None
    elif size[0] != 'smallest':
        size = (int(size[0]), int(size[1]))
    
    # read_in_fibrosis_AUMC_data returns list like [LGE_img, myo_masks, fibrosis_masks, pat_ids]

    if fibrosis_model is not None:
        if only_test:
            test_data = read_in_fibrosis_AUMC_data('test', dataset=dataset, myocard_model_version=myocard_model_version, resize=resize, size=size, normalize=normalize)
            if len(test_data[0]) != len(test_data[1]) or len(test_data[0]) != len(test_data[2]) or len(test_data[0]) != len(test_data[3]):
                raise ValueError(f'LGE imgs, myo masks, aankleuring masks and pat_ids should all have same length but got: {len(test_data[0])}, {len(test_data[1])}, {len(test_data[2])}, {len(test_data[3])}')
            train_data, val_data = None, None
        else:
            train_data = read_in_fibrosis_AUMC_data('train', dataset=dataset, myocard_model_version=myocard_model_version, resize=resize, size=size, normalize=normalize)
            val_data = read_in_fibrosis_AUMC_data('validation', dataset=dataset, myocard_model_version=myocard_model_version, resize=resize, size=size, normalize=normalize)
            test_data = read_in_fibrosis_AUMC_data('test', dataset=dataset, myocard_model_version=myocard_model_version, resize=resize, size=size, normalize=normalize)
            if len(train_data[0]) != len(train_data[1]) or len(train_data[0]) != len(train_data[2]) or len(train_data[0]) != len(train_data[3]):
                raise ValueError(f'train LGE imgs, myo masks, aankleuring masks and pat_ids should all have same length but got: {len(train_data[0])}, {len(train_data[1])}, {len(train_data[2])}, {len(train_data[3])}')
            if len(val_data[0]) != len(val_data[1]) or len(val_data[0]) != len(val_data[2]) or len(val_data[0]) != len(val_data[3]):
                raise ValueError(f'val LGE imgs, myo masks, aankleuring masks and pat_ids should all have same length but got: {len(test_data[0])}, {len(test_data[1])}, {len(test_data[2])}, {len(test_data[3])}')
            if len(test_data[0]) != len(test_data[1]) or len(test_data[0]) != len(test_data[2]) or len(test_data[0]) != len(test_data[3]):
                raise ValueError(f'test LGE imgs, myo masks, aankleuring masks and pat_ids should all have same length but got: {len(test_data[0])}, {len(test_data[1])}, {len(test_data[2])}, {len(test_data[3])}')
    else:
        if only_test:
            test_data = read_in_AUMC_data('test', dataset=dataset, resize=resize, size=size, normalize=normalize)
            if len(test_data[0]) != len(test_data[1]) or len(test_data[0]) != len(test_data[2]) or len(test_data[0]) != len(test_data[3]):
                raise ValueError(f'LGE imgs, myo masks, aankleuring masks and pat_ids should all have same length but got: {len(test_data[0])}, {len(test_data[1])}, {len(test_data[2])}, {len(test_data[3])}')
            train_data, val_data = None, None
        else:
            train_data = read_in_AUMC_data('train', dataset=dataset, resize=resize, size=size, normalize=normalize)
            val_data = read_in_AUMC_data('validation', dataset=dataset, resize=resize, size=size, normalize=normalize)
            test_data = read_in_AUMC_data('test', dataset=dataset, resize=resize, size=size, normalize=normalize)
            if len(train_data[0]) != len(train_data[1]) or len(train_data[0]) != len(train_data[2]) or len(train_data[0]) != len(train_data[3]):
                raise ValueError(f'train LGE imgs, myo masks, aankleuring masks and pat_ids should all have same length but got: {len(train_data[0])}, {len(train_data[1])}, {len(train_data[2])}, {len(train_data[3])}')
            if len(val_data[0]) != len(val_data[1]) or len(val_data[0]) != len(val_data[2]) or len(val_data[0]) != len(val_data[3]):
                raise ValueError(f'val LGE imgs, myo masks, aankleuring masks and pat_ids should all have same length but got: {len(test_data[0])}, {len(test_data[1])}, {len(test_data[2])}, {len(test_data[3])}')
            if len(test_data[0]) != len(test_data[1]) or len(test_data[0]) != len(test_data[2]) or len(test_data[0]) != len(test_data[3]):
                raise ValueError(f'test LGE imgs, myo masks, aankleuring masks and pat_ids should all have same length but got: {len(test_data[0])}, {len(test_data[1])}, {len(test_data[2])}, {len(test_data[3])}')
    if '2D' in dataset:
        if only_test:
            LGE_imgs_test, test_slices = get_all_slices(test_data[0])
            aankleuring_masks_test, _ = get_all_slices(test_data[2])
            pat_ids_test = get_all_patids(test_data[3], test_slices)
            slice_indices_test = get_slice_indices(pat_ids_test)
            test_data = [LGE_imgs_test, test_data[1], aankleuring_masks_test, pat_ids_test, slice_indices_test]
            if len(test_data[0]) != len(test_data[1]):
                raise ValueError(f'Number of LGE images ({len(test_data[0])}) should be equal to number of myocard_predictions ({len(test_data[1])})')
        else:
            LGE_imgs_train, train_slices = get_all_slices(train_data[0])
            aankleuring_masks_train, _ = get_all_slices(train_data[2])
            pat_ids_train = get_all_patids(train_data[3], train_slices)
            slice_indices_train = get_slice_indices(pat_ids_train)
            LGE_imgs_val, val_slices = get_all_slices(val_data[0])
            aankleuring_masks_val, _ = get_all_slices(val_data[2])
            pat_ids_val = get_all_patids(val_data[3], val_slices)
            slice_indices_val = get_slice_indices(pat_ids_val)
            LGE_imgs_test, test_slices = get_all_slices(test_data[0])
            aankleuring_masks_test, _ = get_all_slices(test_data[2])
            pat_ids_test = get_all_patids(test_data[3], test_slices)
            slice_indices_test = get_slice_indices(pat_ids_test)
            if fibrosis_model is not None:
                myo_masks_train, myo_masks_val, myo_masks_test = train_data[1], val_data[1], test_data[1]
            else:
                myo_masks_train, myo_masks_val, myo_masks_test = get_all_slices(train_data[1])[0], get_all_slices(val_data[1])[0], get_all_slices(test_data[1])[0]
            train_data = [LGE_imgs_train, myo_masks_train, aankleuring_masks_train, pat_ids_train, slice_indices_train]
            val_data = [LGE_imgs_val, myo_masks_val, aankleuring_masks_val, pat_ids_val, slice_indices_val]
            test_data = [LGE_imgs_test, myo_masks_test, aankleuring_masks_test, pat_ids_test, slice_indices_test]
            if len(train_data[0]) != len(train_data[1]):
                raise ValueError(f'Number of train LGE images ({len(train_data[0])}) should be equal to number of myocard_predictions ({len(train_data[1])})')
            if len(val_data[0]) != len(val_data[1]):
                raise ValueError(f'Number of val LGE images ({len(val_data[0])}) should be equal to number of myocard_predictions ({len(val_data[1])})')
            if len(test_data[0]) != len(test_data[1]):
                raise ValueError(f'Number of test LGE images ({len(test_data[0])}) should be equal to number of myocard_predictions ({len(test_data[1])})')
    return train_data, val_data, test_data

def perform_single_transformation(transform, LGE_image, max_value):
    if 'hflip' in transform:
        if random.random() < 0.5:
            for i in range(LGE_image.shape[1]):
                LGE_image[:,i,:,:] = transforms.RandomHorizontalFlip(p=1)(LGE_image[:,i,:,:])
    if 'vflip' in transform:
        if random.random() < 0.5:
            for i in range(LGE_image.shape[1]):
                LGE_image[:,i,:,:] = transforms.RandomVerticalFlip(p=1)(LGE_image[:,i,:,:])
    if 'rotate' in transform:
        if random.random() < 0.5:
            times = random.choice([1, 2, 3])
            for i in range(LGE_image.shape[1]):
                LGE_image[:,i,:,:] = torch.rot90(LGE_image[:,i,:,:], times, [1,2])
    if 'gamma_correction' in transform:
        if random.random() < 0.5:
            choice = np.random.choice(np.arange(-1.0,1.1,0.1))
            gamma = 2**choice
            normalized_LGE_img = LGE_image / max_value
            gamma_image = torch.pow(normalized_LGE_img, 1/gamma)
            LGE_image = gamma_image * max_value
            if torch.max(LGE_image) > max_value:
                raise ValueError(f"Computation must have gone wrong. Max value {torch.max(LGE_image).item()} should not exceed {max_value}")
            LGE_image = LGE_image.type(torch.int16)
    if 'shear' in transform:
        if random.random() < 0.3:
            degrees = random.choice([-20,-10,10,20])
            for i in range(LGE_image.shape[1]):
                LGE_image[:,i,:,:] = transforms.functional.affine(img=LGE_image[:,i,:,:], angle=0, translate=[0,0], scale=1, shear=degrees, interpolation=transforms.InterpolationMode.BILINEAR)
    if 'scale_after_gamma' in transform:
        mean, std = torch.mean(LGE_image), torch.std(LGE_image)
        LGE_image = (LGE_image-mean)/std
    return LGE_image

def perform_transformations(transform, LGE_image, myops_mask, aankleuring_mask, max_value, dims='2D'):
    if 'hflip' in transform:
        if random.random() < 0.5:
            if dims=='2D':
                LGE_image = transforms.RandomHorizontalFlip(p=1)(LGE_image)
                myops_mask = transforms.RandomHorizontalFlip(p=1)(myops_mask)
                aankleuring_mask = transforms.RandomHorizontalFlip(p=1)(aankleuring_mask)
            else:
                for i in range(LGE_image.shape[1]):
                    LGE_image[:,i,:,:] = transforms.RandomHorizontalFlip(p=1)(LGE_image[:,i,:,:])
                    myops_mask[:,i,:,:] = transforms.RandomHorizontalFlip(p=1)(myops_mask[:,i,:,:])
                    aankleuring_mask[:,i,:,:] = transforms.RandomHorizontalFlip(p=1)(aankleuring_mask[:,i,:,:])
    if 'vflip' in transform:
        if random.random() < 0.5:
            if dims=='2D':
                LGE_image = transforms.RandomVerticalFlip(p=1)(LGE_image)
                myops_mask = transforms.RandomVerticalFlip(p=1)(myops_mask)
                aankleuring_mask = transforms.RandomVerticalFlip(p=1)(aankleuring_mask)
            else:
                for i in range(LGE_image.shape[1]):
                    LGE_image[:,i,:,:] = transforms.RandomVerticalFlip(p=1)(LGE_image[:,i,:,:])
                    myops_mask[:,i,:,:] = transforms.RandomVerticalFlip(p=1)(myops_mask[:,i,:,:])
                    aankleuring_mask[:,i,:,:] = transforms.RandomVerticalFlip(p=1)(aankleuring_mask[:,i,:,:])
    if 'rotate' in transform:
        if random.random() < 0.5:
            times = random.choice([1, 2, 3])
            if dims=='2D':
                LGE_image = torch.rot90(LGE_image, times, [1,2])
                myops_mask = torch.rot90(myops_mask, times, [1,2])
                aankleuring_mask = torch.rot90(aankleuring_mask, times, [1,2])
            else:
                for i in range(LGE_image.shape[1]):
                    LGE_image[:,i,:,:] = torch.rot90(LGE_image[:,i,:,:], times, [1,2])
                    myops_mask[:,i,:,:] = torch.rot90(myops_mask[:,i,:,:], times, [1,2])
                    aankleuring_mask[:,i,:,:] = torch.rot90(aankleuring_mask[:,i,:,:], times, [1,2])
    if 'gamma_correction' in transform:
        if random.random() < 0.5:
            old_max = torch.max(LGE_image)
            choice = np.random.choice(np.arange(-1.0,1.1,0.1))
            gamma = 2**choice
            normalized_LGE_img = LGE_image / max_value
            gamma_image = torch.pow(normalized_LGE_img, 1/gamma)
            LGE_image = gamma_image * max_value
            if torch.max(LGE_image) > max_value:
                raise ValueError(f"Computation must have gone wrong. Max value {torch.max(LGE_image).item()} should not exceed {max_value}")
            LGE_image = LGE_image.type(torch.int16)
    if 'shear' in transform:
        if random.random() < 0.3:
            degrees = random.choice([-20,-10,10,20])
            if dims=='2D':
                LGE_image = transforms.functional.affine(img=LGE_image, angle=0, translate=[0,0], scale=1, shear=degrees, interpolation=transforms.InterpolationMode.BILINEAR)
                myops_mask = transforms.functional.affine(img=myops_mask, angle=0, translate=[0,0], scale=1, shear=degrees, interpolation=transforms.InterpolationMode.NEAREST)
                aankleuring_mask = transforms.functional.affine(img=aankleuring_mask, angle=0, translate=[0,0], scale=1, shear=degrees, interpolation=transforms.InterpolationMode.NEAREST)
            else:
                for i in range(LGE_image.shape[1]):
                    LGE_image[:,i,:,:] = transforms.functional.affine(img=LGE_image[:,i,:,:], angle=0, translate=[0,0], scale=1, shear=degrees, interpolation=transforms.InterpolationMode.BILINEAR)
                    myops_mask[:,i,:,:] = transforms.functional.affine(img=myops_mask[:,i,:,:], angle=0, translate=[0,0], scale=1, shear=degrees, interpolation=transforms.InterpolationMode.NEAREST)
                    aankleuring_mask[:,i,:,:] = transforms.functional.affine(img=aankleuring_mask[:,i,:,:], angle=0, translate=[0,0], scale=1, shear=degrees, interpolation=transforms.InterpolationMode.NEAREST)
    if 'scale_after_gamma' in transform:
        mean, std = torch.mean(LGE_image), torch.std(LGE_image)
        LGE_image = (LGE_image-mean)/std
    return LGE_image, myops_mask, aankleuring_mask

def get_loss_weights(train_dataset):
    myo_sum = 0
    total_elements = 0
    for myo_mask in train_dataset.myo_masks:
        myo_sum += np.sum(myo_mask)
        total_elements += myo_mask.size
    myo_weights = (total_elements-myo_sum)/myo_sum
    aankleuring_sum = 0
    for aankleuring_mask in train_dataset.aankleuring_masks:
        aankleuring_sum += np.sum(aankleuring_mask)
    aankleuring_weights = (total_elements-aankleuring_sum)/aankleuring_sum
    return [myo_weights, aankleuring_weights]

def get_all_slices(img_data):
    slice_counts = []
    all_slices = []
    for img in img_data:
        splits = np.split(img, img.shape[0])
        slice_counts.append(len(splits))
        all_slices.extend(splits)
    return all_slices, slice_counts

def get_all_patids(pat_ids, count_slices):
    extended_patids = []
    for i in range(len(pat_ids)):
        multiple_list = [pat_ids[i]] * count_slices[i]
        extended_patids.extend(multiple_list)
    return extended_patids

def get_bouding_boxes_slices(img_data, bounding_boxes):
    all_bounding_boxes = []
    for i, bb_value in enumerate(bounding_boxes):
        slices_count = img_data[i].shape[0]
        repeated_bb_values = np.repeat(np.expand_dims(bb_value, 0), slices_count, 0)
        all_bounding_boxes.extend(repeated_bb_values)
    return all_bounding_boxes

def get_slice_indices(pat_ids):
    indices = []
    slice_nr = 0
    prev_pat_id = None
    for pat_id in pat_ids:
        if prev_pat_id is None:
            prev_pat_id = pat_id
        elif prev_pat_id == pat_id:
            slice_nr += 1
        else:
            slice_nr = 0
        indices.append(slice_nr)
        prev_pat_id = pat_id
    if len(indices) != len(pat_ids):
        raise ValueError('Number of slice indices must be equal to number of patient ids')
    return indices

def get_only_fibrosis_data(LGE_imgs, myo_masks, aankleuring_masks, pat_ids, slice_indices):
    new_LGE_imgs, new_myo_masks, new_aankleuring_masks, new_pat_ids, new_slice_indices = [], [], [], [], []
    for i, (LGE_img, myo_mask, aankleuring_mask, pat_id, slice_index) in enumerate(zip(LGE_imgs, myo_masks, aankleuring_masks, pat_ids, slice_indices)):
        if np.all((aankleuring_mask == 0.)):
            continue
        else:
            new_LGE_imgs.append(LGE_img)
            new_myo_masks.append(myo_mask)
            new_aankleuring_masks.append(aankleuring_mask)
            new_pat_ids.append(pat_id)
            new_slice_indices.append(slice_index)
    return new_LGE_imgs, new_myo_masks, new_aankleuring_masks, new_pat_ids, new_slice_indices