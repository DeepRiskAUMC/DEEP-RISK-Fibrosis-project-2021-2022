from torch.utils.data.dataloader import DataLoader
from data_loading.import_AUMC_dataset import read_in_AUMC_classification_data, read_in_AUMC_classification_clinical_data
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np

class AUMCClassificationDataset(Dataset):
    def __init__(self, LGE_images, labels, pat_ids, transform=None, max_value=4095) -> None:
        super().__init__()
        self.LGE_images = LGE_images
        self.labels = labels
        self.pat_ids = pat_ids
        self.transform = transform
        self.max_value = max_value
        print(f"Maximum value of dataset: {self.max_value}. Transforms: {transform}")
    
    def __len__(self):
        return len(self.LGE_images)

    def __getitem__(self, index):
        LGE_image = torch.from_numpy(self.LGE_images[index])
        if LGE_image.dim() < 4:
            LGE_image = LGE_image.unsqueeze(dim=0)
        label = self.labels[index]
        pat_id = self.pat_ids[index]

        if self.transform is not None:
            LGE_image = perform_single_transformation(self.transform, LGE_image, self.max_value)
        return LGE_image, label, pat_id

class AUMCClassificationClinicalDataset(Dataset):
    def __init__(self, LGE_images, clinical_features, labels, pat_ids, transform=None, max_value=4095) -> None:
        super().__init__()
        self.LGE_images = LGE_images
        self.clinical_features = clinical_features
        self.labels = labels
        self.pat_ids = pat_ids
        self.transform = transform
        self.max_value = max_value
        print(f"Maximum value of dataset: {self.max_value}. Transforms: {self.transform}")
    
    def __len__(self):
        return len(self.LGE_images)

    def __getitem__(self, index):
        LGE_image = torch.from_numpy(self.LGE_images[index])
        if LGE_image.dim() < 4:
            LGE_image = LGE_image.unsqueeze(dim=0)
        clinical = torch.FloatTensor(self.clinical_features[index])
        label = self.labels[index]
        pat_id = self.pat_ids[index]

        if self.transform is not None:
            LGE_image = perform_single_transformation(self.transform, LGE_image, self.max_value)
        return LGE_image, clinical, label, pat_id

def load_classification_data(dataset, batch_size=8, val_batch_size='same', num_workers=1, only_test=False, transformations=[], resize='resize', size=(256, 256), normalize=['clip'], mean_values=None):
    if resize == 'resize':
        size = (256,256)
    elif resize == 'none':
        size = None
    elif size[0] != 'smallest':
        size = (int(size[0]), int(size[1]))
    
    if 'AUMC3D' in dataset:
        if only_test:
            val_LGE_imgs, validation_labels, val_pat_ids, val_loss_weights = read_in_AUMC_classification_data('validation', resize=resize, size=size, normalize=normalize, dataset=dataset)
            test_LGE_imgs, test_labels, test_pat_ids, test_loss_weights = read_in_AUMC_classification_data('test', resize=resize, size=size, normalize=normalize, dataset=dataset)
            max_value = max([np.max(val_LGE_imgs), np.max(test_LGE_imgs)])
        else:
            train_LGE_imgs, train_labels, train_pat_ids, train_loss_weights = read_in_AUMC_classification_data('train', resize=resize, size=size, normalize=normalize, dataset=dataset)
            val_LGE_imgs, validation_labels, val_pat_ids, val_loss_weights = read_in_AUMC_classification_data('validation', resize=resize, size=size, normalize=normalize, dataset=dataset)
            test_LGE_imgs, test_labels, test_pat_ids, test_loss_weights = read_in_AUMC_classification_data('test', resize=resize, size=size, normalize=normalize, dataset=dataset)
            max_value = max([np.max(train_LGE_imgs), np.max(val_LGE_imgs), np.max(test_LGE_imgs)])
        if 'scale_after_gamma' in normalize or 'scale_after_gamma' in transformations:
            test_transform = ['scale_after_gamma']
        else:
            test_transform = []
        
        if not only_test:
            train_dataset = AUMCClassificationDataset(train_LGE_imgs, train_labels, train_pat_ids, transform=transformations, max_value=max_value)
        val_dataset = AUMCClassificationDataset(val_LGE_imgs, validation_labels, val_pat_ids, transform=test_transform, max_value=max_value)
        test_dataset = AUMCClassificationDataset(test_LGE_imgs, test_labels, test_pat_ids, transform=test_transform, max_value=max_value)
    else:
        raise ValueError(f'Dataset {dataset} not valid for classification')
    if val_batch_size == 'full_set':
        val_batch_size = len(val_dataset)
    elif val_batch_size == 'same':
        val_batch_size = batch_size
    else:
        raise ValueError()
    if not only_test:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if only_test:
        return val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader, (train_loss_weights, val_loss_weights)

def load_classification_data_clinical(dataset, batch_size=8, val_batch_size='same', num_workers=1, only_test=False, transformations=[], resize='resize', size=(256, 256), normalize=['clip'], mean_values=None, cross_validation=False):
    if resize == 'resize':
        size = (256,256)
    elif resize == 'none':
        size = None
    elif size[0] != 'smallest':
        size = (int(size[0]), int(size[1]))
    
    if 'AUMC3D' in dataset:
        if only_test:
            val_LGE_imgs, val_clinical, validation_labels, val_pat_ids, val_loss_weights, _ = read_in_AUMC_classification_clinical_data('validation', resize=resize, size=size, normalize=normalize, mean_values=mean_values, dataset=dataset)
            test_LGE_imgs, test_clinical, test_labels, test_pat_ids, test_loss_weights, _ = read_in_AUMC_classification_clinical_data('test', resize=resize, size=size, normalize=normalize, mean_values=mean_values, dataset=dataset)
            max_value = max([np.max(val_LGE_imgs), np.max(test_LGE_imgs)])
        else:
            train_LGE_imgs, train_clinical, train_labels, train_pat_ids, train_loss_weights, mean_values = read_in_AUMC_classification_clinical_data('train', resize=resize, size=size, normalize=normalize, dataset=dataset)
            val_LGE_imgs, val_clinical, validation_labels, val_pat_ids, val_loss_weights, _ = read_in_AUMC_classification_clinical_data('validation', resize=resize, size=size, normalize=normalize, mean_values=mean_values, dataset=dataset)
            test_LGE_imgs, test_clinical, test_labels, test_pat_ids, test_loss_weights, _ = read_in_AUMC_classification_clinical_data('test', resize=resize, size=size, normalize=normalize, mean_values=mean_values, dataset=dataset)
            max_value = max([np.max(train_LGE_imgs), np.max(val_LGE_imgs), np.max(test_LGE_imgs)])
        if 'scale_after_gamma' in normalize or 'scale_after_gamma' in transformations:
            test_transform = ['scale_after_gamma']
        else:
            test_transform = []
        
        if not only_test:
            train_dataset = AUMCClassificationClinicalDataset(train_LGE_imgs, train_clinical, train_labels, train_pat_ids, transform=transformations, max_value=max_value)
        val_dataset = AUMCClassificationClinicalDataset(val_LGE_imgs, val_clinical, validation_labels, val_pat_ids, transform=test_transform, max_value=max_value)
        test_dataset = AUMCClassificationClinicalDataset(test_LGE_imgs, test_clinical, test_labels, test_pat_ids, transform=test_transform, max_value=max_value)
    else:
        raise ValueError(f'Dataset {dataset} not valid for classification')
    if val_batch_size == 'full_set':
        val_batch_size = len(val_dataset)
    elif val_batch_size == 'same':
        val_batch_size = batch_size
    else:
        raise ValueError()
    
    if not only_test:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if only_test:
        return val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader, (train_loss_weights, val_loss_weights)

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