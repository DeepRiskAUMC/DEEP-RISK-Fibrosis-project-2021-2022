
import os
import os.path
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from utils.preprocessing import compose_transforms_aumc, compose_transforms_emidec
from utils.train_val_split import train_val_oversample_AMC, train_val_oversample_EMIDEC

from data.dataset_dataloaders import AUMCDatsetLGE_MLP, EmidecDataset
from emidec_data_features.data_load_feature_extract import emidec_data_load_feature_extract

def load_latent_classifier_dataset(args, train=True):
    print(args.base_dir)
    train_transforms, val_transforms = compose_transforms_aumc(args)
    if train:
        TRAIN_VAL_DIR = Path(args.base_dir + '/train')
        assert TRAIN_VAL_DIR.exists()

        TRAIN_VAL_IDS = os.listdir(TRAIN_VAL_DIR)
        # # if MLP:
        # if args.label == 'Mortality':
        #     mortality_labels = [1 if mri_id.split('_')[1] == 'M' else 0 for mri_id in TRAIN_VAL_IDS]
        #     TRAIN_IDS, VAL_IDS = train_val_oversample_AMC(TRAIN_VAL_IDS, mortality_labels, do_oversampling=args.oversample)
        #     # train_test_split(TRAIN_VAL_IDS, test_size=val_size, random_state=11)
        # if args.label == 'AppropriateTherapy':
        #     apt_labels = [1 if mri_id.split('_')[2] == 'Y' else 0 for mri_id in TRAIN_VAL_IDS]
        #     TRAIN_IDS, VAL_IDS = train_val_oversample_AMC(TRAIN_VAL_IDS, apt_labels, do_oversampling=args.oversample)
        # else:
        #     train_size, val_size = args.train_ratio, 1 - args.train_ratio
        #     TRAIN_IDS, VAL_IDS = train_test_split(TRAIN_VAL_IDS, test_size=val_size, random_state=args.seed)

        dataset_all = AUMCDatsetLGE_MLP(TRAIN_VAL_DIR, TRAIN_VAL_IDS, image_name=args.image_name, label_name=args.label,
                                        predmyo=True, transforms=train_transforms, roi_crop=args.roi_crop, masked=args.masked, 
                                        img_ch=args.IMG, train=True)
        print(f"All data: {len(dataset_all)}")
        return dataset_all

    else: 

        TEST_DIR = Path(args.base_dir + '/test')    
        assert TEST_DIR.exists()
        TEST_IDS = os.listdir(TEST_DIR)
        dataset_test = AUMCDatsetLGE_MLP(TEST_DIR, TEST_IDS, image_name=args.image_name, label_name=args.label,
                                                predmyo=True, transforms=val_transforms, roi_crop=args.roi_crop, 
                                                masked=args.masked, img_ch=args.IMG)
        
        print(f"Test data: {len(dataset_test)}")
        return dataset_test
    
def oversample_smote(train_dataset, seed):
        # Prepare data for SMOTE
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        train_latent, train_labels = next(iter(train_loader))

        # train_latent, train_labels = next(iter(train_loader)) 

        if len(train_latent.shape) == 5:
            train_latent = train_latent.reshape(train_labels.shape[0], 64*4*4*4)

        X_for_smote = train_latent.numpy()
        y_for_smote = train_labels.numpy()

        # Calculate class weights to determine SMOTE sampling strategy
        class_weights = compute_class_weight('balanced', classes=np.unique(y_for_smote), y=y_for_smote)
        print('class weights', class_weights)
        # oversample = SMOTE(sampling_strategy=dict(zip(np.unique(y_for_smote), class_weights)), random_state=42)
        oversample = SMOTE(random_state=seed)
        X_resampled, y_resampled = oversample.fit_resample(X_for_smote, y_for_smote)

        # If SMOTE generated samples, update img_array and label
        if len(X_resampled) > 0:
            # Reshape back to the original shape
            X_resampled = X_resampled.reshape(X_resampled.shape[0], 64, 4, 4, 4)

            # Update img_array and label
            train_latent = torch.tensor(X_resampled).float()
            train_labels = torch.tensor(y_resampled).long()
            train_dataset = TensorDataset(train_latent, train_labels)
        return train_dataset
        
        

    
def load_dataset_emidec(args, train=True, external_val=False):
    train_transforms, val_transforms = compose_transforms_emidec(args)
    if train: 
        TRAIN_VAL_DIR = Path(args.base_dir + '/train')
        assert TRAIN_VAL_DIR.exists()
        print(args.base_dir)
        images_train, labels_train, indices = emidec_data_load_feature_extract(TRAIN_VAL_DIR, cropped=True)
        train_images, val_images, train_labels,  val_labels = train_val_oversample_EMIDEC(images_train, labels_train, indices, do_oversampling = args.oversample)
        train_dataset = EmidecDataset(train_images, train_labels, train_transforms) 
        val_dataset = EmidecDataset(val_images, val_labels, val_transforms) 
        if args.class_weights == True:    
            args.class_weights = train_dataset.compute_class_weights()
            print("Class Weights:", args.class_weights)

        return train_dataset,  val_dataset
    else:
        if external_val: 
            TEST_DIR = Path(args.base_dir) 
        else:
            TEST_DIR = Path(args.base_dir + '/test')    
        assert TEST_DIR.exists()
        images_test, labels_test, _ = emidec_data_load_feature_extract(TEST_DIR, cropped=True)
        test_dataset = EmidecDataset(images_test, labels_test, val_transforms) 

        return test_dataset
    
def load_dataset_aumc(args, train=True, external_val=False):
    print(args.base_dir)
    train_transforms, val_transforms = compose_transforms_aumc(args)
    if train:
        TRAIN_VAL_DIR = Path(args.base_dir + '/train')
        assert TRAIN_VAL_DIR.exists()
        

        TRAIN_VAL_IDS = os.listdir(TRAIN_VAL_DIR)
        if args.label == 'Mortality':
            mortality_labels = [1 if mri_id.split('_')[1] == 'M' else 0 for mri_id in TRAIN_VAL_IDS]
            TRAIN_IDS, VAL_IDS = train_val_oversample_AMC(TRAIN_VAL_IDS, mortality_labels, do_oversampling=args.oversample)
        if args.label == 'AppropriateTherapy':
            apt_labels = [1 if mri_id.split('_')[2] == 'Y' else 0 for mri_id in TRAIN_VAL_IDS]
            TRAIN_IDS, VAL_IDS = train_val_oversample_AMC(TRAIN_VAL_IDS, apt_labels, do_oversampling=args.oversample)
        else:
            train_size, val_size = args.train_ratio, 1 - args.train_ratio
            TRAIN_IDS, VAL_IDS = train_test_split(TRAIN_VAL_IDS, test_size=val_size, random_state=args.seed)

        dataset_train = AUMCDatsetLGE_MLP(TRAIN_VAL_DIR, TRAIN_IDS, image_name=args.image_name, label_name=args.label,
                                            predmyo=True, transforms=train_transforms, roi_crop=args.roi_crop, masked=args.masked, 
                                            img_ch=args.IMG, train=True)
        dataset_val = AUMCDatsetLGE_MLP(TRAIN_VAL_DIR, VAL_IDS, image_name=args.image_name, label_name=args.label,
                                            predmyo=True, transforms=val_transforms, roi_crop=args.roi_crop, 
                                            masked=args.masked, img_ch=args.IMG)
        if args.model_type == 'vae_mlp':
            # # Compute class weights
            if args.class_weights:
                if args.label == 'Mortality' or args.label== 'AppropriateTherapy':
                    if args.oversample:
                        args.class_weights = dataset_val.compute_class_weights() 
                        print("Class Weights:", args.class_weights)
                    else:
                        args.class_weights = dataset_train.compute_class_weights() 
                        print("Class Weights:", args.class_weights)
                        
                    # args.class_weights[1] = args.class_weights[1] * 2
                    # print("Class Weights:", args.class_weights)

        print(f"Train data: {len(dataset_train)}")
        print(f"Validation data: {len(dataset_val)}")
        return dataset_train, dataset_val
    else: 
        if external_val:
            TEST_DIR = Path(args.base_dir + '/train') 
        else:
            TEST_DIR = Path(args.base_dir + '/test')    
        assert TEST_DIR.exists()
        TEST_IDS = os.listdir(TEST_DIR)
        if args.model_type == 'vae' or args.model_type == 'ae':
            dataset_test = AUMCDatsetLGE_MLP(TEST_DIR, TEST_IDS, image_name=args.image_name, 
                                                predmyo=True, transforms=val_transforms, roi_crop=args.roi_crop, 
                                                masked=args.masked, img_ch=args.IMG)
        if args.model_type == 'vae_mlp':
             dataset_test = AUMCDatsetLGE_MLP(TEST_DIR, TEST_IDS, image_name=args.image_name, label_name=args.label,
                                                predmyo=True, transforms=val_transforms, roi_crop=args.roi_crop, 
                                                masked=args.masked, img_ch=args.IMG)
        
        print(f"Test data: {len(dataset_test)}")
        return dataset_test