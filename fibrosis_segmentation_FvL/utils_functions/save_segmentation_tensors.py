import sys
sys.path.append("..")
import torch
import numpy as np
import h5py
import csv
import os
import argparse
from utils import load_pretrained_layers
from data_loading.load_classification_data import load_classification_data_clinical
from train_fibrosis_segmentation_with_myo_pl import SegmentationModel as FibSegmentationModel
from train_myocard_segmentation_pl import SegmentationModel as MyoSegmentationModel
from models.segmentation_models import Floor_2d_Unet_without_final_conv

class CreateMyoModels():
    def __init__(self,myo_checkpoint, features_or_probs='features'):
        super().__init__()
        self.features_or_probs = features_or_probs
        full_myo_model = MyoSegmentationModel.load_from_checkpoint(myo_checkpoint)
        self.segment_model_name = full_myo_model.model_name
        try:
            self.bilinear = full_myo_model.bilinear
        except:
            self.bilinear = False
        self.feature_multiplication = full_myo_model.feature_multiplication
        full_myo_model = full_myo_model.model
        if features_or_probs == 'probs':
            self.myo_model = full_myo_model
            self.segment_model_output_size = 1
            self.sigmoid_finish = full_myo_model.sigmoid_finish
        elif features_or_probs == 'features':
            if self.segment_model_name == 'Floor_UNet2D':
                myo_model = Floor_2d_Unet_without_final_conv(1, 1, bilinear=self.bilinear, feature_multiplication=self.feature_multiplication)
                self.myo_model = load_pretrained_layers(myo_model, full_myo_model)
                self.segment_model_output_size = 16 * self.feature_multiplication
            else:
                raise ValueError(f'Not implemented for {self.segment_model_name}')
        else:
            raise ValueError(f"term {features_or_probs} not valid for features_or_probs")
        
        self.myo_model.requires_grad_(False)
        del full_myo_model

    def get_features(self, imgs):
        imgs = imgs.float()
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(dim=0)
        if '2D' in self.segment_model_name:
            try:
                input_datatype = imgs.dtype
            except:
                input_datatype = imgs.type()
            device = 'cuda' if imgs.is_cuda else 'cpu'
            myo_pred = torch.zeros((imgs.shape[0], self.segment_model_output_size, imgs.shape[2], imgs.shape[3], imgs.shape[4]), dtype=input_datatype, device=device)
            for i in range(imgs.shape[2]):
                if torch.any(imgs[:,:,i,:,:] > 0):
                    LGE_slice = imgs[:,:,i,:,:]
                    myo_pred_slice = self.myo_model(LGE_slice)
                    if self.features_or_probs == 'probs':
                        if not self.sigmoid_finish:
                            myo_pred_slice = torch.sigmoid(myo_pred_slice)
                    myo_pred[:,:,i,:,:] = myo_pred_slice
                    del myo_pred_slice
        return myo_pred.squeeze()

class CreateFibModels():
    def __init__(self,fibrosis_checkpoint, features_or_probs='features'):
        super().__init__()
        self.features_or_probs = features_or_probs
        full_segmentation_model = FibSegmentationModel.load_from_checkpoint(fibrosis_checkpoint, myo_checkpoint=r"/home/flieshout/deep_risk_models/fibrosis_segmentation_FvL/outputs/segment_logs/myocard/lightning_logs/version_29/checkpoints/epoch=111-step=13216.ckpt")
        self.segment_model_name = full_segmentation_model.model_name
        self.myo_model = full_segmentation_model.myo_model
        full_fib_model = full_segmentation_model.fib_model

        self.bilinear = full_fib_model.bilinear
        self.feature_multiplication = full_fib_model.feature_multiplication
        
        self.fib_model_name = full_fib_model._get_name()
        if features_or_probs == 'probs':
            self.fib_model = full_fib_model
            self.segment_model_output_size = 1
            self.sigmoid_finish = full_segmentation_model.sigmoid_finish
        elif features_or_probs == 'features':
            if self.fib_model_name == 'Floor_2d_Unet':
                fib_model = Floor_2d_Unet_without_final_conv(2, 1, bilinear=self.bilinear, feature_multiplication=self.feature_multiplication)
                self.fib_model = load_pretrained_layers(fib_model, full_fib_model)
                self.segment_model_output_size = 16 * self.feature_multiplication
        else:
            raise ValueError(f"term {features_or_probs} not valid for features_or_probs")

        self.myo_model.freeze()
        self.fib_model.requires_grad_(False)
        del full_fib_model
        

    def get_features(self, imgs):
        imgs = imgs.float()
        if imgs.dim() == 4:
            imgs = imgs.unsqueeze(dim=0)
        # print(imgs.shape)
        myo_pred = torch.zeros_like(imgs)
        for i in range(imgs.shape[2]):
            if torch.any(imgs[:,:,i,:,:] > 0):
                LGE_slice = imgs[:,:,i,:,:]
                myo_pred_slice = self.myo_model(LGE_slice)
                myo_pred[:,:,i,:,:] = myo_pred_slice
                del myo_pred_slice
        
        if '2D' in self.segment_model_name:
            try:
                input_datatype = imgs.dtype
            except:
                input_datatype = imgs.type()
            device = 'cuda' if imgs.is_cuda else 'cpu'
            fib_pred = torch.zeros((imgs.shape[0], self.segment_model_output_size, imgs.shape[2], imgs.shape[3], imgs.shape[4]), dtype=input_datatype, device=device)
            for i in range(imgs.shape[2]):
                if torch.any(imgs[:,:,i,:,:] > 0):
                    LGE_slice = imgs[:,:,i,:,:]
                    myo_slice = myo_pred[:,:,i,:,:]
                    concat_slice = torch.cat([LGE_slice, myo_slice], dim=1)
                    fib_pred_slice = self.fib_model(concat_slice)
                    if self.features_or_probs == 'probs':
                        if not self.sigmoid_finish:
                            fib_pred_slice = torch.sigmoid(fib_pred_slice)
                    fib_pred[:,:,i,:,:] = fib_pred_slice
                    del fib_pred_slice
        return myo_pred.squeeze(), fib_pred.squeeze()

def save_features(args):
    version_nr = args.model_path.split('version_')[-1].split('/')[0]
    saving_folder = os.path.join('segment_output/segmentation_tensors', f'version_{version_nr}')
    os.makedirs(saving_folder, exist_ok=True)

    model = CreateFibModels(args.model_path, features_or_probs='probs')
    train_loader, val_loader, test_loader, _ = load_classification_data_clinical(args.dataset,
                                                            batch_size=args.batch_size,
                                                            num_workers=args.num_workers,
                                                            only_test=False,
                                                            resize=args.resize,
                                                            size = args.size,
                                                            normalize=args.normalize)

    for dataloader in [train_loader, val_loader, test_loader]:
        for batch in dataloader:
            LGE_imgs, clinical_features, labels, pat_ids = batch
            myocard_feature_tensor, fibrosis_feature_tensor = model.get_features(LGE_imgs)
            for i in range(myocard_feature_tensor.shape[0]):
                torch.save(myocard_feature_tensor[i], os.path.join(saving_folder, f'myocard_{pat_ids[i]}.pt'))
                torch.save(fibrosis_feature_tensor[i], os.path.join(saving_folder, f'fibrosis_{pat_ids[i]}.pt'))

def save_features_hdf5(args):
    version_nr = args.model_path.split('version_')[-1].split('/')[0]
    if 'myocard' in args.model_path.split('segment_logs')[1]:
        myo_or_fib = 'myocardium'
        model = CreateMyoModels(args.model_path, features_or_probs='features')
    else:
        myo_or_fib = 'fibrosis'
        model = CreateFibModels(args.model_path, features_or_probs='features')
    saving_folder = os.path.join(args.output_path, 'segmentation_tensors_hdf5', f'{myo_or_fib}_version_{version_nr}_{args.dataset}')
    os.makedirs(saving_folder, exist_ok=True)
    train_loader, val_loader, test_loader, _ = load_classification_data_clinical(args.dataset,
                                                            batch_size=args.batch_size,
                                                            num_workers=args.num_workers,
                                                            only_test=False,
                                                            resize=args.resize,
                                                            size = args.size,
                                                            normalize=args.normalize)
    with h5py.File(os.path.join(saving_folder, "deeprisk_" + myo_or_fib + "_features_n=535.hdf5"), "w") as f:
        for split, dataloader in zip(['validation', 'test', 'train'], [val_loader, test_loader, train_loader]):
            print(f'Using data from {split} split')
            group = f.create_group(split)
            all_feature_tensors, all_labels, all_pat_ids = None, None, []
            for i, batch in enumerate(dataloader):
                print('batch:', i)
                LGE_imgs, clinical_features, labels, pat_ids = batch
                labels = torch.stack(labels).T
                if myo_or_fib == 'myocardium':
                    features_tensor = model.get_features(LGE_imgs)
                else:
                    _, features_tensor = model.get_features(LGE_imgs)
                if all_feature_tensors is None: 
                    all_LGE_imgs = LGE_imgs
                    all_feature_tensors = features_tensor
                    all_labels = labels
                    all_pat_ids = pat_ids
                else:
                    all_LGE_imgs = torch.cat([all_LGE_imgs, LGE_imgs], dim=0)
                    all_feature_tensors = torch.cat([all_feature_tensors, features_tensor], dim=0)
                    all_labels = torch.cat([all_labels, labels], dim=0)
                    all_pat_ids += pat_ids
            all_LGE_imgs, all_feature_tensors, all_labels = all_LGE_imgs.cpu().detach().numpy(), all_feature_tensors.cpu().detach().numpy(), all_labels.cpu().detach().numpy()
            # all_pat_ids = np.array(all_pat_ids, dtype=object).view(-1,1)
            all_pat_ids = [n.encode("ascii", "ignore") for n in all_pat_ids]
            print('shapes LGE_imgs/feature_tensors/labels/pat_ids:')
            print(all_LGE_imgs.shape, all_feature_tensors.shape, all_labels.shape, len(all_pat_ids))
            print(all_LGE_imgs.dtype, all_feature_tensors.dtype, all_labels.dtype)
            LGE_dataset = group.create_dataset("LGE_imgs", data=all_LGE_imgs)
            features_dataset = group.create_dataset(f"{myo_or_fib}_features", data=all_feature_tensors)
            labels_dataset = group.create_dataset("labels", data=all_labels)
            pat_ids = group.create_dataset("pat_ids", (len(all_pat_ids),1),'S10', data=all_pat_ids)

def save_features_hdf5_cross_validation(args):
    version_nr = args.model_path.split('version_')[-1].split('/')[0]
    if 'myocard' in args.model_path.split('segment_logs')[1]:
        myo_or_fib = 'myocardium'
        model = CreateMyoModels(args.model_path, features_or_probs='features')
    else:
        myo_or_fib = 'fibrosis'
        model = CreateFibModels(args.model_path, features_or_probs='features')
    saving_folder = os.path.join(args.output_path, 'segmentation_tensors_hdf5', f'{myo_or_fib}_version_{version_nr}_{args.dataset}')
    os.makedirs(saving_folder, exist_ok=True)
    train_loader, val_loader, test_loader, _ = load_classification_data_clinical(args.dataset,
                                                            batch_size=args.batch_size,
                                                            num_workers=args.num_workers,
                                                            only_test=False,
                                                            resize=args.resize,
                                                            size = args.size,
                                                            normalize=args.normalize,
                                                            cross_validation=True)
    with h5py.File(os.path.join(saving_folder, "deeprisk_" + myo_or_fib + "_features_" + args.dataset + "_n=419.hdf5"), "w") as f:
        for split, dataloader in zip(['validation', 'test', 'train'], [val_loader, test_loader, train_loader]):
            print(f'Using data from {split} split')
            group = f.create_group(split)
            all_feature_tensors, all_labels, all_pat_ids = None, None, []
            for i, batch in enumerate(dataloader):
                print('batch:', i)
                LGE_imgs, clinical_features, labels, pat_ids = batch
                labels = torch.stack(labels).T
                if myo_or_fib == 'myocardium':
                    features_tensor = model.get_features(LGE_imgs)
                else:
                    _, features_tensor = model.get_features(LGE_imgs)
                if all_feature_tensors is None: 
                    all_LGE_imgs = LGE_imgs
                    all_feature_tensors = features_tensor
                    all_labels = labels
                    all_pat_ids = pat_ids
                else:
                    all_LGE_imgs = torch.cat([all_LGE_imgs, LGE_imgs], dim=0)
                    all_feature_tensors = torch.cat([all_feature_tensors, features_tensor], dim=0)
                    all_labels = torch.cat([all_labels, labels], dim=0)
                    all_pat_ids += pat_ids
            all_LGE_imgs, all_feature_tensors, all_labels = all_LGE_imgs.cpu().detach().numpy(), all_feature_tensors.cpu().detach().numpy(), all_labels.cpu().detach().numpy()
            # all_pat_ids = np.array(all_pat_ids, dtype=object).view(-1,1)
            all_pat_ids = [n.encode("ascii", "ignore") for n in all_pat_ids]
            print('shapes LGE_imgs/feature_tensors/labels/pat_ids:')
            print(all_LGE_imgs.shape, all_feature_tensors.shape, all_labels.shape, len(all_pat_ids))
            print(all_LGE_imgs.dtype, all_feature_tensors.dtype, all_labels.dtype)
            LGE_dataset = group.create_dataset("LGE_imgs", data=all_LGE_imgs)
            features_dataset = group.create_dataset(f"{myo_or_fib}_features", data=all_feature_tensors)
            labels_dataset = group.create_dataset("labels", data=all_labels)
            pat_ids = group.create_dataset("pat_ids", (len(all_pat_ids),1),'S10', data=all_pat_ids)

def save_probs(args):
    version_nr = args.model_path.split('version_')[-1].split('/')[0]
    saving_folder = os.path.join(args.output_path, 'segmentation_probs', f'version_{version_nr}_{args.dataset}')
    os.makedirs(saving_folder, exist_ok=True)

    model = CreateFibModels(args.model_path, features_or_probs='probs')
    train_loader, val_loader, test_loader, _ = load_classification_data_clinical(args.dataset,
                                                            batch_size=args.batch_size,
                                                            num_workers=args.num_workers,
                                                            only_test=False,
                                                            resize=args.resize,
                                                            size = args.size,
                                                            normalize=args.normalize)

    with h5py.File(os.path.join(saving_folder, "deeprisk_myocard_fibrosis_probabilities_n=535.hdf5"), "w") as f:
        for split, dataloader in zip(['validation', 'test', 'train'], [val_loader, test_loader, train_loader]):
            print(f'Using data from {split} split')
            group = f.create_group(split)
            all_myo_prob_tensors, all_fib_prob_tensors, all_labels, all_pat_ids = None, None, None, []
            for i, batch in enumerate(dataloader):
                print('batch:', i)
                LGE_imgs, _, labels, pat_ids = batch
                labels = torch.stack(labels).T
                myocard_probs_tensor, fibrosis_probs_tensor = model.get_features(LGE_imgs)
                if all_myo_prob_tensors is None: 
                    all_LGE_imgs = LGE_imgs
                    all_myo_prob_tensors = myocard_probs_tensor
                    all_fib_prob_tensors = fibrosis_probs_tensor
                    all_labels = labels
                    all_pat_ids = pat_ids
                else:
                    all_LGE_imgs = torch.cat([all_LGE_imgs, LGE_imgs], dim=0)
                    all_myo_prob_tensors = torch.cat([all_myo_prob_tensors, myocard_probs_tensor], dim=0)
                    all_fib_prob_tensors = torch.cat([all_fib_prob_tensors, fibrosis_probs_tensor], dim=0)
                    all_labels = torch.cat([all_labels, labels], dim=0)
                    all_pat_ids += pat_ids
            all_LGE_imgs, all_myo_prob_tensors, all_fib_prob_tensors, all_labels = all_LGE_imgs.cpu().detach().numpy(), all_myo_prob_tensors.cpu().detach().numpy(), all_fib_prob_tensors.cpu().detach().numpy(), all_labels.cpu().detach().numpy()
            # all_pat_ids = np.array(all_pat_ids, dtype=object).view(-1,1)
            all_pat_ids = [n.encode("ascii", "ignore") for n in all_pat_ids]
            print('shapes LGE_imgs/myo_probs/fib_probs/labels/pat_ids:')
            print(all_LGE_imgs.shape, all_myo_prob_tensors.shape, all_fib_prob_tensors.shape, all_labels.shape, len(all_pat_ids))
            print(all_LGE_imgs.dtype, all_myo_prob_tensors.dtype, all_fib_prob_tensors.dtype, all_labels.dtype)
            LGE_dataset = group.create_dataset("LGE_imgs", data=all_LGE_imgs)
            myo_probs_dataset = group.create_dataset("myo_probs", data=all_myo_prob_tensors)
            fib_probs_dataset = group.create_dataset("fib_probs", data=all_fib_prob_tensors)
            labels_dataset = group.create_dataset("labels", data=all_labels)
            pat_ids = group.create_dataset("pat_ids", (len(all_pat_ids),1),'S10', data=all_pat_ids)

def save_probs_cross_validation(args):
    version_nr = args.model_path.split('version_')[-1].split('/')[0]
    saving_folder = os.path.join(args.output_path, 'segmentation_probs', f'version_{version_nr}_{args.dataset}')
    os.makedirs(saving_folder, exist_ok=True)

    model = CreateFibModels(args.model_path, features_or_probs='probs')
    train_loader, val_loader, test_loader, _ = load_classification_data_clinical(args.dataset,
                                                            batch_size=args.batch_size,
                                                            num_workers=args.num_workers,
                                                            only_test=False,
                                                            resize=args.resize,
                                                            size = args.size,
                                                            normalize=args.normalize,
                                                            cross_validation=True)

    with h5py.File(os.path.join(saving_folder, "deeprisk_myocard_fibrosis_probabilities_" + args.dataset + "_n=419.hdf5"), "w") as f:
        for split, dataloader in zip(['validation', 'test', 'train'], [val_loader, test_loader, train_loader]):
            print(f'Using data from {split} split')
            group = f.create_group(split)
            all_myo_prob_tensors, all_fib_prob_tensors, all_labels, all_pat_ids = None, None, None, []
            for i, batch in enumerate(dataloader):
                print('batch:', i)
                LGE_imgs, _, labels, pat_ids = batch
                labels = torch.stack(labels).T
                myocard_probs_tensor, fibrosis_probs_tensor = model.get_features(LGE_imgs)
                if all_myo_prob_tensors is None: 
                    all_LGE_imgs = LGE_imgs
                    all_myo_prob_tensors = myocard_probs_tensor
                    all_fib_prob_tensors = fibrosis_probs_tensor
                    all_labels = labels
                    all_pat_ids = pat_ids
                else:
                    all_LGE_imgs = torch.cat([all_LGE_imgs, LGE_imgs], dim=0)
                    all_myo_prob_tensors = torch.cat([all_myo_prob_tensors, myocard_probs_tensor], dim=0)
                    all_fib_prob_tensors = torch.cat([all_fib_prob_tensors, fibrosis_probs_tensor], dim=0)
                    all_labels = torch.cat([all_labels, labels], dim=0)
                    all_pat_ids += pat_ids
            all_LGE_imgs, all_myo_prob_tensors, all_fib_prob_tensors, all_labels = all_LGE_imgs.cpu().detach().numpy(), all_myo_prob_tensors.cpu().detach().numpy(), all_fib_prob_tensors.cpu().detach().numpy(), all_labels.cpu().detach().numpy()
            # all_pat_ids = np.array(all_pat_ids, dtype=object).view(-1,1)
            all_pat_ids = [n.encode("ascii", "ignore") for n in all_pat_ids]
            print('shapes LGE_imgs/myo_probs/fib_probs/labels/pat_ids:')
            print(all_LGE_imgs.shape, all_myo_prob_tensors.shape, all_fib_prob_tensors.shape, all_labels.shape, len(all_pat_ids))
            print(all_LGE_imgs.dtype, all_myo_prob_tensors.dtype, all_fib_prob_tensors.dtype, all_labels.dtype)
            LGE_dataset = group.create_dataset("LGE_imgs", data=all_LGE_imgs)
            myo_probs_dataset = group.create_dataset("myo_probs", data=all_myo_prob_tensors)
            fib_probs_dataset = group.create_dataset("fib_probs", data=all_fib_prob_tensors)
            labels_dataset = group.create_dataset("labels", data=all_labels)
            pat_ids = group.create_dataset("pat_ids", (len(all_pat_ids),1),'S10', data=all_pat_ids)

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model_path', default='/home/flieshout/deep_risk_models/fibrosis_segmentation_FvL/outputs/segment_logs/fibrosis/lightning_logs/version_77/checkpoints/epoch=95-step=11328.ckpt', type=str,
                        help='Path to trained model')
    parser.add_argument('--normalize', default=['clip', 'scale_before_gamma'], nargs='+', type=str,
                        help='Type of normalization thats performed on the data',
                        choices=['clip', 'scale_before_gamma', 'scale_after_gamma'])
    parser.add_argument('--resize', default='crop', type=str,
                        help='Whether to resize all images to 256x256 or to crop images',
                        choices=['resize', 'crop', 'none'])   
    parser.add_argument('--size', default=['132', '132'], nargs='+', type=str,
                        help='Shape to which the images need to be cropped. Elements of lists are Strings which are later converted to ints.')            
    

    # Other hyperparameters
    parser.add_argument('--task', default='probs_cross_validation', type=str,
                        choices=['probs_cross_validation', 'features_cross_validation', 'probs', 'features'])
    parser.add_argument('--dataset', default='AUMC3D', type=str,
                        help='What dataset to use for the segmentation',
                        choices=['AUMC2D', 'AUMC3D', 'AUMC3D_version2', 'AUMC3D_version3', 'AUMC3D_fold0', 'AUMC3D_fold1', 'AUMC3D_fold2', 'AUMC3D_fold3', 'AUMC3D_fold4'])
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Minibatch size')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--output_path', default='/home/flieshout/deep_risk_models/fibrosis_segmentation_FvL/outputs/segment_output', type=str,
                        help='Folder in which to store the probabilities/tensors')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()
    print('Saving of tensors has started')
    if args.task == 'features':
        save_features_hdf5(args)
        # save_features(args)
    elif args.task == 'probs':
        save_probs(args)
    elif args.task == 'probs_cross_validation':
        save_probs_cross_validation(args)
    elif args.task == 'features_cross_validation':
        save_features_hdf5_cross_validation(args)
    print('Done')