import torch
import h5py
import os
import csv
from tqdm import tqdm
from scipy.spatial import distance
from torchvision import transforms
import SimpleITK as sitk
import random
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import itertools
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

if torch.cuda.is_available():
    CLASSIFICATION_DIR_NAME = '/home/flieshout/deep_risk_models/data/AUMC_classification_data'
    CLASSIFICATION_DIR_NAME_V2 = '/home/flieshout/deep_risk_models/data/AUMC_classification_data_version2'
    CLASSIFICATION_DIR_NAME_V3 = '/home/flieshout/deep_risk_models/data/AUMC_classification_data_version3'
    CLASSIFICATION_DIR_NAME_CROSS_VAL = '/home/flieshout/deep_risk_models/data/AUMC_data_classification_5_splits'
    EXTRA_LABELS_CSV_PATH = '/home/flieshout/deep_risk_models/data/AUMC_classification_data/gender_lvef_labels.csv'
    ALL_NIFTIS_DIR = '/home/flieshout/deep_risk_models/data/all_classification_data_no_splits'
else:
    CLASSIFICATION_DIR_NAME = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification'
    EXTRA_LABELS_CSV_PATH =  r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\AUMC_data_classification\gender_lvef_labels.csv'
    ALL_NIFTIS_DIR = r'L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\Deep_Risk_Floor\all_classification_data_no_splits'

class GraphPositionDataset(Dataset):
    def __init__(self, probs_path, num_myo, num_fib, split, edges_per_node, node_attributes) -> None:
        super().__init__()
        self.node_attributes=node_attributes
        probs_file = h5py.File(probs_path, 'r')
        split_group = probs_file[split]
        all_LGE_imgs = split_group['LGE_imgs']
        all_myo_probs = split_group['myo_probs']
        all_fib_probs = split_group['fib_probs']
        all_labels_dataset = split_group["labels"]
        all_pat_ids = [np.char.decode(x)[0] for x in split_group["pat_ids"]]

        self.pat_ids = []
        self.labels = []
        self.edge_index = []
        self.positions = []
        self.myo_voxel_indices = []
        self.fib_voxel_indices = []
        self.voxel_greyvalues = []
        self.myo_outputs = [] #use this when you want to get the probabilities of the myocard segmentation model for all voxels (including fibrosis-sampled voxels)
        self.fib_outputs = [] #use this when you want to get the probabilities of the fibrosis segmentation model for all voxels (including myocard_samples voxels)
        self.half_myo_half_fib_outputs = [] #use this if you want to get the probabilities of the myocard segmentation model for the myocard-sampled voxels and the probabilities of the fibrosis segmentation model for the fibrosis-sampled voxels

        self.datafolder = ALL_NIFTIS_DIR

        for row in tqdm(range(all_myo_probs.shape[0])):
            LGE_img = all_LGE_imgs[row].squeeze()
            if LGE_img.shape != all_myo_probs[row].shape or LGE_img.shape != all_fib_probs[row].shape:
                raise ValueError('Expected shapes of LGE_image, myo_probs and fib_probs to be of same size but got {LGE_img.shape}, {all_myo_probs[row].shape}, {all_fib_probs[row].shape}')
            pat_id = all_pat_ids[row]
            self.pat_ids.append(pat_id)
            dicom_image = sitk.ReadImage(os.path.join(self.datafolder, f'{pat_id}_LGE_PSIR.mha'))
            spacing = dicom_image.GetSpacing()
            z_position = np.arange(LGE_img.shape[0]) * spacing[2]
            y_position = np.arange(LGE_img.shape[1]) * spacing[1]
            x_position = np.arange(LGE_img.shape[2]) * spacing[0]

            #sample 500 voxels based on the output of the myocardium model and 500 voxels based on the output of the fibrosis model
            fib_probs = all_fib_probs[row].flatten()
            myo_probs = all_myo_probs[row].flatten()
            num_pixels = fib_probs.shape[0]
            fib_probs /= fib_probs.sum() #normalize probabilities
            sampled_fib_voxels = np.random.choice(num_pixels, size=num_fib, replace=False, p=fib_probs)
            left_over_myo_probs = np.delete(myo_probs, sampled_fib_voxels)
            num_pixels = left_over_myo_probs.shape[0]
            left_over_myo_probs /= left_over_myo_probs.sum() #normalize probabilities
            sampled_myo_voxels = np.random.choice(num_pixels, size=num_myo, replace=False, p=left_over_myo_probs)
            self.myo_voxel_indices.append(sampled_myo_voxels)
            self.fib_voxel_indices.append(sampled_fib_voxels)

            # calculate the positions based on the spacing of the SITK image
            sampled_myo_coordinates = np.unravel_index(sampled_myo_voxels, shape=all_myo_probs[row].shape)
            sampled_fib_coordinates = np.unravel_index(sampled_fib_voxels, shape=all_fib_probs[row].shape)
            sampled_z_position_myo = z_position[sampled_myo_coordinates[0]].reshape(-1,1)
            sampled_y_position_myo = y_position[sampled_myo_coordinates[1]].reshape(-1,1)
            sampled_x_position_myo = x_position[sampled_myo_coordinates[2]].reshape(-1,1)
            sampled_z_position_fib = z_position[sampled_fib_coordinates[0]].reshape(-1,1)
            sampled_y_position_fib = y_position[sampled_fib_coordinates[1]].reshape(-1,1)
            sampled_x_position_fib = x_position[sampled_fib_coordinates[2]].reshape(-1,1)

            myo_positions = np.concatenate((sampled_z_position_myo, sampled_y_position_myo, sampled_x_position_myo), axis=1)
            fib_positions = np.concatenate((sampled_z_position_fib, sampled_y_position_fib, sampled_x_position_fib), axis=1)
            all_positions = np.concatenate((myo_positions, fib_positions), axis=0)
            self.positions.append(all_positions)

            # get the grey values of all the sampled voxels
            myo_grey_values = LGE_img.flatten()[sampled_myo_voxels]
            fib_grey_values = LGE_img.flatten()[sampled_fib_voxels]
            self.voxel_greyvalues.append(np.concatenate((myo_grey_values, fib_grey_values), axis=0))

            # get the outputs of the models
            myo_voxels_myo = myo_probs[sampled_myo_voxels]
            myo_voxels_fib = myo_probs[sampled_fib_voxels]
            self.myo_outputs.append(np.concatenate((myo_voxels_myo, myo_voxels_fib), axis=0))

            fib_voxels_myo = fib_probs[sampled_myo_voxels]
            fib_voxels_fib = fib_probs[sampled_fib_voxels]
            self.fib_outputs.append(np.concatenate((fib_voxels_myo, fib_voxels_fib), axis=0))

            self.half_myo_half_fib_outputs.append(np.concatenate((myo_voxels_myo, fib_voxels_fib), axis=0))

            # connect each node with its k closest neighbours
            D = distance.squareform(distance.pdist(all_positions))
            closest = np.argsort(D, axis=1)
            k_closest = closest[:, 1:edges_per_node+1]

            row_indices = np.arange(k_closest.shape[0]).reshape(-1,1)
            row_indices = np.repeat(row_indices, k_closest.shape[1], axis=1)
            edges_oneway = np.stack((k_closest, row_indices), axis=0).reshape(2,-1)
            edges_otherway = np.stack((row_indices, k_closest), axis=0).reshape(2,-1)
            all_edges = np.concatenate((edges_oneway, edges_otherway), axis=1)
            all_edges_list = all_edges.T.tolist()
            all_edges_list.sort()
            no_duplicates_edges_list = list(k for k,_ in itertools.groupby(all_edges_list))
            edges_array = np.array(no_duplicates_edges_list).T
            self.edge_index.append(edges_array)
            self.labels.append(all_labels_dataset[row])
        total_nr_edges = 0
        total_nr_edges = [s.shape[1] for s in self.edge_index]
        print('Average number of edges per graph:', sum(total_nr_edges)/len(total_nr_edges))

    def __len__(self):
        return len(self.pat_ids)

    def __getitem__(self, index):
        label = self.labels[index]
        pat_id = self.pat_ids[index]
        if self.node_attributes == 'grey_values':
            node_features = self.voxel_greyvalues[index]
        elif self.node_attributes == 'output_myo':
            node_features = self.myo_outputs[index]
        elif self.node_attributes == 'output_fib':
            node_features = self.fib_outputs[index]
        elif self.node_attributes == 'output_myo_fib':
            node_features = self.half_myo_half_fib_outputs[index]
            
        edge_index = self.edge_index[index]
        positions = self.positions[index]
        label = torch.tensor(label, dtype=torch.float32).reshape(1,-1)
        node_features = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        positions = torch.tensor(positions, dtype=torch.float32)
        
        graph = Data(x=node_features, edge_index=edge_index, y=label, pos=positions)
        return graph

    def get_loss_weights(self):
        all_labels = np.stack(self.labels, axis=0)
        if all_labels.shape != (len(self.labels), 4):
            raise ValueError(f'Expected label tensor to be of size ({len(self.labels)}, 4) but got {all_labels.shape}')
        num_positive = np.sum(all_labels, axis=0)
        num_negative = np.repeat(np.array([[all_labels.shape[0]]]), all_labels.shape[1], axis=1) - num_positive
        weights = num_negative / num_positive
        return weights

class GraphDistanceDataset(Dataset):
    def __init__(self, probs_path, myo_features_path, fib_features_path, num_myo, num_fib, split, edges_per_node, node_attributes, distance_measure) -> None:
        super().__init__()
        subset = False
        self.positive_subset = False
        self.label_name = None
        self.split = split
        if split == 'train':
            subset_value = 20
        elif split == 'validation':
            subset_value = 5
        else:
            subset_value = 10
        self.node_attributes=node_attributes
        probs_file = h5py.File(probs_path, 'r')
        split_group = probs_file[split]
        all_LGE_imgs = split_group['LGE_imgs']
        all_myo_probs = split_group['myo_probs']
        all_fib_probs = split_group['fib_probs']
        all_labels_dataset = split_group["labels"]
        all_pat_ids = [np.char.decode(x)[0] for x in split_group["pat_ids"]]

        if subset:
            print(all_LGE_imgs.shape, all_myo_probs.shape, all_fib_probs.shape)
            all_LGE_imgs = all_LGE_imgs[:subset_value]
            all_myo_probs = all_myo_probs[:subset_value]
            all_fib_probs = all_fib_probs[:subset_value]
            all_labels_dataset = all_labels_dataset[:subset_value]
            all_pat_ids = all_pat_ids[:subset_value]

        if node_attributes in ['features_myo', 'features_myo_fib']:
            myo_features_file = h5py.File(myo_features_path, 'r')
            myo_feat_split_group = myo_features_file[split]
            myo_patids = [np.char.decode(x)[0] for x in myo_feat_split_group["pat_ids"]]
        if node_attributes in ['features_fib', 'features_myo_fib']:
            fib_features_file = h5py.File(fib_features_path, 'r')
            fib_feat_split_group = fib_features_file[split]
            fib_patids = [np.char.decode(x)[0] for x in fib_feat_split_group["pat_ids"]]

        self.pat_ids = []
        self.labels = []
        self.edge_index = []
        self.edge_attributes = []
        self.myo_voxel_indices = []
        self.fib_voxel_indices = []
        self.node_features = []
        # self.voxel_greyvalues = []
        # self.myo_outputs = [] #use this when you want to get the probabilities of the myocard segmentation model for all voxels (including fibrosis-sampled voxels)
        # self.fib_outputs = [] #use this when you want to get the probabilities of the fibrosis segmentation model for all voxels (including myocard_samples voxels)

        self.datafolder = ALL_NIFTIS_DIR

        print('all probs shape:', all_myo_probs.shape[0])

        for row in tqdm(range(all_myo_probs.shape[0])):
            LGE_img = all_LGE_imgs[row].squeeze()
            if LGE_img.shape != all_myo_probs[row].shape or LGE_img.shape != all_fib_probs[row].shape:
                raise ValueError(f'Expected shapes of LGE_image, myo_probs and fib_probs to be of same size but got {LGE_img.shape}, {all_myo_probs[row].shape}, {all_fib_probs[row].shape}')
            pat_id = all_pat_ids[row]
            self.pat_ids.append(pat_id)
            dicom_image = sitk.ReadImage(os.path.join(self.datafolder, f'{pat_id}_LGE_PSIR.mha'))
            spacing = dicom_image.GetSpacing()
            z_position = np.arange(LGE_img.shape[0]) * spacing[2]
            y_position = np.arange(LGE_img.shape[1]) * spacing[1]
            x_position = np.arange(LGE_img.shape[2]) * spacing[0]

            #sample 500 voxels based on the output of the myocardium model and 500 voxels based on the output of the fibrosis model
            fib_probs = all_fib_probs[row].flatten()
            myo_probs = all_myo_probs[row].flatten()
            if num_fib > -1:
                num_pixels = fib_probs.shape[0]
                fib_probs /= fib_probs.sum() #normalize probabilities
                sampled_fib_voxels = np.random.choice(num_pixels, size=num_fib, replace=False, p=fib_probs)
            else:
                print('Using all voxels with probs > 0.5')
                sampled_fib_voxels = (fib_probs >= 0.5).nonzero()[0]
                num_fib = sampled_fib_voxels.shape[0]
            left_over_myo_probs = myo_probs.copy()
            left_over_myo_probs[sampled_fib_voxels] = 0.0
            if num_myo > -1:
                num_pixels = left_over_myo_probs.shape[0]
                left_over_myo_probs /= left_over_myo_probs.sum() #normalize probabilities
                sampled_myo_voxels = np.random.choice(num_pixels, size=num_myo, replace=False, p=left_over_myo_probs)
            else:
                sampled_myo_voxels = (left_over_myo_probs >= 0.5).nonzero()[0]
                num_myo = sampled_myo_voxels.shape[0]
            self.myo_voxel_indices.append(sampled_myo_voxels)
            self.fib_voxel_indices.append(sampled_fib_voxels)

            # calculate the positions based on the spacing of the SITK image
            sampled_myo_coordinates = np.unravel_index(sampled_myo_voxels, shape=all_myo_probs[row].shape)
            sampled_fib_coordinates = np.unravel_index(sampled_fib_voxels, shape=all_fib_probs[row].shape)
            sampled_z_position_myo = z_position[sampled_myo_coordinates[0]].reshape(-1,1)
            sampled_y_position_myo = y_position[sampled_myo_coordinates[1]].reshape(-1,1)
            sampled_x_position_myo = x_position[sampled_myo_coordinates[2]].reshape(-1,1)
            sampled_z_position_fib = z_position[sampled_fib_coordinates[0]].reshape(-1,1)
            sampled_y_position_fib = y_position[sampled_fib_coordinates[1]].reshape(-1,1)
            sampled_x_position_fib = x_position[sampled_fib_coordinates[2]].reshape(-1,1)

            myo_positions = np.concatenate((sampled_z_position_myo, sampled_y_position_myo, sampled_x_position_myo), axis=1)
            fib_positions = np.concatenate((sampled_z_position_fib, sampled_y_position_fib, sampled_x_position_fib), axis=1)
            all_positions = np.concatenate((myo_positions, fib_positions), axis=0)

            if self.node_attributes == 'grey_values':
                # get the grey values of all the sampled voxels
                myo_grey_values = LGE_img.flatten()[sampled_myo_voxels].reshape(-1,1)
                fib_grey_values = LGE_img.flatten()[sampled_fib_voxels].reshape(-1,1)
                self.node_features.append(np.concatenate((myo_grey_values, fib_grey_values), axis=0))
            elif self.node_attributes == 'probs_myo':
                # get the output probabilities of the models
                myo_probs_myo = myo_probs[sampled_myo_voxels]
                myo_probs_fib = myo_probs[sampled_fib_voxels]
                self.node_features.append(np.concatenate((myo_probs_myo, myo_probs_fib), axis=0))
            elif self.node_attributes == 'probs_fib':
                # get the output probabilities of the models
                fib_probs_myo = fib_probs[sampled_myo_voxels]
                fib_probs_fib = fib_probs[sampled_fib_voxels]
                self.node_features.append(np.concatenate((fib_probs_myo, fib_probs_fib), axis=0))
            elif self.node_attributes == 'features_myo':
                # get the ouput features of the myocardium model
                myo_index = myo_patids.index(pat_id)
                myo_features_myo = myo_feat_split_group['myocardium_features'][myo_index].reshape(64,-1)[:,sampled_myo_voxels]
                myo_features_fib = myo_feat_split_group['myocardium_features'][myo_index].reshape(64,-1)[:,sampled_fib_voxels]
                all_myo_features = np.concatenate((myo_features_myo, myo_features_fib), axis=1).T
                self.node_features.append(all_myo_features)
            elif self.node_attributes == 'features_fib':
                # get the ouput features of the fibrosis model
                fib_index = fib_patids.index(pat_id)
                fib_features_myo = fib_feat_split_group['fibrosis_features'][fib_index].reshape(32,-1)[:,sampled_myo_voxels]
                fib_features_fib = fib_feat_split_group['fibrosis_features'][fib_index].reshape(32,-1)[:,sampled_fib_voxels]
                all_fib_featuers = np.concatenate((fib_features_myo, fib_features_fib), axis=1).T
                self.node_features.append(all_fib_featuers)
            elif self.node_attributes == 'features_myo_fib':
                # get the ouput features of both the myocardium and the fibrosis model
                myo_index = myo_patids.index(pat_id)
                fib_index = fib_patids.index(pat_id)
                myo_features = myo_feat_split_group['myocardium_features'][myo_index].reshape(64,-1)
                myo_features_myo = myo_features[:,sampled_myo_voxels]
                myo_features_fib = myo_features[:,sampled_fib_voxels]
                fib_features = fib_feat_split_group['fibrosis_features'][fib_index].reshape(32,-1)
                fib_features_myo = fib_features[:,sampled_myo_voxels]
                fib_features_fib = fib_features[:,sampled_fib_voxels]
                myo_fib_features_myo = np.concatenate((myo_features_myo, fib_features_myo), axis=0)
                myo_fib_features_fib = np.concatenate((myo_features_fib, fib_features_fib), axis=0)
                myo_fib_features_total = np.concatenate((myo_fib_features_myo, myo_fib_features_fib), axis=1).T
                self.node_features.append(myo_fib_features_total)

            # connect each node with its k closest neighbours and calculate the edge attributes
            D = distance.squareform(distance.pdist(all_positions))
            closest = np.argsort(D, axis=1)
            k_closest = closest[:, 1:edges_per_node+1]
            row_indices = np.arange(k_closest.shape[0]).reshape(-1,1)
            row_indices = np.repeat(row_indices, k_closest.shape[1], axis=1)
            edges_oneway = np.stack((k_closest, row_indices), axis=0).reshape(2,-1)
            edges_otherway = np.stack((row_indices, k_closest), axis=0).reshape(2,-1)
            all_edges = np.concatenate((edges_oneway, edges_otherway), axis=1)
            all_edges_list = all_edges.T.tolist()

            if distance_measure != 'none':
                if distance_measure == 'euclidean':
                    closest_distances = np.take_along_axis(D,k_closest,1)
                    flattened_closest_dist = closest_distances.flatten().reshape(-1,1)
                elif distance_measure == 'relative_position':
                    all_positions_matrix = np.stack([all_positions] * all_positions.shape[0], axis=0)
                    relative_positions = np.abs(all_positions_matrix - np.swapaxes(all_positions_matrix, 0, 1))
                    closest_positions = np.take_along_axis(relative_positions, np.stack([k_closest] * 3, axis=-1),1)
                    flattened_closest_dist = closest_positions.reshape(-1,3)
                else:
                    raise ValueError(f'Distance measure {distance_measure} not recognized')

                all_distances = np.tile(flattened_closest_dist,(2,1)).squeeze()
                all_distances_list = all_distances.tolist()
                all_edges_list_sorted, all_closts_dist_sorted = zip(*sorted(zip(all_edges_list, all_distances_list)))
                no_duplicates_edges_list = []
                no_duplicates_distances_list = []
                distance_index = 0
                for k, g in itertools.groupby(all_edges_list_sorted):
                    no_duplicates_edges_list.append(k)
                    no_duplicates_distances_list.append(all_closts_dist_sorted[distance_index])
                    distance_index += len(list(g))
                edges_array = np.array(no_duplicates_edges_list).T
                pos_array = np.array(no_duplicates_distances_list).reshape(len(no_duplicates_distances_list),-1)
                self.edge_index.append(edges_array)
                self.edge_attributes.append(pos_array)
            else:
                all_edges_list_sorted = sorted(all_edges_list)
                no_duplicates_edges_list = [k for k,_ in itertools.groupby(all_edges_list_sorted)]
                edges_array = np.array(no_duplicates_edges_list).T
                self.edge_index.append(edges_array)
                self.edge_attributes.append(None)

            self.labels.append(all_labels_dataset[row])

        total_nr_edges = 0
        total_nr_edges = [s.shape[1] for s in self.edge_index]
        print('Average number of edges per graph:', sum(total_nr_edges)/len(total_nr_edges))

    def __len__(self):
        return len(self.pat_ids)

    def __getitem__(self, index):
        label = self.labels[index]
        pat_id = self.pat_ids[index]
        node_features = self.node_features[index]
            
        edge_index = self.edge_index[index]
        edge_attr = self.edge_attributes[index]
        label = torch.tensor(label, dtype=torch.float32).reshape(1,-1)
        node_features = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.int64)

        if edge_attr is not None:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=label)
        return graph

    def add_extra_labels(self, label_name, datafolder):
        csv_path = os.path.join(datafolder, 'gender_lvef_labels.csv')
        # get gender and LVEF for every patient and append them to the labels
        with open(csv_path, mode='r') as inp:
            reader = csv.reader(inp, delimiter=';')
            dict_from_csv = {rows[0]:[rows[1], rows[2], rows[3]] for rows in reader} #patid: (gender, LVEF, LVEF_category)
        if label_name=='LVEF':
            #filter out patients that don't have a LVEF value
            new_pat_ids, new_labels, indices_to_keep = [], [], []
            for i, (pat_id, labels) in enumerate(zip(self.pat_ids, self.labels)):
                LVEF_value = dict_from_csv[pat_id][1]
                if LVEF_value != '' and LVEF_value != '-99.0':
                    indices_to_keep.append(i)
                    new_pat_ids.append(pat_id)
                    label = np.append(labels, float(LVEF_value))
                    new_labels.append(label)
            node_features_to_keep = [self.node_features[i] for i in indices_to_keep]
            edge_index_to_keep = [self.edge_index[i] for i in indices_to_keep]
            edge_attributes_to_keep = [self.edge_attributes[i] for i in indices_to_keep]
            self.pat_ids = new_pat_ids
            self.labels = new_labels
            self.node_features = node_features_to_keep
            self.edge_index = edge_index_to_keep
            self.edge_attributes = edge_attributes_to_keep
        elif label_name=='gender':
            for i, (pat_id, labels) in enumerate(zip(self.pat_ids, self.labels)):
                gender_value = dict_from_csv[pat_id][0]
                new_label = np.append(labels, float(gender_value))
                self.labels[i] = new_label
        elif label_name=='LVEF_category':
            new_pat_ids, new_labels, indices_to_keep = [], [], []
            for i, (pat_id, labels) in enumerate(zip(self.pat_ids, self.labels)):
                LVEF_category = dict_from_csv[pat_id][2]
                if LVEF_category != '' and LVEF_category != '-99.0':
                    indices_to_keep.append(i)
                    new_pat_ids.append(pat_id)
                    label = np.append(labels, float(LVEF_category))
                    new_labels.append(label)
            node_features_to_keep = [self.node_features[i] for i in indices_to_keep]
            edge_index_to_keep = [self.edge_index[i] for i in indices_to_keep]
            edge_attributes_to_keep = [self.edge_attributes[i] for i in indices_to_keep]
            self.pat_ids = new_pat_ids
            self.labels = new_labels
            self.node_features = node_features_to_keep
            self.edge_index = edge_index_to_keep
            self.edge_attributes = edge_attributes_to_keep
        else:
            raise ValueError(f'Extra label {label_name} not valid')
        self.label_name = label_name
        if self.positive_subset:
            new_indices_to_keep = [i for i in range(len(self.labels)) if self.labels[i][-1] in [0.0,1.0,2.0]]
            self.pat_ids  = [self.pat_ids[i] for i in new_indices_to_keep]
            self.labels  = [self.labels[i] for i in new_indices_to_keep]
            self.node_features  = [self.node_features[i] for i in new_indices_to_keep]
            self.edge_index  = [self.edge_index[i] for i in new_indices_to_keep]
            self.edge_attributes  = [self.edge_attributes[i] for i in new_indices_to_keep]

    def get_loss_weights(self):
        all_labels = np.stack(self.labels, axis=0)
        if all_labels.shape[0] != len(self.labels) or all_labels.shape[1] < 4:
            raise ValueError(f'Expected label tensor to be of size ({len(self.labels)}, >=4) but got {all_labels.shape}')

        all_weights = []
        #calculate weights for therapy and mortality labels
        for i in range(all_labels.shape[1]-1):
            task_labels = all_labels[:,i]
            num_positive = np.sum(task_labels)
            num_negative = len(task_labels) - num_positive
            pos_weight = num_negative/num_positive
            all_weights.append(pos_weight)

        #calculate weights for extra labels
        if self.label_name == 'gender':
            num_positive = np.sum(task_labels)
            num_negative = len(task_labels) - num_positive
            pos_weight = num_negative/num_positive
            all_weights.append(pos_weight)
        elif self.label_name == 'LVEF_category':
            task_labels = all_labels[:,4]
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(task_labels), y=task_labels).tolist()
            print('class weight sklearn:', class_weights)
            print('num_zeros:', np.count_nonzero(task_labels == 0))
            print('num_ones:', np.count_nonzero(task_labels == 1))
            print('num_twos:', np.count_nonzero(task_labels == 2))
            print('num_threes:', np.count_nonzero(task_labels == 3))
            all_weights = all_weights + class_weights
        all_weights = np.array(all_weights).reshape(1,-1)
        return all_weights

def load_graph_data(probs_path, myo_feat_path, fib_feat_path, model_name, num_myo, num_fib, edges_per_node, node_attributes, batch_size=8, val_batch_size='same', num_workers=1, only_test=False, distance_measure='euclidian', extra_label=None, cross_validation=False):
    if only_test:
        if model_name.split('_')[0] == 'GNN':
            val_dataset = GraphDistanceDataset(probs_path, myo_feat_path, fib_feat_path, num_myo, num_fib, split='validation', edges_per_node=edges_per_node, node_attributes=node_attributes, distance_measure=distance_measure)
            test_dataset = GraphDistanceDataset(probs_path, myo_feat_path, fib_feat_path, num_myo, num_fib, split='test', edges_per_node=edges_per_node, node_attributes=node_attributes, distance_measure=distance_measure)
            if extra_label is not None:
                val_dataset.add_extra_labels(extra_label, val_dataset.datafolder)
                test_dataset.add_extra_labels(extra_label, test_dataset.datafolder)
        else:
            val_dataset = GraphPositionDataset(probs_path, num_myo, num_fib, split='validation', edges_per_node=edges_per_node, node_attributes=node_attributes)
            test_dataset = GraphPositionDataset(probs_path, num_myo, num_fib, split='test', edges_per_node=edges_per_node, node_attributes=node_attributes)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return val_loader, test_loader
    else:
        if model_name.split('_')[0] == 'GNN':
            train_dataset = GraphDistanceDataset(probs_path, myo_feat_path, fib_feat_path, num_myo, num_fib, split='train', edges_per_node=edges_per_node, node_attributes=node_attributes, distance_measure=distance_measure)
            val_dataset = GraphDistanceDataset(probs_path, myo_feat_path, fib_feat_path, num_myo, num_fib, split='validation', edges_per_node=edges_per_node, node_attributes=node_attributes, distance_measure=distance_measure)
            if not cross_validation:
                test_dataset = GraphDistanceDataset(probs_path, myo_feat_path, fib_feat_path, num_myo, num_fib, split='test', edges_per_node=edges_per_node, node_attributes=node_attributes, distance_measure=distance_measure)
            if extra_label is not None:
                train_dataset.add_extra_labels(extra_label, train_dataset.datafolder)
                val_dataset.add_extra_labels(extra_label, val_dataset.datafolder)
                if not cross_validation:
                    test_dataset.add_extra_labels(extra_label, test_dataset.datafolder)
        else:
            train_dataset = GraphPositionDataset(probs_path, num_myo, num_fib, split='train', edges_per_node=edges_per_node, node_attributes=node_attributes)
            val_dataset = GraphPositionDataset(probs_path, num_myo, num_fib, split='validation', edges_per_node=edges_per_node, node_attributes=node_attributes)
            test_dataset = GraphPositionDataset(probs_path, num_myo, num_fib, split='test', edges_per_node=edges_per_node, node_attributes=node_attributes)
        train_loss_weights = train_dataset.get_loss_weights()
        val_loss_weights = val_dataset.get_loss_weights()
        print('train loss weights:', train_loss_weights)
        print('validation loss weights:', val_loss_weights)
        if val_batch_size == 'full_set':
            val_batch_size = len(val_dataset)
        elif val_batch_size == 'same':
            val_batch_size = batch_size
        else:
            raise ValueError()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
        if cross_validation:
            return train_loader, val_loader, None, (train_loss_weights, val_loss_weights)
        else:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            return train_loader, val_loader, test_loader, (train_loss_weights, val_loss_weights)
            

            
