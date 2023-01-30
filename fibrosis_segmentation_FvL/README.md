# Code for cardiac fibrosis segmentation on MRI images and prediction of appropriate device therapy of implantable cardioverter defibrillators
This code is part of my Master Thesis for the Artificial Intelligence Master's at the University of Amsterdam: https://scripties.uba.uva.nl/search?id=record_49746
The work was also presented at the Computing in Cardiology conference 2022 in Tampere, Finland and rewarded with the Bill & Gary Sanders award for best poster. Publication of the accompanying article is still in progress.


This folder provides code for two main purposes:
1. Utilize a U-net structure to automatically segment scar tissue on Cardiac MRI images (SMR). This code can also be used to train a U-net model to segment other (medical) images. However, the code is only tested in our internal dataset of pseudo 3D cardiac MRI images in .mha format and their binary image labels in .nrrd format
2. Train different Deep Learning model types for the classification appropriate device therapy (ADT) from their implantable cardioverter defibrillator (ICD) in the complete follow-up period or within one year after implantation. The input of these classification models relies on the output of the fibrosis segmentation models.

## Part 1: fibrosis segmentation
This part of the codes can train different U-net model configurations for the task of 1) myocardium segmentation and 2) fibrosis segmentation.
There are many model configurations that are implementated and were tested with this code but the final configurations that were used in the thesis include two steps:
1. Segmentation of the myocardium. As the segmentation of the myocardium on CMR images is proven to be a relatively easy task, a (regular) 2D U-net is used to segment the complete myocardium on every 2D CMR slice.
2. Segmentation of the fibrosis. For the segmentation of the fibrosis, either a 2D, 2.5D or 3D U-net can be used. All configurations use a stacked version of the original image and the segmented myocardium as its input. The 2D configuration uses solely 2D convolutions and treats all CMR slices independently. The 3D configuration treats every patient/CMR study as a whole and uses solely 3D convolutions. The 2.5D configuration treats every CMR study as a whole (just like the 3D configuration) but uses a combination of 2D and 3D convolutions.

### How to run the code?
1. Data prerequisites:
- The original CMR studies should be saved in .mha format and the binary myocardium and fibrosis labels should be saved in .nrrd format. The original study images and the binary labels should be stored in seperate folders in the following structure:
```bash
└── files_dir
    ├── train
    │   ├── LGE_niftis
    │   │   ├── [pat_id1]_LGE_PSIR.mha
    │   │   ├── [pat_id2]_LGE_PSIR.mha
    │   │   └── etc.
    │   ├── myo
    │   │   ├── [pat_id1]_myo_mask.nrrd
    │   │   ├── [pat_id2]_myo_mask.nrrd
    │   │   └── etc.
    │   └── aankleuring
    │       ├── [pat_id1]_aankleuring_mask.nrrd
    │       ├── [pat_id2]_aankleuring_mask.nrrd
    │       └── etc.
    ├── validation
    │   ├── LGE_niftis
    │   │   ├── [pat_id1]_LGE_PSIR.mha
    │   │   ├── [pat_id2]_LGE_PSIR.mha
    │   │   └── etc.
    │   ├── myo
    │   │   ├── [pat_id1]_myo_mask.nrrd
    │   │   ├── [pat_id2]_myo_mask.nrrd
    │   │   └── etc.
    │   └── aankleuring
    │       ├── [pat_id1]_aankleuring_mask.nrrd
    │       ├── [pat_id2]_aankleuring_mask.nrrd
    │       └── etc.
    └── test
        ├── LGE_niftis
        │   ├── [pat_id1]_LGE_PSIR.mha
        │   ├── [pat_id2]_LGE_PSIR.mha
        │   └── etc.
        ├── myo
        │   ├── [pat_id1]_myo_mask.nrrd
        │   ├── [pat_id2]_myo_mask.nrrd
        │   └── etc.
        └── aankleuring
            ├── [pat_id1]_aankleuring_mask.nrrd
            ├── [pat_id2]_aankleuring_mask.nrrd
            └── etc.
```
- In import_AUMC_dataset.py set ORIGINAL_DIR_NAME (line 15 if working on machine with GPU available and line 34 if working on machine with CPU only) to the path of [files_dir] as stated in the folder structure above
2. Train the 2D U-net model for segmentation of the complete myocardium using the following command. Pytorch Lightning will save the best model (based on the validation set) in `[logging_directory]/myocard/lightning_logs/version_x`
```shell
python train_myocard_segmentation_pl.py --resize crop --size 132 132 --normalize clip scale_before_gamma --loss_function adaptive_weighted_dice --lr 1e-4 --epochs 200 --model Floor_UNet2D --upsampling convtrans --log_dir [logging_directory]
```
3. Evaluate the trained myocardium model on the test set using the following command.
```shell
python test_segmentation_pl.py --test_train_or_inference test --segment_task myocard --resize crop --size 132 132 --model_path [myocardium_model_checkpoint_path] --output_path [output_folder]
```
 The test results can be found in the file `[output_folder]\AUMC2D\myocard\version_[model_version_nr]\test_segmentation_version_[model_version_nr].txt`. The produced binary masks can be found in `[output_folder]\AUMC2D\myocard\version_[model_version_nr]\test`
 4. Train a 2D, 2.5D or 3D U-net model for the segmentation of the fibrosis using the following command. Pytorch Lightning will save the best model (based on the validation set) in `[logging_directory]/fibrosis/lightning_logs/version_x`\n
**2D model**:
 ```shell
python train_fibrosis_segmentation_with_myo_pl.py --resize crop --size 132 132 --loss_function dice --lr 1e-3 --epochs 200 --batch_size 8 --model Floor_UNet2D_stacked --upsampling convtrans --dataset AUMC2D --feature_multiplication 2 --normalize clip scale_before_gamma --transformations hflip vflip rotate --myo_checkpoint [myocardium_model_checkpoint_path]
 ```
 **2.5D model**
  ```shell
python train_fibrosis_segmentation_with_myo_pl.py --resize crop --size 132 132 --loss_function dice --lr 1e-3 --epochs 200 --batch_size 8 --model UNet3D_half_stacked --upsampling convtrans --dataset AUMC3D --feature_multiplication 2 --normalize clip scale_before_gamma --transformations hflip vflip rotate --myo_checkpoint [myocardium_model_checkpoint_path]
 ```
 **3D model**
   ```shell
train_fibrosis_segmentation_with_myo_pl.py --resize crop --size 132 132 --loss_function dice --lr 1e-3 --epochs 200 --batch_size 6 --model UNet3D_full_stacked --upsampling convtrans --dataset AUMC3D --feature_multiplication 4 --normalize clip scale_before_gamma --transformations hflip vflip rotate --myo_checkpoint [myocardium_model_checkpoint_path]
 ```
 5. Evaluate the trained fibrosis model on the test set using the following command:
 ```shell
 python test_segmentation_pl.py --test_train_or_inference test --segment_task myofib --resize crop --size 132 132 --normalize clip scale_before_gamma --dataset [AUMC2D/AUMC3D] --upsampling convtrans --model_path [fibrosis_model_checkpoint_path] --output_path [output_folder]
 ```
 The test results can be found in the file `[output_folder]\[AUMC2D or AUMC3D]\myofib\version_[model_version_nr]\test_segmentation_version_[model_version_nr].txt`. The produced binary masks can be found in `[output_folder]\[AUMC2D or AUMC3D]\myofib\version_[model_version_nr]\test`
 
 ## Part 2: ICD therapy prediction
 This part of the code can utelize the outputs of the segmentation models to train several classification models for the outcome of ICD therapy prediction (either in the complete follow-up period or withint one year after implantation). The models are trained using five-fold cross validation. Therefore, the data needs to be subdivided into five different folds (and corresponding folders).
 1. Data prerequisites:
 - The original CMR studies should be saved in .mha format and the labels per patient_ID should be saved as a csv file named 'ICD_therapy_labels.csv'. The files should be stored in the following structure:
```bash
└── class_files_dir
    ├── fold0
    │   ├── train
    │   │   ├── [pat_id1]_LGE_PSIR.mha
    │   │   └── etc.
    │   ├── validation
    │   │   ├── [pat_id2]_LGE_PSIR.mha
    │   │   └── etc.
    │   └── test
    │       ├── [pat_id3]_LGE_PSIR.mha
    │       └── etc.
    ├── fold1
    │   ├── train
    │   │   ├── [pat_id2]_LGE_PSIR.mha
    │   │   └── etc.
    │   ├── validation
    │   │   ├── [pat_id3]_LGE_PSIR.mha
    │   │   └── etc.
    │   └── test
    │       ├── [pat_id4]_LGE_PSIR.mha
    │       └── etc.
    ├── fold2
    │   ├── train
    │   │   ├── [pat_id3]_LGE_PSIR.mha
    │   │   └── etc.
    │   ├── validation
    │   │   ├── [pat_id4]_LGE_PSIR.mha
    │   │   └── etc.
    │   └── test
    │       ├── [pat_id5]_LGE_PSIR.mha
    │       └── etc.
    ├── fold3
    │   ├── train
    │   │   ├── [pat_id4]_LGE_PSIR.mha
    │   │   └── etc.
    │   ├── validation
    │   │   ├── [pat_id5]_LGE_PSIR.mha
    │   │   └── etc.
    │   └── test
    │       ├── [pat_id6]_LGE_PSIR.mha
    │       └── etc.
    ├── fold4
    │   ├── train
    │   │   ├── [pat_id5]_LGE_PSIR.mha
    │   │   └── etc.
    │   ├── validation
    │   │   ├── [pat_id6]_LGE_PSIR.mha
    │   │   └── etc.
    │   └── test
    │       ├── [pat_id7]_LGE_PSIR.mha
    │       └── etc.
    ├── ICD_therapy_labels.csv
    └── clinical_features.csv (optional)
 ```
- The ICD_therapy_labels.csv file should consist of five columns in the following order:
 a. SubjectID: denotes the patient/subject id that corresponds to the [pat_id1] in the file names of the CMR studies.
 b. ICD_therapy: binary label (0/1) indicating whether the patient has received ICD therapy in the complete follow-up period
 c. ICD_therapy_365days: binary label (0/1) indicating whether the patient has received ICD therapy within one year after implantation
 d. Mortality: binary label (0/1) indicating whether the patient died in the complete follow-up period
 e. Mortality_365days: binary label (0/1) indicating whether the patient died within one year after implantation

2. We will first use the pre-trained segmentation models to create segmentations of the data that we wish to use for the classification/prediction of ICD therapy and we store these segmentations as hdf5 tensors. We first have to set the correct the file locations to the fold folders within `class_files_dir` in the structure above. To do so, change the lines 23 until 27 in `data_loading/import_AUMC_dataset.py` if a GPU is available on your machine. Else, change lines 42 until 46 in `data_loading/import_AUMC_dataset.py`. Then, create the hdf5 tensors by running the following commands (from within the `utils_functions` folder of this repository) for every fold (AUMC3D_fold0 up until AUMC3D_fold4):
 ```shell
 python save_segmentation_tensors.py --task probs_cross_validation --model_path [myocardium_model_checkpoint_path] --output_path [output_folder] --dataset AUMC3D_fold0
 ```
 ```shell
 python save_segmentation_tensors.py --task probs_cross_validation --model_path [fibrosis_model_checkpoint_path] --output_path [output_folder] --dataset AUMC3D_fold0
 ```
 ```shell
 python save_segmentation_tensors.py --task features_cross_validation --model_path [myocardium_model_checkpoint_path] --output_path [output_folder] --dataset AUMC3D_fold0
 ```
 ```shell
 python save_segmentation_tensors.py --task features_cross_validation --model_path [fibrosis_model_checkpoint_path] --output_path [output_folder] --dataset AUMC3D_fold0
 ```
These commands save the necessary tensors in hdf5 format in subfolders inside [output_folder]

3. Train the different models using the following commands (for every fold):
- *CNN:*
 ```shell
python train_cross_validation.py --model CNN --use_MRI_features False --hidden_layers_conv 16 16 --hidden_layers_lin 64 128 128 --extra_input LGE_image --batch_size 6 --fib_checkpoint [fibrosis_model_checkpoint_path] --prediction_task ICD_therapy --fold 0
 ```
- *CNN_clinical:*
 ```shell
python train_cross_validation.py --model CNN --use_MRI_features True --hidden_layers_conv 16 32 --hidden_layers_lin 32 64 128 --extra_input LGE_image --batch_size 6 --fib_checkpoint [fibrosis_model_checkpoint_path] --prediction_task ICD_therapy --fold 0
 ```
- *MLP:*
 ```shell
python train_cross_validation.py --model encoder --use_MRI_features False --hidden_layers_lin 128 64 32 --flatten_or_maxpool flatten --batch_size 16 --fib_checkpoint [fibrosis_model_checkpoint_path] --prediction_task ICD_therapy --fold 0
 ```
- *MLP_clinical:*
 ```shell
python train_cross_validation.py --model encoder --use_MRI_features True --hidden_layers_lin 128 64 32 --flatten_or_maxpool flatten --batch_size 16 --fib_checkpoint [fibrosis_model_checkpoint_path] --prediction_task ICD_therapy --new_dataset True --fold 0
 ```
- *DenseNet:*
 ```shell
python train_cross_validation.py --model densenet_padding --use_MRI_features True --batch_size 8 --myo_checkpoint [myocardium_model_checkpoint_path] --prediction_task ICD_therapy --fold 0
 ```
- *MMM:*
 ```shell
python train_cross_validation.py --model multi_input --use_MRI_features True --batch_size 8 --fib_checkpoint [fibrosis_model_checkpoint_path] --densenet_checkpoint [densenet_model_checkpoint_path] --prediction_task ICD_therapy --fold 0
 ```
- *grey-none GNN:*
 ```shell
python train_cross_validation.py --model GNN --use_MRI_features False --node_attributes grey_values --distance_measure none --dist_info False --num_myo 500 --num_fib 100 --edges_per_node 25 --hidden_channels_gcn 64 --num_gcn_layers 5 --dropout 0.1 --batch_size 8 --probs_path [hdf5_output_folder]/segmentation_probs/version_xx_AUMC3D_fold0/deeprisk_myocard_fibrosis_probabilities_AUMC3D_fold0_n=419.hdf5 --myo_feat_path [hdf5_output_folder]/segmentation_tensors_hdf5/myocardium_version_xx_AUMC3D_fold0/deeprisk_myocardium_features_AUMC3D_fold0_n=419.hdf5 --fib_feat_path [hdf5_output_folder]/segmentation_tensors_hdf5/fibrosis_version_xx_AUMC3D_fold0/deeprisk_fibrosis_features_AUMC3D_fold0_n=419.hdf5 --ICD_therapy --epochs 100 --lr 1e-3 --fold 0
 ```
- *grey-none GNN* (and the other GNN's by changing --node_attributes and --distance_measure):
 ```shell
python train_cross_validation.py --model GNN --use_MRI_features False --node_attributes grey_values --distance_measure none --dist_info False --num_myo 500 --num_fib 100 --edges_per_node 25 --hidden_channels_gcn 64 --num_gcn_layers 5 --dropout 0.1 --batch_size 8 --probs_path [hdf5_output_folder]/segmentation_probs/version_xx_AUMC3D_fold0/deeprisk_myocard_fibrosis_probabilities_AUMC3D_fold0_n=419.hdf5 --myo_feat_path [hdf5_output_folder]/segmentation_tensors_hdf5/myocardium_version_xx_AUMC3D_fold0/deeprisk_myocardium_features_AUMC3D_fold0_n=419.hdf5 --fib_feat_path [hdf5_output_folder]/segmentation_tensors_hdf5/fibrosis_version_xx_AUMC3D_fold0/deeprisk_fibrosis_features_AUMC3D_fold0_n=419.hdf5 --ICD_therapy --epochs 100 --lr 1e-3 --fold 0
 ```
 
 4. Test the models using the following command:
 - *GNN models:*
 -  ```shell
python test_cross_validation.py --folds 0 1 2 3 4 --batch_size 8 --model_folder [folder_with_trained_model_per_fold] --num_myo 500 --num_fib 100 --probs_prefix [hdf5_output_folder]/segmentation_probs/version_xx_AUMC3D --myo_feat_folder_prefix [hdf5_output_folder]/segmentation_tensors_hdf5/myocardium_version_xx_AUMC3D --fib_feat_folder_prefix [hdf5_output_folder]/segmentation_tensors_hdf5/fibrosis_version_xx_AUMC3D --model_version 0 --which_model AUC --output_path [output_folder]
 ```
 - *all other models:*
 ```shell
python test_cross_validation.py --folds 0 1 2 3 4 --batch_size 16 --model_folder [folder_with_trained_model_per_fold] --model_version 0 --which_model AUC --output_path [output_folder]
 ```
 The test results can be found in [output_folder]
