# Weakly-supervised fibrosis segmentation 

*Author: Roel Klein*

A pipeline for **weak-supervision** fibrosis segmentation in MRI, using either **slice-level** or **stack-level** binary fibrosis labels.

Also includes options for training **fully-supervised** segmentation, both for **myocardium** and **fibrosis**.

## 0. Install instructions
* Clone this repository
* Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
* `conda env create --file deeprisk_weak.yml` from cmd/terminal at cloned repository to create the conda environment.
* `conda activate deeprisk_weak` in terminal to activate the environment.
*  On the L drive, all trained models and data can be found in `L:\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\deeprisk_weakly_supervised_data_and_models.zip`. Unpack the parts that you need in your cloned repository. Fibrosis pseudolabels are in the classification model log folders. Similarly, fibrosis segmentation predictions are in the fibrosis segmentation model log folder, and myocardium segmentation predictions are in the myocardium segmentation log folders.


## 1. Training myocardium segmentation & inference
Most of the fibrosis classification/segmentation models require myocardium segmentations as input, so we create those first.

* `python train_myo_segmentation.py` to train a model.

* `python inference_myocard_segmentation.py --load_checkpoint /path/to/model.ckpt --input_path /path/to/images --output_path /path/to/myo_predictions` to make prediction myocardium segmentation images.

## 2. Training fibrosis classifier & creating segmentation pseudo-labels
For the weakly-supervised fibrosis segmentation, we can use either one fibrosis label per 2D image slice (slice-level supervision) or one fibrosis label per 3D stack (stack-level supervision). Obviously, slice-level supervision will perform better if we have equal data for both supervision levels.
 * **Slice-level supervision**
    1. `python train_classifier_2d.py --myo_mask_pooling` to train fibrosis classification model.
    2. `python make_and_evaluate_cams_2D.py --save --load_checkpoint /path/to/model.ckpt --otsu_mask` to create fibrosis pseudolabel images.
    3. (Optional) `python evaluate_classification_model.py --dataset deeprisk --load_checkpoint /path/to/model.ckpt` to create a dictionary of per-slice fibrosis predictions.
    
 * **Stack-level supervision**
    1. `python train_classifier_3d.py --myo_mask_pooling --weighted_loss` to train fibrosis classification model.
    2. `python make_and_evaluate_cams_3D.py --save --load_checkpoint /path/to/model.ckpt --otsu_mask` to create fibrosis pseudolabel images.
    3. (Optional) `python evaluate_classification_model.py --dataset deeprisk --load_checkpoint /path/to/model.ckpt` to create a dictionary of per-stack fibrosis predictions.



## 3. Training fibrosis segmentation & inference

*  Training with chosen supervision-level:
    * `python train_fib_segmentation.py --train_with_ground truth`, to train a **fully-supervised** model, i.e. with ground truth fibrosis segmentation. Uses myocardium segmentations as model input by default, which especially helps with relatively low number of ground truth segmentations.
    * `python train_fib_segmentation.py --pseudoseg_path /path/to/pseudolabels --model Floor_UNet2D`, to train a **weakly-supervised** model with pseudolabels. Weak supervision seems to perform better when myocardium segmentations are not given as input, which is why `--model Floor_UNet2D` is used instead of the default `--model Floor_UNet2D_stacked` (which stacks MRI image + myocardium segmentation as inputs). 

* `python inference_fibrosis_segmentation.py --dataset deeprisk --load_checkpoint /path/to/model.ckpt --input_path /path/to/images --pred_myo_path /path/to/myocardium_predictions --output_path /path/to/fibrosis_predictions` to make prediction fibrosis images.

## 4. Segmentation evaluation and visualization
Some evaluations and visualizations are already logged during training of the models, and can be accessed using tensorboard on the model folder. Post-training there are the following options:

* Run `python evaluate_segmentation_model --dataset deeprisk --pred_myo_dir /path/to/myocardium_segmentations --pred_fib_dir /path/to/fibrosis_predictions`, to generate a dictionary with different per-patient and per-slice statistics and metrics. This can be used in `notebooks/evaluate_segmentations.ipynb` to generate different metric plots.

* Create example image plots with side-by-side ground truth and inferenced myocardium/fibrosis segmentations in `notebooks/plot_inferenced_segmentations.ipynb`

* Look at segmentations/images in an external viewer, such as [MITK](https://www.mitk.org/wiki/Downloads)


