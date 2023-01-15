""" Trains a model for per-slice binary fibrosis classification.
The trained model can be used to create fibrosis pseudo-labels for
segmentation with make_and_evaluate_cams_2D.py.
Afterwards, training a segmentation model with those pseudolabels means you have
trained a fibrosis segmentation model with (weak) slice-level supervision.
"""

from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision
from datasets import DeepRiskDataset2D
from models.convnext import LightningConvNeXt
from models.drn import LightningDRN
from models.pl_stage_net import SingleStageBase
from models.resnet import LightningResNet
from models.simplenet import LightningSimpleNet
from preprocessing import compose_transforms_with_segmentations
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader


def load_dataset(hparams):
    # compose transforms
    train_transforms = compose_transforms_with_segmentations(hparams, "train")
    val_transforms = compose_transforms_with_segmentations(hparams, "val")

    print("train transforms:")
    print(train_transforms)
    print("val_transforms:")
    print(val_transforms)

    if hparams.dataset == 'deeprisk':
        # set and check paths
        DATA_DIR = Path(hparams.data_path)
        IMG_DIR = DATA_DIR.joinpath(hparams.img_path)
        LABELS_FILE = DATA_DIR.joinpath(hparams.weak_labels_path)
        if hparams.seg_labels_dir != None:
            SEG_LABELS_DIR = DATA_DIR.joinpath(hparams.seg_labels_dir)
        else:
            SEG_LABELS_DIR = None
        if hparams.no_roi_crop == True:
            MYOSEG_DIR = None
        else:
            MYOSEG_DIR = DATA_DIR.joinpath(Path(hparams.myoseg_path))
            assert MYOSEG_DIR.exists(), f"{MYOSEG_DIR} does not exist"
        assert DATA_DIR.exists()
        assert IMG_DIR.exists()
        assert LABELS_FILE.exists()

        # create dataset splits
        dataset_train = DeepRiskDataset2D(IMG_DIR, LABELS_FILE, seg_labels_dir=SEG_LABELS_DIR,
                                            transform=train_transforms, split="train",
                                            train_frac = hparams.train_frac,
                                            split_seed=hparams.splitseed, myoseg_dir=MYOSEG_DIR,
                                            include_no_myo=hparams.include_no_myo,
                                            roi_crop=hparams.roi_crop,
                                            img_type=hparams.img_type)
        dataset_val = DeepRiskDataset2D(IMG_DIR, LABELS_FILE, seg_labels_dir=SEG_LABELS_DIR,
                                        transform=val_transforms, split="val",
                                        train_frac = hparams.train_frac,
                                        split_seed=hparams.splitseed, myoseg_dir=MYOSEG_DIR,
                                        include_no_myo=hparams.include_no_myo,
                                        roi_crop=hparams.roi_crop,
                                        img_type=hparams.img_type)
        dataset_test = DeepRiskDataset2D(IMG_DIR, LABELS_FILE, seg_labels_dir=SEG_LABELS_DIR,
                                        transform=val_transforms, split="test",
                                        train_frac = hparams.train_frac,
                                        split_seed=hparams.splitseed, myoseg_dir=MYOSEG_DIR,
                                        include_no_myo=hparams.include_no_myo,
                                        roi_crop=hparams.roi_crop,
                                        img_type=hparams.img_type)

        print(f"Train data: {len(dataset_train)}, positive: {sum(dataset_train.labels)}")
        print(f"Train data with gt: {dataset_train.get_n_gt_slices()}, positive with gt: {dataset_train.get_n_fibrotic_gt_slices()}")
        print(f"Validation data: {len(dataset_val)}, positive: {sum(dataset_val.labels)}")
        print(f"Validation data with gt: {dataset_val.get_n_gt_slices()}, positive with gt: {dataset_val.get_n_fibrotic_gt_slices()}")
        print(f"Test data: {len(dataset_test)}, positive: {sum(dataset_test.labels)}")
        print(f"Test data with gt: {dataset_test.get_n_gt_slices()}, positive with gt: {dataset_test.get_n_fibrotic_gt_slices()}")
    elif hparams.dataset == 'cifar10':
        DATA_DIR = Path(hparams.data_path)
        CIFAR_DIR = DATA_DIR.joinpath('cifar10_data')
        assert DATA_DIR.exists()
        dataset_train = torchvision.datasets.CIFAR10(root=CIFAR_DIR, train=True,
                                        download=True, transform=train_transforms)
        dataset_val = torchvision.datasets.CIFAR10(root=CIFAR_DIR, train=False,
                                            download=True, transform=val_transforms)
        dataset_test = dataset_val

    return dataset_train, dataset_val, dataset_test


def init_model(hparams):
    """ Initializes the correct model for the input (hyper)parameters and returns it."""
    kwargs = {}
    # add dataset specific arguments
    if hparams.dataset == 'deeprisk':
        kwargs['in_chans'] = 1
        kwargs['myo_input'] = hparams.myo_input
        if hparams.myo_input == True:
            kwargs['in_chans'] += 1
        kwargs['num_classes'] = 1
    elif hparams.dataset == 'cifar10':
        kwargs['in_chans'] = 3
        kwargs['num_classes'] = 10
    # add arguments used by every model
    kwargs['heavy_log'] = hparams.heavy_log
    kwargs['lr'] = hparams.lr
    kwargs['weight_decay'] = hparams.weight_decay
    kwargs['label_smoothing'] = hparams.label_smoothing
    kwargs['schedule'] = hparams.schedule
    kwargs['warmup_iters'] = hparams.warmup_iters
    kwargs['max_iters'] = hparams.max_iters
    kwargs['pooling'] = hparams.pooling
    kwargs['myo_mask_pooling'] = hparams.myo_mask_pooling
    kwargs['myo_mask_threshold'] = hparams.myo_mask_threshold
    kwargs['myo_mask_dilation'] = hparams.myo_mask_dilation
    kwargs['myo_mask_prob'] = hparams.myo_mask_prob
    kwargs['feature_dropout'] = hparams.feature_dropout
    kwargs['cam_dropout'] = hparams.cam_dropout
    kwargs['cheat_dice'] = hparams.cheat_dice
    kwargs['downsampling'] = hparams.downsampling
    # add model specific arguments
    if hparams.model in ['resnet18', 'resnet50', 'resnet101']:
        kwargs['pretrain'] = hparams.pretrain
        kwargs['model'] = hparams.model
        kwargs['highres'] = hparams.highres
        kwargs['no_cam'] = hparams.no_cam
    elif hparams.model == 'simple':
        kwargs['highres'] = hparams.highres
    elif hparams.model == 'convnext':
        if hparams.depths != None:      kwargs['depths'] =  hparams.depths
        if hparams.dims != None:        kwargs['dims'] =    hparams.dims
        if hparams.strides != None:     kwargs['strides'] = hparams.strides
        kwargs['drop_path_rate'] = hparams.drop_path_rate
        kwargs['stem_type'] = hparams.stem
        kwargs['norm_type'] = hparams.model_norm
        kwargs['no_cam'] = hparams.no_cam
        kwargs['layer_scale_init_value'] = hparams.layer_scale_init_value
    elif hparams.model in ['drnc', 'drnd', 'drnc26', 'drnd22', 'drnd24']:
        if hparams.model in ['drnc', 'drnc26']:                             kwargs['arch'] =    'C'
        if hparams.model in ['drnd', 'drnd22', 'drnd24']:                   kwargs['arch'] =    'D'
        if hparams.depths != None:                                          kwargs['depths'] =  hparams.depths
        if hparams.depths == None and hparams.model in ['drnd', 'drnd22']:    kwargs['depths'] =  [1, 1, 2, 2, 2, 2, 1, 1]
        if hparams.depths == None and hparams.model == 'drnd24':            kwargs['depths'] =  [1, 1, 2, 2, 2, 2, 2, 2]
        if hparams.depths == None and hparams.model in ['drnc', 'drnc26']:    kwargs['depths'] =  [1, 1, 2, 2, 2, 2, 1, 1]
        if hparams.dims != None: kwargs['dims'] = hparams.dims

    # set correct class for model
    if hparams.singlestage == False:
        MODEL_CLASSES = {'resnet18' : LightningResNet, 'resnet50' : LightningResNet, 'resnet101' : LightningResNet,
                     'simple' : LightningSimpleNet,
                     'convnext' : LightningConvNeXt,
                     'drnc' : LightningDRN, 'drnc26' : LightningDRN, 'drnd' : LightningDRN, 'drnd22' : LightningDRN, 'drnd24' : LightningDRN}
        model_class = MODEL_CLASSES[hparams.model]
    elif hparams.singlestage == True:
        # single stage needs some parameters changed
        model_class = SingleStageBase
        kwargs['backbone'] = hparams.model
        kwargs['num_classes'] = kwargs['num_classes'] + 1
        kwargs['pre_weights_path'] = hparams.ss_pre_weights_path
        kwargs['pretrain'] = hparams.ss_pretrain
        kwargs['focal_lambda'] = hparams.focal_lambda
        kwargs['sg_psi'] = hparams.sg_psi
        kwargs['use_aspp'] = hparams.aspp
        kwargs['use_focal'] = hparams.focal
        kwargs['first_dim'] = hparams.dims[0]

    # load checkpoint if specified
    if hparams.load_checkpoint != False:
        model_class = model_class.load_from_checkpoint
        kwargs['checkpoint_path'] = hparams.load_checkpoint
    # load model
    print(f'model {kwargs=}')
    return model_class(**kwargs)



def train_model(hparams):
    pl.seed_everything(hparams.trainseed, workers=True)

    # prepare dataloaders
    dataset_train, dataset_val, dataset_test = load_dataset(hparams)
    train_loader = DataLoader(dataset_train,
                            batch_size=hparams.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=hparams.num_workers)
    val_loader = DataLoader(dataset_val,
                            batch_size=hparams.batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=hparams.num_workers)
    test_loader = DataLoader(dataset_test,
                            batch_size=hparams.batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=hparams.num_workers)

    hparams.max_iters = len(train_loader) * hparams.max_epochs
    print(f'{hparams.max_iters=}')

    # gives potential speedup if input size does not change
    torch.backends.cudnn.benchmark = True

    # define model & trainer
    model = init_model(hparams)
    logger = TensorBoardLogger("tb_logs_classification_2D", name=hparams.logdir)
    trainer = pl.Trainer(deterministic=False,
                         gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         fast_dev_run=hparams.fast_dev_run,
                         callbacks=[ModelCheckpoint(),
                                    LearningRateMonitor("step")],
                         logger=logger
                         )

    # train & test model
    trainer.fit(model, train_loader, val_loader)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    return



if __name__ == "__main__":
    parser = ArgumentParser()
    # dataset related
    parser.add_argument("--dataset", type=str, default="deeprisk", choices=['deeprisk', 'cifar10'],
                        help="Dataset to train on. On cifar10, probably not everything works.")
    parser.add_argument("--img_type", type=str, default='PSIR', choices=['PSIR', 'MAG'],
                         help="For deeprisk, which type of LGE images to use.")
    # hardware related
    parser.add_argument("--gpus", default=0,
                         help="Number of gpus to use. Set to 0 for cpu.")
    parser.add_argument("--num_workers", type=int, default=0,
                         help="Number of workers for dataloaders.")
    # reproducability related
    parser.add_argument("--trainseed", type=int, default=42,
                         help="Seed used for training.")
    parser.add_argument("--splitseed", type=int, default=84,
                         help="See datasets.py for more information.")
    parser.add_argument("--train_frac", type=float, default=0.6,
                         help="See datasets.py for more information.")
    # paths related (base data dir and relative paths to other dirs)
    parser.add_argument("--data_path", type=str, default=r"../data",
                         help="Path to directory containing all data.")
    parser.add_argument("--img_path", type=str, default=r"all_niftis_n=657",
                         help="Relative path from data_path to image directory.")
    parser.add_argument("--weak_labels_path", type=str, default=r"weak_labels_n=657.xlsx",
                         help="Relative path from data_path to excel sheet with weak labels.")
    parser.add_argument("--load_checkpoint", type=str, default=False,
                         help="Full path + filename of model checkpoint. When False, train from scratch.")
    parser.add_argument("--myoseg_path", type=str, default=r"myocard_predictions/deeprisk_myocardium_predictions",
                         help="Relative path from data_path to directory with predicted myocardium segmentations.")
    parser.add_argument("--seg_labels_dir", type=str, default=r"fibrosis_labels_n=117",
                         help="Relative path from data_path to directory with ground truth fibrosis segmentations.")
    parser.add_argument("--logdir", type=str, default=None,
                         help="Save run logs under this directory.")
    parser.add_argument("--ss_pre_weights_path", type=str, default=None,
                         help="Some parameter necessary for single-stage weak supervision, which didn't end up working.")
    # model related
    parser.add_argument("--singlestage", action='store_true',
                         help="Whether to use single-stage weak supervision, which didn't end up working (Feel free to ignore).")
    parser.add_argument("--model", type=str, default='drnd', 
                        choices=['convnext', 'resnet18', 'resnet50', 'resnet101', 'simple', 'drnc', 'drnc26', 'drnd', 'drnd22', 'drnd24', 'resnet38', 'vgg16'],
                        help="Model architecture to use.")
    parser.add_argument("--no_cam", action="store_true",
                         help="For ResNet and Convnext, whether to use CAMs. Used for testing equivalence between networks with and without CAM.")
    parser.add_argument("--pretrain", action='store_true',
                         help="Only for Resnet, whether to use pretrained models.")
    parser.add_argument("--highres", action='store_true',
                         help="For Resnet and simplenet, whether increase CAM resolution by reducing convolution strides.")
    parser.add_argument("--depths", nargs="+", type=int, default=None,
                         help="For convnext, drnc and drnd, number of blocks in each model stage (within one stage the same spatial resolution is kept). Set to None for default depths.")
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64, 128, 128, 128, 128, 128],
                         help="For convext, drnc and drnd, number of dimensions in each model stage.")
    parser.add_argument("--strides", nargs="+", type=int, default=[4, 2, 2, 2],
                         help="For convnext, strides used in different stages.")
    parser.add_argument("--stem", type=str, default="patch", choices=["patch", "resnet_like"],
                         help="For convnext, determines the first convolutional layer.")
    parser.add_argument("--model_norm", type=str, default="layer", choices=["layer", "batch"],
                         help="For convnext, whether to use layer or batchnorm. Other Models always use batchnorm.")
    parser.add_argument("--pooling", type=str, default="avg", choices=["avg", "max", "ngwp"],
                         help="Type of pooling to use at end of classifier.")
    parser.add_argument("--myo_mask_pooling", action="store_true",
                         help="Whether to only pool over pixels within the predicted myocardium segmentation.")
    parser.add_argument("--myo_mask_threshold", type=float, default=0.4,
                         help="Threshold to use for converting predicted myocardium segmentations into a binary mask.")
    parser.add_argument("--myo_mask_dilation", type=int, default=3,
                         help="Size of dilation kernel to apply to myocardium masks.")
    parser.add_argument("--myo_mask_prob", type=float, default=1,
                         help="Probability with which to restrict the final pooling to the myocardium mask.")
    parser.add_argument("--downsampling", type=int, choices=[4, 8], default=4,
                         help="For DRN, the downsampling factor from input size to CAM size.")
    parser.add_argument("--myo_input", action="store_true",
                         help="Whether to give predicted myocardium segmentations as input to the classifier.")
    # learning hyperparameters
    parser.add_argument("--max_epochs", type=int, default=5,
                         help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="Adam learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0,
                         help="AdamW weight decay.")
    parser.add_argument("--label_smoothing", type=float, default=0,
                         help="Label smoothing, which can regularize multi-class problems. Not really useful for binary classification problems (just scales the loss?).")
    parser.add_argument("--drop_path_rate", type=float, default=0,
                         help="Convnext parameter, see convnext.py for details.")
    parser.add_argument("--layer_scale_init_value", type=float, default=0,
                         help="Convnext parameter, see convnext.py for details.")
    parser.add_argument("--schedule", type=str, default=None, choices=["step", "cosine"],
                         help="Learning rate schedule. Either with hardcoded step-schedule or cosine schedule (which generally works well for Adam if you don't know the correct learning rate).")
    parser.add_argument("--warmup_iters", type=int, default=0,
                         help="Number of iterations for linear warmup scaling of the learning rate.")
    parser.add_argument("--ss_pretrain", type=int, default=5,
                         help="Some single-stage weak supervision parameter. Use at own dicretion")
    parser.add_argument("--focal_lambda", type=float, default=0.01,
                         help="Some single-stage weak supervision parameter. Use at own discretion.")
    parser.add_argument("--sg_psi", type=float, default=0.3,
                         help="Some single-stage weak supervision parameter. Use at own discretion.")
    parser.add_argument("--aspp", action="store_true",
                         help="Some single-stage weak supervision parameter. Use at own discretion.")
    parser.add_argument("--focal", action="store_true",
                         help="Some single-stage weak supervision parameter. Use at own discretion.")
    # data augmentation
    parser.add_argument("--image_norm", type=str, default="per_image", choices=["per_image", "global_statistic", "global_agnostic", "no_norm"],
                         help="Type of image normalization. See preprocessing.py for details.")
    parser.add_argument("--no_roi_crop", action='store_true',
                         help="Include flag to not do a region of interest crop around the heart. Will use centercrop instead.")
    parser.add_argument("--include_no_myo", default=True,
                         help="Whether to include image slices that don't contain myocardium in the dataset.")
    parser.add_argument("--roi_crop", type=str, default="fixed", choices=['fitted', 'fixed'],
                         help="Type of region of interest crop. See dataset.py for details.")
    parser.add_argument("--center_crop", type=int, default=224,
                         help="Size of center crop if you opt for no ROI crop.")
    parser.add_argument("--input_size", type=int, default=224,
                         help="Image size of model inputs.")
    parser.add_argument("--rotate", type=int, default=90,
                         help="Possible degrees of rotation for data augmentation.")
    parser.add_argument("--translate", nargs=2, type=float, default=(0.1, 0.1),
                         help="Possible translation range in x and y direction for data augmentation, as a factor of image size.")
    parser.add_argument("--scale", nargs=2, type=float, default=(0.8, 1.2),
                         help="Possible scaling factor range in x and y direction for data augmentation.")
    parser.add_argument("--shear", nargs=4, type=int, default=(-5, 5, -5, 5),
                         help="Shearing parameters for data augmentation.")
    parser.add_argument("--randomaffine_prob", type=float, default=0.5,
                         help="Probability of doing a randomaffine data augmentation (rotation+scale+translation+shear).")
    parser.add_argument("--brightness", type=float, default=0.1,
                         help="Brightness parameter for random colorjitter data augmentation.")
    parser.add_argument("--contrast", type=float, default=0.1,
                         help="Contrast parameter for random colorjitter data augmentation")
    parser.add_argument("--hflip", type=float, default=0.5,
                         help="Probability of random horizontal flip data augmentation")
    parser.add_argument("--vflip", type=float, default=0.5,
                         help="Probability of random vertical flip data augmentation.")
    parser.add_argument("--randomcrop", type=int, default=False,
                         help="Size of randomcrop data augmentation. Set to False for no random crop.")
    parser.add_argument("--randomcrop_prob", type=float, default=1.0,
                         help="Probability of taking a random crop data augmentation.")
    parser.add_argument("--randomerasing_scale", nargs=2, type=float, default=(0.02, 0.33),
                         help="Range for erasing area of random erasing data augmentation, as factor of image size.")
    parser.add_argument("--randomerasing_ratio", nargs=2, type=float, default=(0.3, 3.3),
                         help="Range for aspect ratio of random erasing data augmentation.")
    parser.add_argument("--randomerasing_probs", nargs="+", type=float, default=[],
                         help="Probability of random erasing data augmentation. Set multiple probabilities for multiple erasing transforms.")
    parser.add_argument("--feature_dropout", type=float, default=0.0,
                         help="Probability of a feature being dropped out at the last layer.")
    parser.add_argument("--cam_dropout", type=float, default=0.0,
                         help="Probability of a spatial location being dropped at before the final pooling.")
    # misc
    parser.add_argument("--fast_dev_run", type=int, default=False,
                         help="Do a test run with x batches.")
    parser.add_argument("--heavy_log", type=int, default=10,
                         help="Only log some metrics/images every heavy_log epochs. Improves speed and prevents logs from containing to many images.")
    parser.add_argument("--cheat_dice", action="store_true",
                         help="Whether to cheat the Dice scores during logging. This uses the ground truth fibrosis segmentations, but downsampled and optionally restricted to predicted myocardium segmentations. Can give an indication of the maximum Dice score.")

    args = parser.parse_args()

    train_model(args)
