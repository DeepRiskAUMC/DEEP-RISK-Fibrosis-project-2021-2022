"""Trains a fibrosis segmentation model,
either with ground truth labels or (weakly supervised) pseudo-labels.
"""

from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
from datasets import (DeepRiskDatasetSegmentation2D,
                      DeepRiskDatasetSegmentation3D)
from models.segmentation_models.pl_fib_segmentation_model import \
    SegmentationModel
from preprocessing import compose_transforms_with_segmentations
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader


def load_dataset(hparams):
    if hparams.dataset == 'deeprisk':
        # set and check paths
        DATA_DIR = Path(hparams.data_path)
        IMG_DIR = DATA_DIR.joinpath(hparams.img_path)
        PSEUDOSEG_DIR = Path(hparams.pseudoseg_path)
        LABELS_FILE = DATA_DIR.joinpath(hparams.weak_labels_path)

        assert DATA_DIR.exists()
        assert IMG_DIR.exists()
        assert LABELS_FILE.exists()

        if hparams.gt_seg_dir != None:
            GT_SEG_DIR = DATA_DIR.joinpath(hparams.gt_seg_dir)
        else:
            GT_SEG_DIR = None

        if hparams.no_roi_crop == True:
            MYOSEG_DIR = None
        else:
            MYOSEG_DIR = DATA_DIR.joinpath(Path(hparams.myoseg_path))
            assert MYOSEG_DIR.exists(), f"{MYOSEG_DIR} does not exist"

        if hparams.gt_myoseg_dir != None:
            GT_MYOSEG_DIR = DATA_DIR.joinpath(hparams.gt_myoseg_dir)
            assert GT_MYOSEG_DIR.exists()
        else:
            GT_MYOSEG_DIR = None

        if 'cascaded' in hparams.model:
            # image types: 
            # mri image, pseudo fibrosis,
            # ground truth fibrosis, preliminary fibrosis prediction,
            # ground truth myocardium
            num_img_types = 5
            if MYOSEG_DIR != None:
                # also predicted myocardium
                num_img_types += 1
            train_transforms = compose_transforms_with_segmentations(hparams, 'train', num_img_types)
            val_transforms = compose_transforms_with_segmentations(hparams, 'val', num_img_types)
            print("train transforms:")
            print(train_transforms)
            print("val_transforms:")
            print(val_transforms)
            # cascaded model needs preliminary fibrosis predictions to refine
            # IDEA : preliminary with 2D model, refinement with 3D model
            PRED_FIB_DIR = DATA_DIR.joinpath(hparams.pred_fib_path)
            assert PRED_FIB_DIR.exists()
            dataset_train = DeepRiskDatasetSegmentation3D(IMG_DIR, LABELS_FILE,
                                                            pseudo_fib_dir=PSEUDOSEG_DIR,
                                                            gt_fib_dir=GT_SEG_DIR,
                                                            pred_fib_dir=PRED_FIB_DIR,
                                                            gt_myo_dir=GT_MYOSEG_DIR,
                                                            pred_myo_dir=MYOSEG_DIR,
                                                            transform=train_transforms,
                                                            split='train',
                                                            roi_crop=hparams.roi_crop,
                                                            img_type=hparams.img_type,
                                                            pixel_gt_only=hparams.train_with_gt)
            dataset_val = DeepRiskDatasetSegmentation3D(IMG_DIR, LABELS_FILE,
                                                            pseudo_fib_dir=PSEUDOSEG_DIR,
                                                            gt_fib_dir=GT_SEG_DIR,
                                                            pred_fib_dir=PRED_FIB_DIR,
                                                            gt_myo_dir=GT_MYOSEG_DIR,
                                                            pred_myo_dir=MYOSEG_DIR,
                                                            transform=val_transforms,
                                                            split='val',
                                                            roi_crop=hparams.roi_crop,
                                                            img_type=hparams.img_type,
                                                            pixel_gt_only=hparams.train_with_gt)
            dataset_test = DeepRiskDatasetSegmentation3D(IMG_DIR, LABELS_FILE,
                                                            pseudo_fib_dir=PSEUDOSEG_DIR,
                                                            gt_fib_dir=GT_SEG_DIR,
                                                            pred_fib_dir=PRED_FIB_DIR,
                                                            gt_myo_dir=GT_MYOSEG_DIR,
                                                            pred_myo_dir=MYOSEG_DIR,
                                                            transform=val_transforms,
                                                            split='test',
                                                            roi_crop=hparams.roi_crop,
                                                            img_type=hparams.img_type,
                                                            pixel_gt_only=hparams.train_with_gt)
        else:
            # compose transforms
            train_transforms = compose_transforms_with_segmentations(hparams, "train", 5)
            val_transforms = compose_transforms_with_segmentations(hparams, "val", 5)

            print("train transforms:")
            print(train_transforms)
            print("val_transforms:")
            print(val_transforms)

            dataset_train = DeepRiskDatasetSegmentation2D(IMG_DIR, LABELS_FILE, pseudoseg_dir=PSEUDOSEG_DIR, gt_seg_dir=GT_SEG_DIR,
                                                transform=train_transforms, split="train",
                                                train_frac = hparams.train_frac,
                                                split_seed=hparams.splitseed, myoseg_dir=MYOSEG_DIR,
                                                gt_myoseg_dir=GT_MYOSEG_DIR,
                                                include_no_myo=hparams.include_no_myo,
                                                roi_crop=hparams.roi_crop,
                                                pixel_gt_only=hparams.train_with_gt,
                                                img_type=hparams.img_type)
            dataset_val = DeepRiskDatasetSegmentation2D(IMG_DIR, LABELS_FILE, pseudoseg_dir=PSEUDOSEG_DIR, gt_seg_dir=GT_SEG_DIR,
                                            transform=val_transforms, split="val",
                                            train_frac = hparams.train_frac,
                                            split_seed=hparams.splitseed, myoseg_dir=MYOSEG_DIR,
                                            gt_myoseg_dir=GT_MYOSEG_DIR,
                                            include_no_myo=hparams.include_no_myo,
                                            roi_crop=hparams.roi_crop,
                                            pixel_gt_only=hparams.train_with_gt,
                                            img_type=hparams.img_type)
            dataset_test = DeepRiskDatasetSegmentation2D(IMG_DIR, LABELS_FILE, pseudoseg_dir=PSEUDOSEG_DIR, gt_seg_dir=GT_SEG_DIR,
                                            transform=val_transforms, split="test",
                                            train_frac = hparams.train_frac,
                                            split_seed=hparams.splitseed, myoseg_dir=MYOSEG_DIR,
                                            gt_myoseg_dir=GT_MYOSEG_DIR,
                                            include_no_myo=hparams.include_no_myo,
                                            roi_crop=hparams.roi_crop,
                                            pixel_gt_only=hparams.train_with_gt,
                                            img_type=hparams.img_type)

            print(f"Train data: {len(dataset_train)}")
            print(f"Validation data: {len(dataset_val)}")
            print(f"Test data: {len(dataset_test)}")

    return dataset_train, dataset_val, dataset_test



def init_model(hparams):
    kwargs = {}
    # add dataset specific arguments
    if hparams.dataset == 'deeprisk':
        kwargs['in_chans'] = 1
        kwargs['num_classes'] = 1
    # add arguments used by every model
    kwargs['heavy_log'] = hparams.heavy_log
    kwargs['lr'] = hparams.lr
    kwargs['weight_decay'] = hparams.weight_decay
    kwargs['schedule'] = hparams.schedule
    kwargs['warmup_iters'] = hparams.warmup_iters
    kwargs['max_iters'] = hparams.max_iters
    kwargs['bilinear'] = True if hparams.upsampling == 'upsample' else False
    kwargs['model_name'] = hparams.model
    kwargs['loss_function_string'] = hparams.loss_function
    kwargs['underfitting_warmup'] = hparams.underfitting_warmup
    kwargs['underfitting_k'] = hparams.underfitting_k
    kwargs['feature_multiplication'] = hparams.feature_multiplication
    # ground truth training/finetuning
    # IDEA: pretrain with large set of pseudolabels,
    # then refine with smaller set of ground truth labels.
    # Did not find improvement in performance so far on Deep Risk,
    # but finetuning (with ground truth) could be useful for new datasets.
    kwargs['train_with_gt'] = hparams.train_with_gt
    kwargs['freeze_n'] = hparams.freeze_n
    kwargs['train_bn'] = hparams.train_bn

    model_class = SegmentationModel
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

    # potential speedup if input size does not change
    torch.backends.cudnn.benchmark = True

    # define model & trainer
    model = init_model(hparams)
    logger = TensorBoardLogger("tb_logs_segmentation_fib", name=hparams.logdir)

    trainer = pl.Trainer(deterministic=False,
                         gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         fast_dev_run=hparams.fast_dev_run,
                         callbacks=[ModelCheckpoint(),
                                    LearningRateMonitor("step")],
                         logger=logger
                         )

    # train & test
    trainer.fit(model, train_loader, val_loader)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    return



if __name__ == "__main__":
    parser = ArgumentParser()
    # choose dataset
    parser.add_argument("--dataset", type=str, default="deeprisk", choices=['deeprisk'],
                         help="Dataset to train on.")
    parser.add_argument("--img_type", type=str, default='PSIR', choices=['PSIR', 'MAG'],
                         help="For deeprisk, which type of LGE images to use.")
    # hardware
    parser.add_argument("--gpus", default=0,
                         help="Number of gpus to use. Set to 0 for cpu.")
    parser.add_argument("--num_workers", type=int, default=0,
                         help="Number of workers for dataloaders.")
    # reproducability
    parser.add_argument("--trainseed", type=int, default=42,
                         help="Seed used for training.")
    parser.add_argument("--splitseed", type=int, default=84,
                         help="See datasets.py for more information.")
    parser.add_argument("--train_frac", type=float, default=0.6,
                         help="See datasets.py for more information.")
    # paths (base data dir and relative paths to other dirs)
    parser.add_argument("--data_path", type=str, default=r"../data",
                         help="Path to directory containing all data.")
    parser.add_argument("--img_path", type=str, default=r"all_niftis_n=657",
                         help="Relative path from data_path to image directory.")
    parser.add_argument("--weak_labels_path", type=str, default=r"weak_labels_n=657.xlsx",
                         help="Relative path from data_path to excel sheet with weak labels.")
    parser.add_argument("--load_checkpoint", type=str, default=False,
                         help="Full path + filename of model checkpoint. When False, train from scratch.")
    parser.add_argument("--pseudoseg_path", type=str, default=r"\\amc.intra\users\R\rcklein\home\deeprisk\weakly_supervised\tb_logs\drnd_masked_avg_dilation=3\version_2\pseudolabels",
                         help="Path to directory with fibrosis pseudolabels (which should be located in the directory where a classification model is saved).")
    parser.add_argument("--myoseg_path", type=str, default=r"myocard_predictions/deeprisk_myocardium_predictions",
                         help="Relative path from data_path to directory with predicted myocardium segmentations.") #r"nnUnet_results\nnUNet\2d\Task500_MyocardSegmentation\predictions"
    parser.add_argument("--pred_fib_path", type=str, default=None,
                        help="Relative path from data_path to directory with predicted fibrosis segmentations. Needed for cascaded segmentation model.")
    parser.add_argument("--gt_seg_dir", type=str, default=r"fibrosis_labels_n=117",
                        help="Relative path from data_path to directory with ground truth fibrosis segmentations.")
    parser.add_argument("--gt_myoseg_dir", type=str, default=r"myo_labels_n=117",
                        help="Relative path from data_path to directory with ground truth myocardium segmentations.")
    parser.add_argument("--logdir", type=str, default=None,
                         help="Save run logs under this directory.")
    # model
    parser.add_argument('--model', default='Floor_UNet2D_stacked', type=str,
                        help='What model to use for the segmentation',
                        choices=['UNet2D', 'UNet2D_stacked',
                                 'UNet2D_small', 'UNet2D_stacked_small',
                                 'CANet2D', 'CANet2D_stacked',
                                 'Floor_UNet2D', 'Floor_UNet2D_stacked',
                                 'UNet3D', 'UNet3D_channels', 'UNet3D_channels_stacked', 'UNet3D_half', 'UNet3D_cascaded', 'UNet3D_cascaded_stacked'])
    parser.add_argument('--upsampling', default='convtrans', type=str,
                        help='What kind of model upsampling we want to use',
                        choices=['upsample', 'convtrans'])
    parser.add_argument('--feature_multiplication', default='2', type=int,
                        help='The factor by which the number of features in the model are is multiplied')


    # learning hyperparameters
    parser.add_argument('--loss_function', default='dice', type=str, help='What loss function to use for the segmentation', choices=['dice', 'dice+WCE'])
    parser.add_argument('--underfitting_warmup', type=int, default=None,
                         help="Number of epochs before starting loss underfitting. Default=None does not use underfitting, which is recommended. ")
    parser.add_argument('--underfitting_k', default=0.5,
                         help="Experimental setting, where 'noisy' (false?) positive labels are ignored if the model predicts less than k. Is supposed to fix the problem of false positives in pseudolabels. Unsurprisingly, resulted in unstable training.")
    parser.add_argument("--max_epochs", type=int, default=5,
                         help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="Adam learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0,
                         help="AdamW weight decay.")
    parser.add_argument("--schedule", type=str, default=None, choices=["step", "cosine"],
                         help="Learning rate schedule. Either with hardcoded step-schedule or cosine schedule (which generally works well for Adam if you don't know the correct learning rate).")
    parser.add_argument("--warmup_iters", type=int, default=0,
                         help="Number of iterations for linear warmup scaling of the learning rate.")
    parser.add_argument("--train_with_gt", action='store_true',
                        help="Set flag to train with ground truth fibrosis segmentations instead of pseudolabels.")
    parser.add_argument("--freeze_n", type=int, default=None,
                        help="Freeze first n layers of segmentation model. Default=None leaves all layers unfrozen (i.e. trainable).")
    parser.add_argument("--train_bn", action='store_true',
                        help="Set flag to exempt batchnorm layers from being frozen.")
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
    # misc
    parser.add_argument("--fast_dev_run", type=int, default=False,
                         help="Do a test run with x batches.")
    parser.add_argument("--heavy_log", type=int, default=10,
                         help="Only log some metrics/images every heavy_log epochs. Improves speed and prevents logs from containing to many images.")
    




    args = parser.parse_args()

    train_model(args)
