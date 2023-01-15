""" Like train_classifier_2d.py, but uses stack-level inputs. 
There is a option for both per-stack and per-slice fibrosis classification.
The trained model can be used to create fibrosis pseudo-labels for
segmentation with make_and_evaluate_cams_{2D/3D}.py.
Afterwards, training a segmentation model with those pseudolabels means you have
trained a fibrosis segmentation model with either (weak) slice-level or stack-level
 supervision, depending on the classification task chosen (--classification_level arg).
 """

from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from datasets import DeepRiskDataset3D
from models.pl_classification_model_3d import ClassificationModel3D
from preprocessing import compose_transforms_with_segmentations
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader


def load_dataset(hparams):
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
        dataset_train = DeepRiskDataset3D(IMG_DIR, LABELS_FILE, seg_labels_dir=SEG_LABELS_DIR,
                                            transform=None, split="train",
                                            train_frac = hparams.train_frac,
                                            split_seed=hparams.splitseed, myoseg_dir=MYOSEG_DIR,
                                            include_no_myo=hparams.include_no_myo,
                                            roi_crop=hparams.roi_crop,
                                            img_type=hparams.img_type)
        dataset_val = DeepRiskDataset3D(IMG_DIR, LABELS_FILE, seg_labels_dir=SEG_LABELS_DIR,
                                        transform=None, split="val",
                                        train_frac = hparams.train_frac,
                                        split_seed=hparams.splitseed, myoseg_dir=MYOSEG_DIR,
                                        include_no_myo=hparams.include_no_myo,
                                        roi_crop=hparams.roi_crop,
                                        img_type=hparams.img_type)
        dataset_test = DeepRiskDataset3D(IMG_DIR, LABELS_FILE, seg_labels_dir=SEG_LABELS_DIR,
                                        transform=None, split="test",
                                        train_frac = hparams.train_frac,
                                        split_seed=hparams.splitseed, myoseg_dir=MYOSEG_DIR,
                                        include_no_myo=hparams.include_no_myo,
                                        roi_crop=hparams.roi_crop,
                                        img_type=hparams.img_type)
        # compose transforms
        train_transforms = compose_transforms_with_segmentations(hparams, "train", depth_size=dataset_train.max_num_slices)
        val_transforms = compose_transforms_with_segmentations(hparams, "val", depth_size=dataset_val.max_num_slices)
        test_transforms = compose_transforms_with_segmentations(hparams, "val", depth_size=dataset_test.max_num_slices)
        dataset_train.add_transform(train_transforms)
        dataset_val.add_transform(val_transforms)
        dataset_test.add_transform(test_transforms)

        print("train transforms:")
        print(train_transforms)
        print("val_transforms:")
        print(val_transforms)

        print(f"Train data: {len(dataset_train)}, positive: {sum([1 if 1 in x else 0 for x in dataset_train.labels])}")
        print(f"Validation data: {len(dataset_val)}, positive: {sum([1 if 1 in x else 0 for x in dataset_val.labels])}")
        print(f"Test data: {len(dataset_test)}, positive: {sum([1 if 1 in x else 0 for x in dataset_test.labels])}")

    return dataset_train, dataset_val, dataset_test



def init_model(hparams):
    kwargs = {}
    # add dataset specific arguments
    if hparams.dataset == 'deeprisk':
        kwargs['in_chans'] = 1
        kwargs['myo_input'] = hparams.myo_input
        if hparams.myo_input == True:
            kwargs['in_chans'] += 1
        kwargs['num_classes'] = 1
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
    kwargs['classification_level'] =  hparams.classification_level
    kwargs['pos_weight'] = hparams.pos_weight
    # add model specific arguments
    if hparams.model in ['drnd', 'drnd22', 'drnd24']:
        if hparams.model in ['drnd', 'drnd22', 'drnd24']:                   kwargs['arch'] =    'D'
        if hparams.depths != None:                                          kwargs['depths'] =  hparams.depths
        if hparams.depths == None and hparams.model in ['drnd', 'drnd22']:    kwargs['depths'] =  [1, 1, 2, 2, 2, 2, 1, 1]
        if hparams.depths == None and hparams.model == 'drnd24':            kwargs['depths'] =  [1, 1, 2, 2, 2, 2, 2, 2]
        if hparams.dims != None: kwargs['dims'] = hparams.dims

    model_class = ClassificationModel3D

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
    if hparams.weighted_loss == True:
        # calculate corrected weight of positive examples: 
        # positive weight = (number of negative labels / number of positive labels)
        # note that the negative weight is always left at 1
        if hparams.classification_level == "3D":
            pos = sum([1 if 1 in x else 0 for x in dataset_train.labels])
            total = len(dataset_train.labels)
            neg = total - pos
        elif hparams.classification_level == "2D":
            neg = sum([1 if y == 0 else 0 for x in dataset_train.labels for y in x])
            pos = sum([1 if y == 1 else 0 for x in dataset_train.labels for y in x])
        hparams.pos_weight = neg / pos
        print(f"{neg=} {pos=} {hparams.pos_weight=}")
    else:
        hparams.pos_weight = 1

    # define model & trainer
    model = init_model(hparams)
    logger = TensorBoardLogger("tb_logs_classification_3D", name=hparams.logdir)

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
    # dataset related
    parser.add_argument("--dataset", type=str, default="deeprisk", choices=['deeprisk'],
                         help="Dataset to train on.")
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
                         help="""Full path + filename of model checkpoint.
                          When False, train from scratch.""")
    parser.add_argument("--myoseg_path", type=str,
                         default=r"myocard_predictions/deeprisk_myocardium_predictions",
                         help="""Relative path from data_path to directory with
                          predicted myocardium segmentations.""")
    parser.add_argument("--seg_labels_dir", type=str, default=r"fibrosis_labels_n=117",
                         help="""Relative path from data_path to directory with
                         ground truth fibrosis segmentations.""")
    parser.add_argument("--logdir", type=str, default=None,
                         help="Save run logs under this directory.")
    # model related
    parser.add_argument("--model", type=str, default='drnd',
                         choices=['drnd', 'drnd22', 'drnd24'],
                         help="Model architecture to use.")
    parser.add_argument("--depths", nargs="+", type=int, default=None,
                         help="""For convnext, drnc and drnd, number of blocks
                         in each model stage (within one stage the same spatial resolution
                         is kept). Set to None for default depths.""")
    parser.add_argument("--dims", nargs="+", type=int,
                         default=[16, 32, 64, 128, 128, 128, 128, 128],
                         help="For convext, drnc and drnd, number of dimensions in each model stage.")
    parser.add_argument("--pooling", type=str, default="avg", choices=["avg", "max", "ngwp"],
                         help="Type of pooling to use at end of classifier.")
    parser.add_argument("--myo_mask_pooling", action="store_true",
                         help="""Whether to only pool over pixels within the
                         predicted myocardium segmentation.""")
    parser.add_argument("--myo_mask_threshold", type=float, default=0.4,
                         help="""Threshold to use for converting predicted
                         myocardium segmentations into a binary mask.""")
    parser.add_argument("--myo_mask_dilation", type=int, default=3,
                         help="Size of dilation kernel to apply to myocardium masks.")
    parser.add_argument("--myo_mask_prob", type=float, default=1,
                         help="""Probability with which to restrict the
                         final pooling to the myocardium mask.""")
    parser.add_argument("--downsampling", type=int, choices=[4, 8], default=4,
                         help="For DRN, the downsampling factor from input size to CAM size.")
    parser.add_argument("--classification_level", type=str, default="3D", choices=["2D", "3D"],
                         help="""Whether the model should make per-slice (2D) or 
                         per-stack (3D) predictions. Model will be trained with corresponding 2D/3D labels.""")
    parser.add_argument("--myo_input", action="store_true",
                         help="Whether to give predicted myocardium segmentations as input to the classifier.")
    # learning hyperparameters
    parser.add_argument("--max_epochs", type=int, default=5,
                         help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="Adam learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0,
                         help="AdamW weight decay.")
    parser.add_argument("--label_smoothing", type=float, default=0,
                         help="""Label smoothing, which can regularize multi-class problems.
                          Not really useful for binary classification problems (just scales the loss?).""")
    parser.add_argument("--schedule", type=str, default=None, choices=["step", "cosine"],
                         help="""Learning rate schedule. Either with hardcoded step-schedule or cosine schedule 
                         (which generally works well for Adam if you don't know the correct learning rate).""")
    parser.add_argument("--warmup_iters", type=int, default=0,
                         help="Number of iterations for linear warmup scaling of the learning rate.")
    parser.add_argument("--weighted_loss", action='store_true',
                         help="Give flag to train with a class-weighted version of the loss.")
    # data augmentation
    parser.add_argument("--image_norm", type=str, default="per_image",
                         choices=["per_image", "global_statistic", "global_agnostic", "no_norm"],
                         help="Type of image normalization. See preprocessing.py for details.")
    parser.add_argument("--no_roi_crop", action='store_true',
                         help="Include flag to NOT do a region of interest crop around the heart. Will use centercrop instead.")
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
                         help="""Probability of random erasing data augmentation.
                         Set multiple probabilities for multiple erasing transforms.""")
    parser.add_argument("--feature_dropout", type=float, default=0.0,
                         help="Probability of a feature being dropped out at the last layer.")
    parser.add_argument("--cam_dropout", type=float, default=0.0,
                         help="Probability of a spatial location being dropped at before the final pooling.")
    # misc
    parser.add_argument("--fast_dev_run", type=int, default=False,
                         help="Do a test run with x batches.")
    parser.add_argument("--heavy_log", type=int, default=10,
                         help="""Only log some metrics/images every heavy_log epochs.
                         Improves speed prevents logs from containing to many images.""")
    parser.add_argument("--cheat_dice", action="store_true",
                         help="""Whether to cheat the Dice scores during logging. 
                         This uses the ground truth fibrosis segmentations, but downsampled
                         and optionally restricted to predicted myocardium segmentations. 
                         Can give an indication of the maximum achievable Dice score.""")

    args = parser.parse_args()

    train_model(args)
