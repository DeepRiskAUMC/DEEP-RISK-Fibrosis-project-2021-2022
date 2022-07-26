import pytorch_lightning as pl
import torch
import numpy as np
from pathlib import Path
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from preprocessing import normalize_image, set_image_range, compose_transforms, compose_transforms_with_segmentations
from datasets import DeepRiskDataset2D
from models.resnet import LightningResNet
from models.simplenet import LightningSimpleNet
from models.convnext import LightningConvNeXt
from models.drn import LightningDRN
from models.pl_stage_net import SingleStageBase

def load_dataset(hparams):
    #transforms
    #train_transforms = compose_transforms(hparams, "train")
    #val_transforms = compose_transforms(hparams, "val")
    train_transforms = compose_transforms_with_segmentations(hparams, "train")
    val_transforms = compose_transforms_with_segmentations(hparams, "val")

    print("train transforms:")
    print(train_transforms)
    print("val_transforms:")
    print(val_transforms)

    if hparams.dataset == 'deeprisk':
        # paths
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

    # potential speedup if input size does not change
    torch.backends.cudnn.benchmark = True

    # model & trainer
    model = init_model(hparams)
    logger = TensorBoardLogger("tb_logs_classification_2D", name=hparams.logdir)

    trainer = pl.Trainer(deterministic=False,
                         gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         fast_dev_run=hparams.fast_dev_run,
                         callbacks=[ModelCheckpoint(),
                                    LearningRateMonitor("step")],
                         #default_root_dir=hparams.logdir,
                         logger=logger
                         )

    # train & test
    trainer.fit(model, train_loader, val_loader)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    return



if __name__ == "__main__":
    parser = ArgumentParser()
    # choose dataset
    parser.add_argument("--dataset", type=str, default="deeprisk", choices=['deeprisk', 'cifar10'])
    parser.add_argument("--img_type", type=str, default='PSIR', choices=['PSIR', 'MAG'])
    # hardware
    parser.add_argument("--gpus", default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    # reproducability
    parser.add_argument("--trainseed", type=int, default=42)
    parser.add_argument("--splitseed", type=int, default=84)
    parser.add_argument("--train_frac", type=float, default=0.6) # fraction of pixel-labeled data used for trainset after removing testset (as of 07/03 corresponds to train/val/test 241/21/20)
    # paths (base data dir and relative paths to other dirs)
    parser.add_argument("--data_path", type=str, default=r"../data")
    parser.add_argument("--img_path", type=str, default=r"all_niftis_n=657")
    parser.add_argument("--weak_labels_path", type=str, default=r"weak_labels_n=657.xlsx")
    parser.add_argument("--load_checkpoint", type=str, default=False)
    parser.add_argument("--myoseg_path", type=str, default=r"myocard_predictions/deeprisk_myocardium_predictions") #r"nnUnet_results\nnUNet\2d\Task500_MyocardSegmentation\predictions"
    parser.add_argument("--seg_labels_dir", type=str, default=r"fibrosis_labels_n=117")
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--ss_pre_weights_path", type=str, default=None)
    # model
    parser.add_argument("--singlestage", action='store_true')
    parser.add_argument("--model", type=str, default='drnd', choices=['convnext',
                                                                          'resnet18', 'resnet50', 'resnet101',
                                                                          'simple',
                                                                          'drnc', 'drnc26', 'drnd', 'drnd22', 'drnd24',
                                                                          'resnet38', 'vgg16'])
    parser.add_argument("--no_cam", action="store_true")   # only for resnet, convnext (mainly used to test if changes made for cam don't affect classification)
    parser.add_argument("--pretrain", action='store_true') # only for resnet
    parser.add_argument("--highres", action='store_true') # only for resnet, simplenet
    # model
    parser.add_argument("--depths", nargs="+", type=int, default=None) # None uses default depths of model
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64, 128, 128, 128, 128, 128]) # None uses default dims of model
    parser.add_argument("--strides", nargs="+", type=int, default=[4, 2, 2, 2])
    parser.add_argument("--stem", type=str, default="patch", choices=["patch", "resnet_like"]) # convnext only
    parser.add_argument("--model_norm", type=str, default="layer", choices=["layer", "batch"]) # convnext only, others always use batchnorm
    parser.add_argument("--pooling", type=str, default="avg", choices=["avg", "max", "ngwp"]) # last layer avg or max pooling
    parser.add_argument("--myo_mask_pooling", action="store_true") # whether to restrict pooling to myocardium predictions
    parser.add_argument("--myo_mask_threshold", type=float, default=0.4)
    parser.add_argument("--myo_mask_dilation", type=int, default=3) # how much to dilate the myocardium prediction mask (kernel size of max pool with stride 1)
    parser.add_argument("--myo_mask_prob", type=float, default=1)
    parser.add_argument("--downsampling", type=int, choices=[4, 8], default=4) # currently only for DRN
    parser.add_argument("--myo_input", action="store_true")
    # learning hyperparameters
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--drop_path_rate", type=float, default=0) # convnext only\
    parser.add_argument("--layer_scale_init_value", type=float, default=0) # convnext only, <= 0 for no layer scale
    parser.add_argument("--schedule", type=str, default=None, choices=["step", "cosine"])
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--ss_pretrain", type=int, default=5) # singlestage only
    parser.add_argument("--focal_lambda", type=float, default=0.01) # singlestage only
    parser.add_argument("--sg_psi", type=float, default=0.3) # singlestage only
    parser.add_argument("--aspp", action="store_true") # singlestage only
    parser.add_argument("--focal", action="store_true") # singlestage only
    # data augmentation
    parser.add_argument("--image_norm", type=str, default="per_image", choices=["per_image", "global_statistic", "global_agnostic", "no_norm"])
    parser.add_argument("--no_roi_crop", action='store_true')
    parser.add_argument("--include_no_myo", default=True)
    parser.add_argument("--roi_crop", type=str, default="fixed", choices=['fitted', 'fixed'])
    parser.add_argument("--center_crop", type=int, default=224) # only used if no roi crop
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--rotate", type=int, default=90)
    parser.add_argument("--translate", nargs=2, type=float, default=(0.1, 0.1))
    parser.add_argument("--scale", nargs=2, type=float, default=(0.8, 1.2))
    parser.add_argument("--shear", nargs=4, type=int, default=(-5, 5, -5, 5))
    parser.add_argument("--randomaffine_prob", type=float, default=0.5)
    parser.add_argument("--brightness", type=float, default=0.1)
    parser.add_argument("--contrast", type=float, default=0.1)
    parser.add_argument("--hflip", type=float, default=0.5)
    parser.add_argument("--vflip", type=float, default=0.5)
    parser.add_argument("--randomcrop", type=int, default=False) # random crop size, taken from the resized image
    parser.add_argument("--randomcrop_prob", type=float, default=1.0)
    parser.add_argument("--randomerasing_scale", nargs=2, type=float, default=(0.02, 0.33))
    parser.add_argument("--randomerasing_ratio", nargs=2, type=float, default=(0.3, 3.3))
    parser.add_argument("--randomerasing_probs", nargs="+", type=float, default=[]) # set multiple probs for multiple randomerasing transforms
    parser.add_argument("--feature_dropout", type=float, default=0.0)
    parser.add_argument("--cam_dropout", type=float, default=0.0)
    # misc
    parser.add_argument("--fast_dev_run", type=int, default=False)
    parser.add_argument("--heavy_log", type=int, default=10)
    parser.add_argument("--cheat_dice", action="store_true")




    args = parser.parse_args()

    train_model(args)
