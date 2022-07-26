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

from preprocessing import compose_transforms, compose_transforms_with_segmentations
from datasets import DeepRiskDatasetMyoSegmentation2D, DeepRiskDatasetSegmentation3D
from models.segmentation_models.pl_base_myosegmentation_model import MyoSegmentationModel

def load_dataset(hparams):
    #transforms
    hparams.no_roi_crop = True


    if hparams.dataset == 'deeprisk':
        # paths
        DATA_DIR = Path(hparams.data_path)
        IMG_DIR = DATA_DIR.joinpath(hparams.img_path)

        GT_SEG_DIR = DATA_DIR.joinpath(hparams.gt_seg_dir)
        LABELS_FILE = DATA_DIR.joinpath(hparams.weak_labels_path)

        assert DATA_DIR.exists()
        assert IMG_DIR.exists()
        assert GT_SEG_DIR.exists()
        assert LABELS_FILE.exists()

        if 'cascaded' in hparams.model:
            train_transforms = compose_transforms_with_segmentations(hparams, 'train', 3)
            val_transforms = compose_transforms_with_segmentations(hparams, 'val', 3)
            print("train transforms:")
            print(train_transforms)
            print("val_transforms:")
            print(val_transforms)

            PRED_MYO_DIR = DATA_DIR.joinpath(hparams.myoseg_path)
            assert PRED_MYO_DIR.exists()
            dataset_train = DeepRiskDatasetSegmentation3D(IMG_DIR, LABELS_FILE,
                                                            pseudo_fib_dir=None,
                                                            gt_fib_dir=None,
                                                            pred_fib_dir=None,
                                                            gt_myo_dir=GT_SEG_DIR,
                                                            pred_myo_dir=PRED_MYO_DIR,
                                                            transform=train_transforms,
                                                            split='train',
                                                            roi_crop='none',
                                                            img_type=hparams.img_type,
                                                            pixel_gt_only=True)
            dataset_val = DeepRiskDatasetSegmentation3D(IMG_DIR, LABELS_FILE,
                                                            pseudo_fib_dir=None,
                                                            gt_fib_dir=None,
                                                            pred_fib_dir=None,
                                                            gt_myo_dir=GT_SEG_DIR,
                                                            pred_myo_dir=PRED_MYO_DIR,
                                                            transform=val_transforms,
                                                            split='val',
                                                            roi_crop='none',
                                                            img_type=hparams.img_type,
                                                            pixel_gt_only=True)
            dataset_test = DeepRiskDatasetSegmentation3D(IMG_DIR, LABELS_FILE,
                                                            pseudo_fib_dir=None,
                                                            gt_fib_dir=None,
                                                            pred_fib_dir=None,
                                                            gt_myo_dir=GT_SEG_DIR,
                                                            pred_myo_dir=PRED_MYO_DIR,
                                                            transform=val_transforms,
                                                            split='test',
                                                            roi_crop='none',
                                                            img_type=hparams.img_type,
                                                            pixel_gt_only=True)
        else:
            train_transforms = compose_transforms_with_segmentations(hparams, "train", 2)
            val_transforms = compose_transforms_with_segmentations(hparams, "val", 2)

            print("train transforms:")
            print(train_transforms)
            print("val_transforms:")
            print(val_transforms)

            dataset_train = DeepRiskDatasetMyoSegmentation2D(IMG_DIR, GT_SEG_DIR,
                                                transform=train_transforms, split="train",
                                                train_frac = hparams.train_frac,
                                                split_seed=hparams.splitseed,
                                                include_no_myo=hparams.include_no_myo,
                                                img_type=hparams.img_type)

            dataset_val = DeepRiskDatasetMyoSegmentation2D(IMG_DIR, GT_SEG_DIR,
                                                transform=val_transforms, split="val",
                                                train_frac = hparams.train_frac,
                                                split_seed=hparams.splitseed,
                                                include_no_myo=hparams.include_no_myo,
                                                img_type=hparams.img_type)

            dataset_test = DeepRiskDatasetMyoSegmentation2D(IMG_DIR, GT_SEG_DIR,
                                                transform=val_transforms, split="test",
                                                train_frac = hparams.train_frac,
                                                split_seed=hparams.splitseed,
                                                include_no_myo=hparams.include_no_myo,
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
    kwargs['feature_multiplication'] = hparams.feature_multiplication

    model_class = MyoSegmentationModel
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

    # setting True gives potential speedup if input size does not change
    torch.backends.cudnn.benchmark = False

    # model & trainer
    model = init_model(hparams)
    logger = TensorBoardLogger("tb_logs_segmentation_myo", name=hparams.logdir)

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
    parser.add_argument("--dataset", type=str, default="deeprisk", choices=['deeprisk'])
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
    parser.add_argument("--load_checkpoint", type=str, default=False)
    parser.add_argument("--gt_seg_dir", type=str, default=r"myo_labels_n=117")
    parser.add_argument("--myoseg_path", type=str, default=r"myocard_predictions/deeprisk_myocardium_predictions")
    parser.add_argument("--weak_labels_path", type=str, default=r"weak_labels_n=657.xlsx")
    parser.add_argument("--logdir", type=str, default=None)
    # model
    parser.add_argument('--model', default='Floor_UNet2D', type=str,
                        help='What model to use for the segmentation',
                        choices=['UNet2D', 'UNet2D_stacked',
                                 'UNet2D_small', 'UNet2D_stacked_small',
                                 'CANet2D', 'CANet2D_stacked',
                                 'Floor_UNet2D', 'Floor_UNet2D_stacked',
                                 'UNet3D', 'UNet3D_channels', 'UNet3D_channels_stacked', 'UNet3D_half', 'UNet3D_cascaded'])
    parser.add_argument('--upsampling', default='convtrans', type=str,
                        help='What kind of model upsampling we want to use',
                        choices=['upsample', 'convtrans'])
    parser.add_argument('--feature_multiplication', default='2', type=int,
                        help='The factor by which the number of features in the model are is multiplied')

    # learning hyperparameters
    parser.add_argument('--loss_function', default='dice', type=str, help='What loss function to use for the segmentation', choices=['dice', 'dice+WCE'])
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--schedule", type=str, default=None, choices=["step", "cosine"])
    parser.add_argument("--warmup_iters", type=int, default=0)
    # data augmentation
    parser.add_argument("--image_norm", type=str, default="per_image", choices=["per_image", "global_statistic", "global_agnostic", "no_norm"])
    parser.add_argument("--include_no_myo", default=True)
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
    # misc
    parser.add_argument("--fast_dev_run", type=int, default=False)
    parser.add_argument("--heavy_log", type=int, default=1)




    args = parser.parse_args()

    train_model(args)
