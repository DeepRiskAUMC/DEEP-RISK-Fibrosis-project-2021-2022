import torch
from pathlib import Path
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import yaml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pytorch_lightning as pl
from collections import defaultdict
import pickle
import skimage.metrics
import json
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from datasets import DeepRiskDatasetSegmentation3D, EmidecDataset3D
from models.drn import LightningDRN
from models.pl_classification_model_3d import ClassificationModel3D
from preprocessing import get_array_from_nifti, compose_inference_transforms_with_segmentations
from inference_myocard_segmentation import visualize_stack


def load_model(hparams):
    # load params from checkpoint
    MODEL_DIR = Path(hparams.load_checkpoint).parent.parent
    kwargs = {}
    with open(MODEL_DIR.joinpath("hparams.yaml"), 'r') as stream:
        parsed_yaml = yaml.safe_load(stream)
        for k, v in parsed_yaml.items():
            setattr(hparams, k, v)
            kwargs[k] = v

    # load model itself
    if 'classification_level' in kwargs:
        model_class = ClassificationModel3D
    else:
        model_class = LightningDRN
    model_class = model_class.load_from_checkpoint
    kwargs['checkpoint_path'] = hparams.load_checkpoint
    # load model
    print(f'model {kwargs=}')
    model= model_class(**kwargs)
    model.eval()
    return model



def load_dataset(hparams):
    num_imgs = 5 # image, gt myo, gt fib, pred myo
    transform = compose_inference_transforms_with_segmentations(image_norm=hparams.image_norm, center_crop=None,
                                                                        input_size=hparams.input_size, num_imgs=num_imgs)
    print(f"{transform=}")
    DATA_DIR = Path(hparams.data_dir)
    IMG_DIR = DATA_DIR.joinpath(hparams.img_dir)
    LABELS_FILE = DATA_DIR.joinpath(hparams.weak_labels_file)
    #PSEUDO_FIB_DIR = DATA_DIR.joinpath(hparams.pseudo_fib_dir)
    PSEUDO_FIB_DIR = None # at this point not necessary for evaluation
    GT_FIB_DIR = DATA_DIR.joinpath(hparams.gt_fib_dir)
    PRED_FIB_DIR = None
    GT_MYO_DIR = DATA_DIR.joinpath(hparams.gt_myo_dir)
    PRED_MYO_DIR = DATA_DIR.joinpath(hparams.pred_myo_dir)
    assert DATA_DIR.exists()
    assert IMG_DIR.exists()
    assert LABELS_FILE.exists()
    if hparams.dataset == 'deeprisk':
        if hparams.split != 'All':
            splits = [hparams.split.lower()] # this is stupid but now necessary
        else:
            splits = ['train', 'val', 'test']

        datasets = []
        for split in splits:
            datasets.append(DeepRiskDatasetSegmentation3D(IMG_DIR, LABELS_FILE, pseudo_fib_dir=PSEUDO_FIB_DIR,
                                                    gt_fib_dir=GT_FIB_DIR, pred_fib_dir = PRED_FIB_DIR,
                                                    gt_myo_dir=GT_MYO_DIR, pred_myo_dir = PRED_MYO_DIR,
                                                    transform=transform, split=split,
                                                    roi_crop =hparams.roi_crop, img_type=hparams.img_type,
                                                    pixel_gt_only=hparams.pixel_gt_only))
    elif hparams.dataset == 'emidec':
        if hparams.img_type != 'PSIR':
            print(f"Warning: img_type {img_type} not supported for emidec dataset, using PSIR instead.")
        dataset = EmidecDataset3D(IMG_DIR, pred_myo_dir=PRED_MYO_DIR, pred_fib_dir=PRED_FIB_DIR,
                                    transform=transform, roi_crop=hparams.roi_crop)
        datasets = [dataset]
    return datasets


def evaluate_batch(batch, model, output_dict, hparams):
    # unpack batch
    assert len(batch['img']) == 1
    image = batch['img']
    pred_myo = batch['pred_myo']
    gt_fib = batch['gt_fib']
    img_path = batch['img_path'][0]

    mri_name = str(Path(Path(img_path).stem).stem)
    if hparams.verbose:
        print(f"{image.shape=} {pred_myo.shape=}")

    # for 2D model, move slices to batch dimension
    if not isinstance(model, ClassificationModel3D):
        B, C, D, H, W = image.shape
        image = image.view(B*D, C, H, W)
        pred_myo = pred_myo.view(B*D, C, H, W)
        gt_fib = gt_fib.view(B*D, C, H, W)


    pred = model.output2pred(model.forward(image, myo_seg=pred_myo)).view(-1)
    if hparams.verbose:
        print(f"{pred=}")
        print(f"{torch.amax(gt_fib, (-2, -1)).view(-1)=}")
    if args.visualize == True:
        cam = model.make_cam(image, myo_seg=pred_myo, upsampling='bilinear')
        visualize_stack(image, gt_fib, pred_myo, cam)
    output_dict[mri_name] = pred.cpu().detach().numpy()
    return output_dict


def main(hparams):
    datasets = list(load_dataset(hparams))
    MODEL_DIR = Path(hparams.load_checkpoint).parent.parent
    model = load_model(hparams)

    output_dict = {}
    for dataset in datasets:
        if hparams.split == 'All' or hparams.split.lower() == dataset.split:
            loader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                drop_last=False,
                                num_workers=hparams.num_workers)

            for batch in tqdm(loader):
                output_dict = evaluate_batch(batch, model, output_dict, hparams)


    OUT_FILE = MODEL_DIR.joinpath(f'{hparams.dataset}_predictions.pkl')
    if hparams.dry_run == False:
        with open(OUT_FILE, 'wb') as f_out:
            # save dict as pickle
            pickle.dump(output_dict, f_out)
    print("Finished calculating predictions, view results using notebooks/evaluate_segmentations.ipynb")
    return


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = ArgumentParser()

    # debugging arguments
    parser.add_argument('--dry_run', action='store_true') # dry run does not save the results
    parser.add_argument('--verbose', action='store_true') # print shapes during conversion
    parser.add_argument('--visualize', action='store_true') # print shapes during conversion

    parser.add_argument('--dataset', type=str, required=True, choices=['deeprisk', 'emidec']) # no default to prevent accidental overwriting of dataset segmentations
    parser.add_argument("--img_type", type=str, default='PSIR', choices=['PSIR', 'MAG'])
    parser.add_argument("--split", type=str, default="All", choices=["Train", "Val", "Test", "All"])
    parser.add_argument('--pixel_gt_only', action='store_true')

    parser.add_argument("--gpus", default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    # paths
    parser.add_argument('--data_dir', type=str, default=r"../data")
    parser.add_argument('--load_checkpoint', type=str, required=True)
    parser.add_argument('--img_dir', type=str, default=r"all_niftis_n=657")
    parser.add_argument('--weak_labels_file', type=str, default=r"weak_labels_n=657.xlsx")
    parser.add_argument('--gt_fib_dir', type=str, default=r"fibrosis_labels_n=117")
    parser.add_argument('--gt_myo_dir', type=str, default=r"myo_labels_n=117")
    parser.add_argument('--pred_myo_dir', type=str, default=r"myocard_predictions/deeprisk_myocardium_predictions")


    # preprocessing / data augmentation hyperparameters
    parser.add_argument("--image_norm", type=str, default="per_image", choices=["per_image", "global_statistic", "global_agnostic"])
    parser.add_argument("--no_roi_crop", action='store_true')
    parser.add_argument("--include_no_myo", default=True)
    parser.add_argument("--roi_crop", type=str, default="fixed", choices=['fitted', 'fixed'])
    parser.add_argument("--center_crop", type=int, default=224) # only used if no roi crop
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--rotate", type=int, default=0)
    parser.add_argument("--translate", nargs=2, type=float, default=(0, 0))
    parser.add_argument("--scale", nargs=2, type=float, default=(1, 1))
    parser.add_argument("--shear", nargs=4, type=int, default=(0, 0, 0, 0))
    parser.add_argument("--randomaffine_prob", type=float, default=0.0)
    parser.add_argument("--brightness", type=float, default=0)
    parser.add_argument("--contrast", type=float, default=0)
    parser.add_argument("--hflip", type=float, default=0.0)
    parser.add_argument("--vflip", type=float, default=0.0)
    parser.add_argument("--randomcrop", type=int, default=False) # random crop size, taken from the resized image
    parser.add_argument("--randomcrop_prob", type=float, default=0.0)
    parser.add_argument("--randomerasing_scale", nargs=2, type=float, default=(0.02, 0.33))
    parser.add_argument("--randomerasing_ratio", nargs=2, type=float, default=(0.3, 3.3))
    parser.add_argument("--randomerasing_probs", nargs="+", type=float, default=[]) # set multiple probs for multiple randomerasing transforms
    parser.add_argument("--feature_dropout", type=float, default=0.0)
    parser.add_argument("--cam_dropout", type=float, default=0.0)
    args = parser.parse_args()
    main(args)
