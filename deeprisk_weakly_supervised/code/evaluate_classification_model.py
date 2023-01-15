""" Outputs a dictionary that contains fibrosis classification predictions.
 Graphs/plots/final metrics can be seen in notebooks/evaluate_segmentations.ipynb
 """
import pickle
from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from datasets import DeepRiskDatasetSegmentation3D, EmidecDataset3D
from inference_myocard_segmentation import visualize_stack
from models.drn import LightningDRN
from models.pl_classification_model_3d import ClassificationModel3D
from preprocessing import compose_inference_transforms_with_segmentations
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        # TODO add functionality for other 2D models
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
            print(f"Warning: img_type {hparams.img_type} not supported for emidec dataset, using PSIR instead.")
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
    # load data and model
    datasets = list(load_dataset(hparams))
    MODEL_DIR = Path(hparams.load_checkpoint).parent.parent
    model = load_model(hparams)

    # add predictions to dictionary
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

    # save dictionary
    OUT_FILE = MODEL_DIR.joinpath(f'{hparams.dataset}_predictions.pkl')
    if hparams.dry_run == False:
        with open(OUT_FILE, 'wb') as f_out:
            pickle.dump(output_dict, f_out)
    print("Finished calculating predictions, view results using notebooks/evaluate_segmentations.ipynb")
    return


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = ArgumentParser()

    # debugging arguments
    parser.add_argument('--dry_run', action='store_true',
                        help="Set flag to not save results.")
    parser.add_argument('--verbose', action='store_true',
                        help="Set flag to print extra debugging information.")
    parser.add_argument('--visualize', action='store_true',
                        help="Set flag to plot images and predictions.")    
    parser.add_argument('--dataset', type=str, required=True,
                         choices=['deeprisk', 'emidec'],
                         help="Select a dataset.")
    parser.add_argument("--split", type=str, default="All",
                         choices=["Train", "Val", "Test", "All"],
                         help="Dataset split to run on. Default='All' to run on all splits.")
    parser.add_argument('--img_type', type=str, default="PSIR", choices=["PSIR", "MAG"],
                        help="""For deeprisk, which type of LGE images to use.""")
    parser.add_argument('--pixel_gt_only', action='store_true',
                        help="""Set flag to only compute predictions
                        for images with ground truth segmentation labels.""")
    parser.add_argument("--num_workers", type=int, default=0,
                         help="Number of workers for dataloaders.")
    # paths
    parser.add_argument("--data_dir", type=str, default=r"../data",
                         help="Path to directory containing all data.")
    parser.add_argument("--img_dir", type=str, default=r"all_niftis_n=657",
                         help="Relative path from data_path to image directory.")
    parser.add_argument("--weak_labels_file", type=str, default=r"weak_labels_n=657.xlsx",
                         help="Relative path from data_path to excel sheet with weak labels.")
    parser.add_argument("--load_checkpoint", type=str, required=True,
                         help="""Full path + filename of model checkpoint.
                          When False, train from scratch.""")
    parser.add_argument("--pred_myo_dir", type=str,
                         default=r"myocard_predictions/deeprisk_myocardium_predictions",
                         help="""Relative path from data_path to directory with
                          predicted myocardium segmentations.""")
    parser.add_argument("--gt_fib_dir", type=str, default=r"fibrosis_labels_n=117",
                         help="""Relative path from data_path to directory with
                         ground truth fibrosis segmentations.""")
    parser.add_argument('--gt_myo_dir', type=str, default=r"myo_labels_n=117",
                        help="""Relative path from data_dir to directory with
                        ground truth myocardium segmentations.""")
    # preprocessing / data augmentation hyperparameters
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
    parser.add_argument("--rotate", type=int, default=0,
                         help="Possible degrees of rotation for data augmentation.")
    parser.add_argument("--translate", nargs=2, type=float, default=(0, 0),
                         help="Possible translation range in x and y direction for data augmentation, as a factor of image size.")
    parser.add_argument("--scale", nargs=2, type=float, default=(1, 1),
                         help="Possible scaling factor range in x and y direction for data augmentation.")
    parser.add_argument("--shear", nargs=4, type=int, default=(0, 0, 0, 0),
                         help="Shearing parameters for data augmentation.")
    parser.add_argument("--randomaffine_prob", type=float, default=0.0,
                         help="Probability of doing a randomaffine data augmentation (rotation+scale+translation+shear).")
    parser.add_argument("--brightness", type=float, default=0.0,
                         help="Brightness parameter for random colorjitter data augmentation.")
    parser.add_argument("--contrast", type=float, default=0.0,
                         help="Contrast parameter for random colorjitter data augmentation")
    parser.add_argument("--hflip", type=float, default=0.0,
                         help="Probability of random horizontal flip data augmentation")
    parser.add_argument("--vflip", type=float, default=0.0,
                         help="Probability of random vertical flip data augmentation.")
    parser.add_argument("--randomcrop", type=int, default=False,
                         help="Size of randomcrop data augmentation. Set to False for no random crop.")
    parser.add_argument("--randomcrop_prob", type=float, default=0.0,
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
    
    args = parser.parse_args()
    main(args)
