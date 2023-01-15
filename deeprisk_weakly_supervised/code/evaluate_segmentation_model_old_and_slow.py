""" This was an earlier method of evaluation segmentation models,
which (as opposed to evaluate_segmentation_model.py) did not
require having done inference beforehand. However, because of
this it is also pretty slow in comparison."""
import json
import pickle
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import skimage.metrics
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_fib_segmentation import init_model, load_dataset


def load_model(hparams):
    # load params from checkpoint
    MODEL_DIR = Path(hparams.load_checkpoint).parent.parent
    with open(MODEL_DIR.joinpath("hparams.yaml"), 'r') as stream:
        parsed_yaml = yaml.safe_load(stream)
        for k, v in parsed_yaml.items():
            if k == 'bilinear':
                k, v = ("upsampling", "upsample") if v == True else ("upsampling", "convtrans")
            elif k == "loss_function_string":
                k = "loss_function"
            elif k == "model_name":
                k = 'model'
            setattr(hparams, k, v)

    # load model itself
    model = init_model(hparams)
    model.eval()
    return model



def calc_intersection(a : torch.Tensor, b : torch.Tensor):
    assert a.shape == b.shape
    return (a * b).sum().item()

def calc_cdice_c_denom(pred: torch.Tensor, target : torch.Tensor):
    assert pred.shape == target.shape
    return ((pred > 0) * target).sum().item()


def calc_sum(a : torch.Tensor):
    return a.sum().item()




def calc_circumference(a : torch.Tensor):
    a  = a.clone().float()
    edge_mask = F.max_pool2d(a, kernel_size=3, stride=1, padding=3//2) - a
    return edge_mask.sum().item()



def calc_thickness(a : torch.Tensor):
    a = a.clone().float()
    thickness = 0
    is_empty = (a.sum() == 0)
    while not is_empty:
        thickness += 1
        # erosion implemented as minpool, minpool is maxpool on negative mask
        a = -F.max_pool2d(-a, kernel_size=3, stride=1, padding=3//2)
        is_empty = (a.sum() == 0)
    return thickness



def calc_hausdorff(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape
    dist = skimage.metrics.hausdorff_distance(a.cpu().numpy(), b.cpu().numpy())
    return dist



def calc_contact(a: torch.Tensor, b : torch.Tensor):
    a, b = a.clone().float(), b.clone().float()
    edge_mask = F.max_pool2d(b, kernel_size=3, stride=1, padding=3//2) - b
    contact_mask = edge_mask * a
    return contact_mask.sum().item()



def evaluate_batch(model, batch, metric_dict, split, pred_threshold=0.5, gt_threshold=0.0):
    # unpack batch
    image = batch['img']
    pseudoseg = batch['pseudoseg']
    pred_myo = batch['myo_seg']
    gt_fib = batch['gt_seg']
    gt_myo = batch['gt_myo_seg']
    crop_corners = batch['crop_corners']
    spacing = batch['spacing']
    img_path = batch['img_path']

    # inference
    if 'stacked' in model.model_name:
        input = torch.stack([image.squeeze(1), pred_myo.squeeze(1)], dim=1)
    else:
        input = image
    pred_fib, attention_coefs = model.forward(input.float())

    # colculate & store metrics for each slice
    for i in range(len(img_path)):
        mri_name = str(Path(img_path[i]).stem)
        # first time for a patient
        if mri_name not in metric_dict:
            metric_dict[mri_name] = {}
            # set per-stack attributes
            metric_dict[mri_name]['Split'] = split
            metric_dict[mri_name]['Original_spacing'] = [s[0].item() for s in spacing]
            resize_factor = (crop_corners[1][0] - crop_corners[0][0]) / image.shape[-1]
            metric_dict[mri_name]['Resampled_spacing'] = [(spacing[0][0]*resize_factor).item(), (spacing[1][0]*resize_factor).item(), spacing[2][0].item(), spacing[3][0].item()]
            # set empty lists for per-image attributes
            metric_dict[mri_name]['Intersection_fib'] = []
            metric_dict[mri_name]['Intersection_myo'] = []
            metric_dict[mri_name]['Continuous_intersection_fib'] = []
            metric_dict[mri_name]['Continuous_intersection_myo'] = []
            metric_dict[mri_name]['Continuous_dice_c_denom_fib'] = []
            metric_dict[mri_name]['Continuous_dice_c_denom_myo'] = []

            metric_dict[mri_name]['Sum_gt_fib'] = []
            metric_dict[mri_name]['Sum_gt_myo'] = []
            metric_dict[mri_name]['Sum_pred_fib'] = []
            metric_dict[mri_name]['Sum_pred_myo'] = []

            metric_dict[mri_name]['Continuous_sum_gt_fib'] = []
            metric_dict[mri_name]['Continuous_sum_gt_myo'] = []
            metric_dict[mri_name]['Continuous_sum_pred_fib'] = []
            metric_dict[mri_name]['Continuous_sum_pred_myo'] = []

            metric_dict[mri_name]['Circumference_gt_fib'] = []
            metric_dict[mri_name]['Circumference_gt_myo'] = []
            metric_dict[mri_name]['Circumference_pred_fib'] = []
            metric_dict[mri_name]['Circumference_pred_myo'] = []

            metric_dict[mri_name]['Thickness_gt_fib'] = []
            metric_dict[mri_name]['Thickness_gt_myo'] = []
            metric_dict[mri_name]['Thickness_pred_fib'] = []
            metric_dict[mri_name]['Thickness_pred_myo'] = []

            metric_dict[mri_name]['Hausdorff_fib'] = []
            metric_dict[mri_name]['Hausdorff_myo'] = []

            metric_dict[mri_name]['Contact_gt_myo_fib'] = []
            metric_dict[mri_name]['Contact_pred_myo_fib'] = []

            metric_dict[mri_name]['Ground_truth_segmentations_available'] = False

        metric_dict[mri_name]['Continuous_sum_pred_fib'].append(calc_sum(pred_fib[i]))
        metric_dict[mri_name]['Continuous_sum_pred_myo'].append(calc_sum(pred_myo[i]))
        # binarize predictions
        pred_fib_i = pred_fib[i] > pred_threshold
        pred_myo_i = pred_myo[i] > pred_threshold

        metric_dict[mri_name]['Sum_pred_fib'].append(calc_sum(pred_fib_i))
        metric_dict[mri_name]['Sum_pred_myo'].append(calc_sum(pred_myo_i))

        metric_dict[mri_name]['Circumference_pred_fib'].append(calc_circumference(pred_fib_i))
        metric_dict[mri_name]['Circumference_pred_myo'].append(calc_circumference(pred_myo_i))

        metric_dict[mri_name]['Thickness_pred_fib'].append(calc_thickness(pred_fib_i))
        metric_dict[mri_name]['Thickness_pred_myo'].append(calc_thickness(pred_myo_i))

        metric_dict[mri_name]['Contact_pred_myo_fib'].append(calc_contact(pred_myo_i, pred_fib_i))




        # dice -> needs valid ground truth label
        if gt_fib[i].amin() > -0.01:
            assert gt_myo[i].amin() > -0.01, "No valid ground truth myocardium."
            metric_dict[mri_name]['Ground_truth_segmentations_available'] = True
            # binarize ground truth
            gt_fib_i = gt_fib[i] > gt_threshold
            gt_myo_i = gt_myo[i] > gt_threshold

            metric_dict[mri_name]['Intersection_fib'].append(calc_intersection(pred_fib_i, gt_fib_i))
            metric_dict[mri_name]['Intersection_myo'].append(calc_intersection(pred_myo_i, gt_myo_i))

            metric_dict[mri_name]['Continuous_intersection_fib'].append(calc_intersection(pred_fib[i], gt_fib_i))
            metric_dict[mri_name]['Continuous_intersection_myo'].append(calc_intersection(pred_myo[i], gt_myo_i))

            metric_dict[mri_name]['Continuous_dice_c_denom_fib'].append(calc_cdice_c_denom(pred_fib[i], gt_fib_i))
            metric_dict[mri_name]['Continuous_dice_c_denom_myo'].append(calc_cdice_c_denom(pred_myo[i], gt_myo_i))

            metric_dict[mri_name]['Sum_gt_fib'].append(calc_sum(gt_fib_i))
            metric_dict[mri_name]['Sum_gt_myo'].append(calc_sum(gt_myo_i))

            metric_dict[mri_name]['Continuous_sum_gt_fib'].append(calc_sum(gt_fib[i]))
            metric_dict[mri_name]['Continuous_sum_gt_myo'].append(calc_sum(gt_myo[i]))

            metric_dict[mri_name]['Circumference_gt_fib'].append(calc_circumference(gt_fib_i))
            metric_dict[mri_name]['Circumference_gt_myo'].append(calc_circumference(gt_myo_i))

            metric_dict[mri_name]['Thickness_gt_fib'].append(calc_thickness(gt_fib_i))
            metric_dict[mri_name]['Thickness_gt_myo'].append(calc_thickness(gt_myo_i))

            metric_dict[mri_name]['Hausdorff_fib'].append(calc_hausdorff(pred_fib_i, gt_fib_i))
            metric_dict[mri_name]['Hausdorff_myo'].append(calc_hausdorff(pred_myo_i, gt_myo_i))

            metric_dict[mri_name]['Contact_gt_myo_fib'].append(calc_contact(gt_myo_i, gt_fib_i))

        # other metrics can be calculated later using these metrics, e.g.:
        #       2D/ 3D Dice using Intersection & Sum
        #       Scar burden using Sum myo and Sum fib
        #       Patchiness related attempts using Sum/Circumference/Contact
        #       Transmurality using Thickness myo & Thickness fib
    return metric_dict


def evaluate_model(hparams):
    pl.seed_everything(hparams.trainseed, workers=True)
    # prepare dataloaders
    dataset_train, dataset_val, dataset_test = load_dataset(hparams)

    model = load_model(hparams)
    for pred_threshold in hparams.pred_thresholds:
        master_dict = {}
        for split, dataset in zip(['Train', 'Val', 'Test'], [dataset_train, dataset_val, dataset_test]):
            if hparams.split == 'All' or hparams.split == split:
                loader = DataLoader(dataset,
                                    batch_size=hparams.batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=hparams.num_workers)

                for batch in tqdm(loader):
                    master_dict = evaluate_batch(model, batch, master_dict, split=split, pred_threshold=pred_threshold)

        MODEL_DIR = Path(hparams.load_checkpoint).parent.parent
        OUT_FILE = MODEL_DIR.joinpath(f'metrics@{pred_threshold:.2f}.pkl')
        with open(OUT_FILE, 'wb') as f_out:
            # pickle can't handle lambda, so convert defaultdict to normal dict
            master_dict = json.loads(json.dumps(master_dict))
            # save dict as pickle
            pickle.dump(master_dict, f_out)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pred_thresholds", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # choose dataset
    parser.add_argument("--dataset", type=str, default="deeprisk", choices=['deeprisk'])
    parser.add_argument("--img_type", type=str, default='MAG', choices=['PSIR', 'MAG'])
    # hardware
    parser.add_argument("--gpus", default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    # reproducability
    parser.add_argument("--trainseed", type=int, default=42)
    parser.add_argument("--splitseed", type=int, default=84)
    parser.add_argument("--train_frac", type=float, default=0.6) # fraction of pixel-labeled data used for trainset after removing testset (as of 07/03 corresponds to train/val/test 241/21/20)
    parser.add_argument("--split", type=str, default="Val", choices=["Train", "Val", "Test", "All"])
    # paths (base data dir and relative paths to other dirs)
    parser.add_argument("--data_path", type=str, default=r"../data")
    parser.add_argument("--img_path", type=str, default=r"all_niftis_n=657")
    parser.add_argument("--weak_labels_path", type=str, default=r"weak_labels_n=657.xlsx")
    parser.add_argument("--load_checkpoint", type=str, required=True)

    parser.add_argument("--pseudoseg_path", type=str, default=r"tb_logs_classification_2D/drnd_myo_input/version_1/pseudolabels")
    parser.add_argument("--myoseg_path", type=str, default=r"myocard_predictions/unet_myo_version_2") #r"nnUnet_results\nnUNet\2d\Task500_MyocardSegmentation\predictions"
    parser.add_argument("--gt_seg_dir", type=str, default=r"fibrosis_labels_n=117")
    parser.add_argument("--gt_myoseg_dir", type=str, default=r"myo_labels_n=117")
    # model
    parser.add_argument('--model', default='UNet2D_stacked', type=str,
                        help='What model to use for the segmentation',
                        choices=['UNet2D', 'UNet2D_stacked',
                                 'UNet2D_small', 'UNet2D_stacked_small',
                                 'CANet2D', 'CANet2D_stacked',
                                 'Floor_UNet2D', 'Floor_UNet2D_stacked',
                                 'UNet3D', 'UNet3D_channels', 'UNet3D_channels_stacked', 'UNet3D_half'])
    parser.add_argument('--upsampling', default='upsample', type=str,
                        help='What kind of model upsampling we want to use',
                        choices=['upsample', 'convtrans'])
    parser.add_argument('--feature_multiplication', default='4', type=int,
                        help='The factor by which the number of features in the model are is multiplied')

    # learning hyperparameters
    parser.add_argument('--loss_function', default='dice', type=str, help='What loss function to use for the segmentation', choices=['dice', 'dice+WCE'])
    parser.add_argument('--underfitting_warmup', type=int, default=None)
    parser.add_argument('--underfitting_k', default=0.5)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--schedule", type=str, default=None, choices=["step", "cosine"])
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--train_with_gt", action='store_true')
    parser.add_argument("--freeze_n", type=int, default=None)
    parser.add_argument("--train_bn", action='store_true')
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
    parser.add_argument("--randomaffine_prob", type=float, default=0.0)
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=0.0)
    parser.add_argument("--hflip", type=float, default=0.0)
    parser.add_argument("--vflip", type=float, default=0.0)
    parser.add_argument("--randomcrop", type=int, default=False) # random crop size, taken from the resized image
    parser.add_argument("--randomcrop_prob", type=float, default=1.0)
    parser.add_argument("--randomerasing_scale", nargs=2, type=float, default=(0.02, 0.33))
    parser.add_argument("--randomerasing_ratio", nargs=2, type=float, default=(0.3, 3.3))
    parser.add_argument("--randomerasing_probs", nargs="+", type=float, default=[]) # set multiple probs for multiple randomerasing transforms
    # misc
    parser.add_argument("--fast_dev_run", type=int, default=False)
    parser.add_argument("--heavy_log", type=int, default=10)




    args = parser.parse_args()

    evaluate_model(args)
