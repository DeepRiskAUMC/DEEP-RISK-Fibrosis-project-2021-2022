""" Outputs a dictionary that contains all
metrics and/or statistics, so we can compute several
metrics based on this dictionary, without having to
 look at all the predicted and ground truth images again.
 Some other features are also calculated, which might
 by relevant for other prediction tasks?
 Graphs/plots/final metrics can be seen in
 notebooks/evaluate_segmentations.ipynb
 """

import json
import pickle
from argparse import ArgumentParser
from pathlib import Path

import skimage.metrics
import torch
import torch.nn.functional as F
from datasets import DeepRiskDatasetSegmentation3D, EmidecDataset3D
from emidec_metrics import hd, hd95
from inference_myocard_segmentation import visualize_stack
from torch.utils.data import DataLoader
from tqdm import tqdm


def calc_intersection(a : torch.Tensor, b : torch.Tensor):
    """ Calculate number of intersection pixels between two
    binary images/batches-of-images/tensors. 
    """
    assert a.shape == b.shape
    return (a * b).sum(dim=-1).sum(dim=-1).squeeze().tolist()

def calc_cdice_c_denom(pred: torch.Tensor, target : torch.Tensor):
    """ Calculate the continuous Dice denominator between continuous
    prediction and binary ground truth images/batches-of-images/tensors. 
    """
    assert pred.shape == target.shape
    return ((pred > 0) * target).sum(dim=-1).sum(dim=-1).squeeze().tolist()


def calc_sum(a : torch.Tensor):
    """ Calculate sum of images/batches-of-images/tensors. """
    return a.sum(dim=-1).sum(dim=-1).squeeze().tolist()


def calc_circumference(a : torch.Tensor):
    """ Calculate (approximate) circumference around segmentation
    using the gained area after a dilation. 
    """
    a  = a.clone().float()
    edge_mask = F.max_pool2d(a, kernel_size=3, stride=1, padding=3//2) - a
    return edge_mask.sum(dim=-1).sum(dim=-1).squeeze().tolist()



def calc_thickness(a : torch.Tensor):
    """ Calculate thickness of a segmentation:
    the number of erosions after which no positive values
    are left.  
    """
    result = []
    for i in range(a.shape[0]):
        a_i = a[i].clone().float()
        thickness = 0
        is_empty = (a_i.sum() == 0)
        while not is_empty:
            thickness += 1
            # erosion implemented as minpool, minpool is maxpool on negative mask
            a_i = -F.max_pool2d(-a_i, kernel_size=3, stride=1, padding=3//2)
            is_empty = (a_i.sum() == 0)
        result.append(thickness)
    return result



def calc_hausdorff(a: torch.Tensor, b: torch.Tensor):
    """ Calculate Hausdorff distance between two
    binary images/batches-of-images/tensors.
    (Maximum of minimum distances between two segmentations,
    i.e. what is the maximum distasnce I have to travel to go
    from one to the other.) 
    """
    assert a.shape == b.shape
    result = []
    for i in range(a.shape[0]):
        dist = skimage.metrics.hausdorff_distance(a[i].cpu().numpy(), b[i].cpu().numpy())
        result.append(dist)
    return result



def calc_contact(a: torch.Tensor, b : torch.Tensor):
    """ Calculate (approximately) how much contact there is between
    segmentations. Order matters, since b is dilated on, after which
    the overlap of the added edge with a is calculated. 
    """
    result = []
    a, b = a.clone().float(), b.clone().float()
    for i in range(a.shape[0]):
        edge_mask = F.max_pool2d(b[i], kernel_size=3, stride=1, padding=3//2) - b[i]
        contact_mask = edge_mask * a[i]
        result.append(contact_mask.sum().item())
    return result


def evaluate_batch(batch, metric_dict, split, gt_threshold=0.0, myo_threshold=None, args=None):
    """ Calculates metrics/statistics for one batch and adds to the metric dictionary."""
    # unpack batch
    assert len(batch['img']) == 1
    image = batch['img'][0]
    gt_myo = batch['gt_myo'][0]
    pred_myo = batch['pred_myo'][0]
    gt_fib = batch['gt_fib'][0]
    pred_fib = batch['pred_fib'][0]
    spacing = [s[0] for s in batch['spacing']]
    img_path = batch['img_path'][0]

    mri_name = str(Path(Path(img_path).stem).stem)
    voxelspacing = [spacing[2].item(), spacing[1].item(), spacing[0].item()]
    continuous_sum_pred_fib = calc_sum(pred_fib)
    continuous_sum_pred_myo = calc_sum(pred_myo)

    if args.visualize == True:
        print(f"{img_path=}")
        print(f"Myo Dice @0.5 = {((gt_myo > 0)*(pred_myo > 0.5)).sum() * 2 / ((gt_myo > 0).sum() + (pred_myo > 0.5).sum())}")
        visualize_stack(image, gt_myo, pred_myo, gt_fib, pred_fib)

    if gt_fib.amin() > -0.01:
        assert gt_myo.amin() > -0.01, "No valid ground truth myocardium."
        ground_truth_available = True
        gt_fib = gt_fib > gt_threshold
        gt_myo = gt_myo > gt_threshold
        # metrics that don't depend on threshold
        # continuous / using probability map
        continuous_intersection_fib = calc_intersection(pred_fib, gt_fib)
        continuous_intersection_myo = calc_intersection(pred_myo, gt_myo)
        continuous_dice_c_denom_fib = calc_cdice_c_denom(pred_fib, gt_fib)
        continuous_dice_c_denom_myo = calc_cdice_c_denom(pred_myo, gt_myo)

        # ground truth features
        continuous_sum_gt_fib = calc_sum(gt_fib)
        continuous_sum_gt_myo = calc_sum(gt_myo)
        sum_gt_fib = calc_sum(gt_fib)
        sum_gt_myo = calc_sum(gt_myo)
        circumference_gt_fib = calc_circumference(gt_fib)
        circumference_gt_myo = calc_circumference(gt_myo)
        thickness_gt_fib = calc_thickness(gt_fib)
        thickness_gt_myo = calc_thickness(gt_myo)
        contact_gt = calc_contact(gt_myo, gt_fib)
    else:
        ground_truth_available = False
        continuous_intersection_fib = []
        continuous_intersection_myo = []
        continuous_dice_c_denom_fib = []
        continuous_dice_c_denom_myo = []
        continuous_sum_gt_fib = []
        continuous_sum_gt_myo = []
        sum_gt_fib = []
        sum_gt_myo = []
        circumference_gt_fib = []
        circumference_gt_myo = []
        thickness_gt_fib = []
        thickness_gt_myo = []
        contact_gt = []

    for t in metric_dict:
        metric_dict[t][mri_name] = {}
        metric_dict[t][mri_name]['Split'] = split.capitalize()
        metric_dict[t][mri_name]['Original_spacing'] = [s.item() for s in spacing]
        metric_dict[t][mri_name]['Resampled_spacing'] = [s.item() for s in spacing]
        metric_dict[t][mri_name]['Ground_truth_segmentations_available'] = ground_truth_available
        metric_dict[t][mri_name]['Continuous_sum_pred_fib'] = continuous_sum_pred_fib
        metric_dict[t][mri_name]['Continuous_sum_pred_myo'] = continuous_sum_pred_myo
        metric_dict[t][mri_name]['Continuous_intersection_fib'] = continuous_intersection_fib
        metric_dict[t][mri_name]['Continuous_intersection_myo'] = continuous_intersection_myo
        metric_dict[t][mri_name]['Continuous_dice_c_denom_fib'] = continuous_dice_c_denom_fib
        metric_dict[t][mri_name]['Continuous_dice_c_denom_myo'] = continuous_dice_c_denom_myo
        metric_dict[t][mri_name]['Continuous_sum_gt_fib'] = continuous_sum_gt_fib
        metric_dict[t][mri_name]['Continuous_sum_gt_myo'] = continuous_sum_gt_myo
        metric_dict[t][mri_name]['Sum_gt_fib'] = sum_gt_fib
        metric_dict[t][mri_name]['Sum_gt_myo'] = sum_gt_myo
        metric_dict[t][mri_name]['Circumference_gt_fib'] = circumference_gt_fib
        metric_dict[t][mri_name]['Circumference_gt_myo'] = circumference_gt_myo
        metric_dict[t][mri_name]['Thickness_gt_fib'] = thickness_gt_fib
        metric_dict[t][mri_name]['Thickness_gt_myo'] = thickness_gt_myo
        metric_dict[t][mri_name]['Contact_gt_myo_fib'] = contact_gt
        # calculate threshold specific features
        pred_fib_t = pred_fib > float(t)
        if myo_threshold == None:
            pred_myo_t = pred_myo > float(t)
        else:
            pred_myo_t = pred_myo > myo_threshold
        metric_dict[t][mri_name]['Sum_pred_fib'] = calc_sum(pred_fib_t)
        metric_dict[t][mri_name]['Sum_pred_myo'] = calc_sum(pred_myo_t)
        metric_dict[t][mri_name]['Circumference_pred_fib'] = calc_circumference(pred_fib_t)
        metric_dict[t][mri_name]['Circumference_pred_myo'] = calc_circumference(pred_myo_t)
        metric_dict[t][mri_name]['Thickness_pred_fib'] = calc_thickness(pred_fib_t)
        metric_dict[t][mri_name]['Thickness_pred_myo'] = calc_thickness(pred_myo_t)
        metric_dict[t][mri_name]['Contact_pred_myo_fib'] = calc_contact(pred_myo_t, pred_fib_t)
        if ground_truth_available:
            metric_dict[t][mri_name]['Intersection_fib'] = calc_intersection(pred_fib_t, gt_fib)
            metric_dict[t][mri_name]['Intersection_myo'] = calc_intersection(pred_myo_t, gt_myo)
            metric_dict[t][mri_name]['Hausdorff_fib'] = calc_hausdorff(pred_fib_t, gt_fib)
            metric_dict[t][mri_name]['Hausdorff_myo'] = calc_hausdorff(pred_myo_t, gt_myo)
            if pred_fib_t.amax() > 0 and gt_fib.amax() > 0:
                metric_dict[t][mri_name]['Hausdorff_3D_fib'] = hd(pred_fib_t.squeeze().numpy(), gt_fib.squeeze().numpy(), voxelspacing=voxelspacing)
                metric_dict[t][mri_name]['Hausdorff95_3D_fib'] = hd95(pred_fib_t.squeeze().numpy(), gt_fib.squeeze().numpy(), voxelspacing=voxelspacing)
            else:
                metric_dict[t][mri_name]['Hausdorff_3D_fib'] = float('nan')
                metric_dict[t][mri_name]['Hausdorff95_3D_fib'] = float('nan')
            if pred_myo_t.amax() > 0 and gt_myo.amax() > 0:
                metric_dict[t][mri_name]['Hausdorff_3D_myo'] = hd(pred_myo_t.squeeze().numpy(), gt_myo.squeeze().numpy(), voxelspacing=voxelspacing)
                metric_dict[t][mri_name]['Hausdorff95_3D_myo'] = hd95(pred_myo_t.squeeze().numpy(), gt_myo.squeeze().numpy(), voxelspacing=voxelspacing)
            else:
                metric_dict[t][mri_name]['Hausdorff_3D_myo'] = float('nan')
                metric_dict[t][mri_name]['Hausdorff95_3D_myo'] = float('nan')
        else:
            metric_dict[t][mri_name]['Intersection_fib'] = []
            metric_dict[t][mri_name]['Intersection_myo'] = []
            metric_dict[t][mri_name]['Hausdorff_fib'] = []
            metric_dict[t][mri_name]['Hausdorff_myo'] = []
            metric_dict[t][mri_name]['Hausdorff_3D_fib'] = float('nan')
            metric_dict[t][mri_name]['Hausdorff_3D_myo'] = float('nan')
            metric_dict[t][mri_name]['Hausdorff95_3D_fib'] = float('nan')
            metric_dict[t][mri_name]['Hausdorff95_3D_myo'] = float('nan')

        # other metrics can easily be calculated later using these metrics, e.g.:
        #       smoothed/unsmoothed 2D/3D Dice using Intersection & Sum
        #       Scar burden using Sum myo and Sum fib
        #       Patchiness related attempts using Sum/Circumference/Contact
        #       Transmurality using Thickness myo & Thickness fib
    return metric_dict




def load_dataset(hparams):
    transform = None # don't use transforms for final evaluation, predictions are already made
    DATA_DIR = Path(hparams.data_dir)
    IMG_DIR = DATA_DIR.joinpath(hparams.img_dir)
    LABELS_FILE = DATA_DIR.joinpath(hparams.weak_labels_file)
    PSEUDO_FIB_DIR = None # at this point not necessary for evaluation
    GT_FIB_DIR = DATA_DIR.joinpath(hparams.gt_fib_dir)
    PRED_FIB_DIR = Path(hparams.pred_fib_dir)
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
                                                    roi_crop ='none', img_type=hparams.img_type,
                                                    pixel_gt_only=hparams.pixel_gt_only))
    elif hparams.dataset == 'emidec':
        dataset = EmidecDataset3D(IMG_DIR, pred_myo_dir=PRED_MYO_DIR, pred_fib_dir=PRED_FIB_DIR,
                                    transform=transform, roi_crop='none')
        datasets = [dataset]
    return datasets


def main(hparams):
    datasets = list(load_dataset(hparams))
    MODEL_DIR = Path(hparams.pred_fib_dir).parent

    threshold_dicts = {f'{t:.2f}':{} for t in hparams.thresholds}
    for dataset in datasets:
        if hparams.split == 'All' or hparams.split.lower() == dataset.split:
            loader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                drop_last=False,
                                num_workers=hparams.num_workers)

            for batch in tqdm(loader):
                threshold_dicts = evaluate_batch(batch, threshold_dicts, split=dataset.split, myo_threshold=hparams.myo_threshold, args=hparams)

    for t in threshold_dicts:
        OUT_FILE = MODEL_DIR.joinpath(f'{hparams.dataset}metrics@{t}.pkl')
        if hparams.dry_run == False:
            with open(OUT_FILE, 'wb') as f_out:
                # pickle can't handle lambda, so convert defaultdict to normal dict using json
                dict_t = json.loads(json.dumps(threshold_dicts[t]))
                # save dict as pickle
                pickle.dump(dict_t, f_out)
    print("Finished calculating metrics, view results using notebooks/evaluate_segmentations.ipynb")
    return



if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=0,
                         help="Number of workers for dataloaders.")
    parser.add_argument("--thresholds", nargs="*", type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Fibrosis prediction cutoffs for metrics.")
    parser.add_argument("--myo_threshold", type=float, default=0.4,
                        help="Myocardium prediction cutoff")
    # debugging arguments
    parser.add_argument('--dry_run', action='store_true',
                        help="Set flag to not save results.")
    parser.add_argument('--verbose', action='store_true',
                        help="Set flag to print extra debugging information.")
    parser.add_argument('--visualize', action='store_true',
                        help="Set flag to plot images and predictions.")
    # data  arguments
    parser.add_argument('--dataset', type=str, required=True,
                         choices=['deeprisk', 'emidec'],
                         help="Select a dataset.")
    parser.add_argument("--split", type=str, default="All",
                         choices=["Train", "Val", "Test", "All"],
                         help="Dataset split to run on. Default='All' to run on all splits.")
    parser.add_argument('--img_type', type=str, default="PSIR", choices=["PSIR", "MAG"],
                        help="""For deeprisk, which type of LGE images to use.
                        Not really relevant here, since our predictions are already made.""")
    parser.add_argument('--pixel_gt_only', action='store_true',
                        help="""Set flag to only compute metrics and stats
                        for images with ground truth segmentation labels. Otherwise,
                        several statistics are also computed for images without ground
                        truth (of course the metrics can't be computed without ground truth).""")
    # paths
    parser.add_argument("--data_dir", type=str, default=r"../data",
                         help="Path to directory containing all data.")
    parser.add_argument("--img_dir", type=str, default=r"all_niftis_n=657",
                         help="Relative path from data_dir to image directory.")
    parser.add_argument("--weak_labels_file", type=str, default=r"weak_labels_n=657.xlsx",
                         help="Relative path from data_dir to excel sheet with weak labels.")
    parser.add_argument('--gt_myo_dir', type=str, default=r"myo_labels_n=117",
                        help="Relative path from data_dir to directory with ground truth myocardium segmentations.")
    parser.add_argument("--pred_myo_dir", type=str, default=r"myocard_predictions/deeprisk_myocardium_predictions",
                         help="Relative path from data_dir to directory with predicted myocardium segmentations.")
    parser.add_argument("--gt_fib_dir", type=str, default=r"fibrosis_labels_n=117",
                         help="Relative path from data_dir to directory with ground truth fibrosis segmentations.")
    parser.add_argument('--pred_fib_dir', type=str, required=True,
                        help="Relative path from data_dir to directory with predicted fibrosis segmentations.")
    args = parser.parse_args()
    main(args)
