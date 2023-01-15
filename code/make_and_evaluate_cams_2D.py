""" 
Used to generate fibrosis 'pseudo-labels' based on:
    -   A trained slice-level fibrosis classification model
    -   Per-slice fibrosis labels
    -   (Predicted) Myocardium segmentations
    
These fibrosis pseudo-labels can then be used to train a fibrosis segmentation model.      
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import SimpleITK as sitk
import skimage
import torch
import torchvision
import yaml
from densecrf import densecrf
from preprocessing import uncrop
from skimage.filters import threshold_multiotsu
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_classifier_2d import init_model, load_dataset

# set to true to only evaluate correctly predicted image slices
ONLY_CORRECT  = False  


def calculate_metrics(cam, label, threshold):
    """Calculate Dice score, precision and recall between Class Activation Map and ground truth label,
    with an threshold applied to the CAM.

    Returns dictionary with results.
    """
    assert cam.shape == label.shape, f"shapes {cam.shape} and {label.shape} don't match"
    assert cam.shape[0] == label.shape[0] == 1 # batch size 1 only
    cam, label = cam.squeeze(), label.squeeze()

    label = label > 0.0
    cam = cam >= threshold

    intersection = (label * cam).sum()
    dice = 2 * intersection / (cam.sum() + label.sum())
    if cam.sum() == 0:
        precision = 0
    else:
        precision = intersection / cam.sum()

    recall = intersection / label.sum()
    return {'dice' : dice, 'precision' : precision, 'recall' : recall}


def overlap_stats(cam, label, threshold):
    """Calculate intersection, sum of CAM and sum of label between Class Activation Map and ground truth label,
    with an threshold applied to the CAM. (For a single slice)
    Later, these statistics can be merged from different slices, making it possible to compute the
    Dice, precision and recall of a stack of slices.  

    Returns dictionary with results.
    """
    assert cam.shape == label.shape, f"shapes {cam.shape} and {label.shape} don't match"
    assert cam.shape[0] == label.shape[0] == 1 # batch size 1 only
    cam, label = cam.squeeze(), label.squeeze()

    label = label > 0.0
    cam = cam >= threshold

    intersection = (label * cam).sum()
    sum_pred = cam.sum()
    sum_gt = label.sum()
    return {'intersection': intersection.item(), 'sum_pred' : sum_pred.item(), 'sum_gt' : sum_gt.item()}


def merge_cams(model, image, myo_seg=None, select_values="pos_only", upsampling="bilinear",
                 max_translation=8, translation_step=2, fullsize_mask=False, cheat_gt=None):
    """ Using a classification model, extract Class Activation Maps to multiple shifted input images
    and merge them.
    The purpose is to get smoother CAMs, which can have blocking artefacts due to low resolution
    otherwise.

    Inputs:
        -   model:      Trained classification model, which should have the model.make_cam() function.
        -   image:      A (batched) series of images, whatever shape model requires (..., W, H)
        -   myo_seg:    (Predicted) Myocardium segmentation, whatever shape model requires (..., W, H)
        -   max_translation:    Maximal translation in both x and y direction.
        -   translation_step:   Difference in translation between each shifted image.
        -   other arguments:    select_values/upsampling/cheat_gt, see pl_classification_model_2d.py/pl_classification_model_3d.py
                                for more details
    """
    def _shift_image(image, t_x, t_y):
        image = image.clone()
        # shift
        shift_image  = torch.roll(image, shifts=(t_x, t_y), dims=(-2, -1))
        # replace rolled over values with zero padding
        if t_x >= 0:
            shift_image[...,:t_x,:] = 0
        else:
            shift_image[...,t_x:,:] = 0

        if t_y >= 0:
            shift_image[...,:,:t_y] = 0
        else:
            shift_image[...,:,t_y:] = 0
        return shift_image

    cam_sum = torch.zeros_like(image)
    for x in range(0, max_translation, translation_step):
        for y in range(0, max_translation, translation_step):
            shift_image = _shift_image(image, x, y)
            if myo_seg != None:
                shift_myo_seg = _shift_image(myo_seg, x, y)
            else:
                shift_myo_seg = myo_seg

            shift_cam = model.make_cam(shift_image, select_values=select_values,
                                    upsampling=upsampling,
                                    myo_seg=shift_myo_seg,
                                    fullsize_mask=fullsize_mask,
                                    cheat_gt=cheat_gt)

            cam = _shift_image(shift_cam, -x, -y)
            cam_sum += cam
    merged_cam = cam_sum / (torch.amax(cam_sum, (2, 3), keepdims=True) + 1e-5)
    return merged_cam



def visualize_cam(img, myo_seg, fibrosis_seg, cam, prediction, dice,
                  probs=None, pseudo=None, pseudo_dice=None):
    """For (batched) image slices:
    Plot the image, myocardium segmentation, (Ground truth) fibrosis segmentation,
     Clas Activation Map, (Optional) Probabilities as input for dCRF,
    (Optional) Refined pseudo-gt fibrosis, Dice scores, classification model prediction
    """
    grid_img = torchvision.utils.make_grid(img, nrow=1, normalize=True, padding=2).detach().cpu()
    
    grid_myo = torchvision.utils.make_grid(myo_seg, nrow=1, padding=2).cpu().detach()
    grid_myo = np.ma.masked_where(grid_myo <= 0, grid_myo)

    grid_fibrosis = torchvision.utils.make_grid(fibrosis_seg, nrow=1, padding=2).cpu().detach()
    grid_fibrosis = np.ma.masked_where(grid_fibrosis <= 0, grid_fibrosis)

    grid_cam = torchvision.utils.make_grid(cam, nrow=1, padding=2).cpu().detach()

    plots = 3 if pseudo == None else 5
    fig, axs = plt.subplots(1, plots, figsize=(20, 6))

    title_s = f"Prediction:{prediction.item():.2f}, Dice:{dice.item():.2f}"
    if pseudo_dice != None:
        title_s += f", Pseudo dice:{pseudo_dice.item():.2f}"
    fig.suptitle(title_s)

    fig.colorbar(cm.ScalarMappable(cmap="coolwarm"), ax=axs.ravel().tolist())
    axs[0].imshow(grid_img[0,:,:], cmap="gray")
    axs[0].set_title("Reference image")

    axs[1].imshow(grid_img[0,:,:], cmap="gray")
    axs[1].imshow(grid_myo[0,:,:], cmap="Blues", alpha=0.5, vmin=0, vmax=1)
    axs[1].imshow(grid_fibrosis[0,:,:], cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axs[1].set_title("Myo prediction and fibrosis ground truth")

    axs[2].imshow(grid_img[0,:,:], cmap="gray")
    axs[2].imshow(grid_cam[0,:,:], cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)
    axs[2].set_title("CAM")

    if pseudo != None:
        grid_probs = torchvision.utils.make_grid(probs, nrow=1, padding=2).cpu().detach()
        grid_probs = np.ma.masked_where(probs <= 0, probs)
        axs[3].imshow(grid_img[0,:,:], cmap="gray")
        axs[3].imshow(grid_probs.squeeze(), cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)
        axs[3].set_title("Probs for dCRF")

        grid_pseudo = torchvision.utils.make_grid(pseudo, nrow=1, padding=2).cpu().detach()
        grid_pseudo = np.ma.masked_where(grid_pseudo <= 0, grid_pseudo)
        axs[4].imshow(grid_img[0,:,:], cmap="gray")
        axs[4].imshow(grid_pseudo[0,:,:], cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)
        axs[4].set_title("CAM+dCRF")

    for i in range(plots):
        axs[i].set_axis_off()

    plt.show()
    return


def cam_to_prob(cam, cam_threshold=0.3, binarize=True,
                ambiguous_threshold=0.1, ambiguous_prob=0.5,
                background_prob=0):
    """ Converts a Class Activation Map into probabilities, such that 
    they can serve as input for dCRF.
    Inputs:
        -   cam:                    Class Activation Map
        -   cam_threshold:          CAM value above which probabilities get set to 1.
                                    Default=0.3
        -   ambiguous_threshold:    CAM value above which probabilities are non-zero.
                                    Default=0.5.
        -   ambiguous_prob:         See binarize
        -   binarize:               If False, find probabilities of CAM values between
                                    ambiguous_threshold and cam_threshold by 
                                    interpolation.
                                    If True, set probabilities of CAM values between
                                    ambiguous_threshold and cam_threshold to
                                    ambiguous_prob.
                                    Default=True
        -   background_prob:        The minimum probability. Default=0.
    """
    # in cam, all positive values indicate prob > 0.5
    prob = (cam >= cam_threshold).type_as(cam)
    if binarize == False:
        prob += (cam > ambiguous_threshold).type_as(cam) * (cam < cam_threshold).type_as(cam) * (0.5 + (cam - ambiguous_threshold) / (2 * cam_threshold))
    else:
        if ambiguous_prob > 0:
            ambiguous = ((cam > ambiguous_threshold) * (cam < cam_threshold)).type_as(cam)
            prob += ambiguous * ambiguous_prob

    if background_prob > 0:
        background = (cam <= 0).type_as(cam)
        prob += background * background_prob
    return prob.detach()


def otsu_mask_cam(cam, image, cam_threshold=0.0):
    """ Perform a 3 class multi-otsu thresholding on a Class Activation Map,
    which will separate 3 intensity classes with 2 thresholds. Pixel values below
    the lowest threshold is likely not fibrosis but healthy myocardium,
    because we know that on LGE MRI fibrosis is higher intensity.

    Determine multi-otsu threshold by looking only at pixels where the CAM is 
    higher than cam_threshold.
    For example, the CAM might have been masked using the myocardium segmentations,
    in which case the zero-value CAM pixels correspond to pixels outside of the
    myocardium. We obtain better results by only doing multi-otsu thresholding
    within the myocardium, since there are fewer intermingling tissues with
    intermingling intensities. We only want to look at healthy myocardium vs fibrosis.
    (and remove healthy myocardium)
    """
    # select threshold within cam region
    # can't multiotsu on empty prediction
    if cam.amax() < 0.01:
        return cam

    hist, bin_centers = skimage.exposure.histogram(image[cam > cam_threshold].detach().numpy())
    thresholds = threshold_multiotsu(hist=(hist, bin_centers), classes=3)
    # remove lower than bottom threshold from cam (likely healthy myocardium)
    cam[image < thresholds[0]] = 0
    return cam



def evaluate_batch(batch, model, confidence_weighting=False,
                    threshold=0.3, iters=0, crf_params=None,
                    visualize=False, select_values="pos_only",
                    upsampling="conv_transpose", otsu_mask=True,
                    otsu_cam_threshold=0.0, cam_prob_binarize=False,
                    compute_no_gt=False, merge_cam=True,
                    ambiguous_threshold=0.1,
                    myo_fullsize_mask=False,
                    no_myo_mask=False,
                    metrics_level='2D'):
    """ Generate a fibrosis pseudo-label and calculate metrics.
    Only works for batch size 1!
    """
    img = batch["img"]
    weak_label = batch["label"]
    myo_seg = batch['myo_seg']
    fibrosis_seg_label = batch["fibrosis_seg_label"]
    crop_corners = batch['crop_corners']

    if fibrosis_seg_label.amax() != 1 and fibrosis_seg_label.amax() > 0:
        print("Something weird might be going on...")
        print(f"{batch['img_path']}{batch['slice_idx']}:{fibrosis_seg_label.amax()}")
    
    # only compute dice and pseudolabel if slice has fibrosis
    if fibrosis_seg_label.amax() > 0.01 or (compute_no_gt == True and weak_label == 1):
        # set myocardium input
        if no_myo_mask == True:
            myo_seg_input = None
        else:
            myo_seg_input = myo_seg
        
        # make cam
        if merge_cam == True:
            cam = merge_cams(model, img, myo_seg=myo_seg_input,
                                select_values=select_values, upsampling=upsampling,
                                max_translation=8, translation_step=1, fullsize_mask=myo_fullsize_mask).detach()
        else:
            cam = model.make_cam(img, select_values=select_values,
                                    upsampling=upsampling,
                                    myo_seg=myo_seg_input,
                                    fullsize_mask=myo_fullsize_mask).detach()
            cam = cam / (torch.amax(cam, (2, 3), keepdims=True) + 1e-5)

        # get classification prediction value
        pred = model.output2pred(model(img, myo_seg=myo_seg))

        if ONLY_CORRECT == True and pred < 0.5:
            return {}, {}, torch.zeros_like(img)

        if confidence_weighting == True:
            # weight cam with model prediction
            cam = cam * pred

        if otsu_mask == True:
            pseudo = cam.clone()
            pseudo = otsu_mask_cam(pseudo, img, otsu_cam_threshold)
        else:
            pseudo = cam.clone()

        if iters > 0:
            # denseCRF
            img_int8 = (255*img).type(torch.uint8)
            probs = cam_to_prob(pseudo, cam_threshold=threshold, binarize=cam_prob_binarize, ambiguous_threshold=ambiguous_threshold)
            pseudo = densecrf(img_int8[0], probs[0], crf_params)

        pseudo_metrics = calculate_metrics(pseudo, fibrosis_seg_label, threshold)
        pseudo_dice = pseudo_metrics['dice']

        metrics = calculate_metrics(cam, fibrosis_seg_label, threshold)
        dice = metrics['dice']

        if metrics_level == '3D':
            pseudo_metrics = overlap_stats(pseudo, fibrosis_seg_label, threshold)
            metrics = overlap_stats(cam, fibrosis_seg_label, threshold)

        if visualize == True:
            if iters > 0:
                visualize_cam(img, myo_seg, fibrosis_seg_label, cam, pred, dice, probs, pseudo, pseudo_dice)
            else:
                visualize_cam(img, myo_seg, fibrosis_seg_label, cam, pred, dice)

    else:
        metrics, pseudo_metrics = {}, {}
        pseudo = torch.zeros_like(img)

    return metrics, pseudo_metrics, pseudo





def evaluate_setting(dataloader, model, hparams, threshold, out_dir=None):
    """ Calculate metrics and save results for a certain CAM threshold."""
    metric_collector = {}
    if hparams.metrics_level == '2D':
        metric_collector['cam'] = {"dice" : 0, "precision" : 0, "recall" : 0}
        metric_collector['pseudo'] = {"dice" : 0, "precision" : 0, "recall" : 0}
    elif hparams.metrics_level == '3D':
        metric_collector['cam'] = {}
        metric_collector['pseudo'] = {}
    total = 0

    crf_params = (hparams.w1, hparams.alpha, hparams.beta, hparams.w2, hparams.gamma, hparams.iters)

    prev_img_path = None
    pseudo_stack = []
    for batch in tqdm(dataloader):
        img_path = batch["img_path"][0]
        metrics, pseudo_metrics, pseudo = evaluate_batch(batch, model,
                                        confidence_weighting=hparams.confidence_weighting,
                                        threshold=threshold, iters=hparams.iters,
                                        crf_params =crf_params, select_values=hparams.select_values,
                                        upsampling=hparams.upsampling, visualize=hparams.visualize,
                                        otsu_mask=hparams.otsu_mask, otsu_cam_threshold=hparams.otsu_cam_threshold,
                                        merge_cam=hparams.merge_cam, compute_no_gt=hparams.save,
                                        ambiguous_threshold=hparams.ambiguous_threshold,
                                        myo_fullsize_mask=hparams.myo_fullsize_mask,
                                        no_myo_mask=hparams.no_myo_mask,
                                        metrics_level=hparams.metrics_level)
        # track metrics
        if len(metrics) > 0:
            total += 1
        if hparams.metrics_level == '2D':
            for m in metrics:
                metric_collector['cam'][m] += metrics[m]
            for m in pseudo_metrics:
                metric_collector['pseudo'][m] += pseudo_metrics[m]
        elif hparams.metrics_level == '3D':
            for cam_type, metric_type in zip(['cam', 'pseudo'], [metrics, pseudo_metrics]):
                if len(metric_type) == 0:
                    continue
                if img_path not in metric_collector[cam_type]:
                    metric_collector[cam_type][img_path] = {}
                    for m in metric_type:
                        metric_collector[cam_type][img_path][m] = []
                for m in metric_type:
                    metric_collector[cam_type][img_path][m].append(metric_type[m])

        # save pseudo labels
        if hparams.save == True:
            # wait until pseudolabels of one mri are collected
            collected_stack = (prev_img_path != None and prev_img_path != img_path)
            if collected_stack:
                output_file = str(out_dir.joinpath(Path(prev_img_path).stem)) + "_pseudo-label.nrrd"
                pseudo_stack = np.concatenate(pseudo_stack, axis=0)[None, ...]
                pseudo_stack = sitk.JoinSeries([sitk.GetImageFromArray(array, isVector=False) for array in pseudo_stack])
                pseudo_stack.SetOrigin(origin)
                pseudo_stack.SetSpacing(spacing)
                compress = True
                sitk.WriteImage(pseudo_stack, output_file, compress)
                # reset to empty stack
                pseudo_stack = []

            prev_img_path = img_path
            # reshape pseudo to original image shape
            pseudo = uncrop(pseudo, batch['crop_corners'], batch['original_shape'], is_mask=True)
            pseudo_stack.append(pseudo)
            origin = [x.item() for x in batch['origin']]
            spacing = [x.item() for x in batch['spacing']]

    # save last mri pseudo labels (won't happen in loop, since there is no next image)
    if hparams.save == True:
        output_file = str(out_dir.joinpath(Path(prev_img_path).stem)) + "_pseudo-label.nrrd"
        pseudo_stack = np.concatenate(pseudo_stack, axis=0)[None, ...]
        pseudo_stack = sitk.JoinSeries([sitk.GetImageFromArray(array, isVector=False) for array in pseudo_stack])
        pseudo_stack.SetOrigin(origin)
        pseudo_stack.SetSpacing(spacing)
        compress = True
        sitk.WriteImage(pseudo_stack, output_file, compress)

    # calculate metric averages
    if hparams.metrics_level == '2D':
        result = {k: {m : sum /total for m, sum in d.items()} for k, d in metric_collector.items() if total > 0}
    elif hparams.metrics_level == '3D':
        result = {}
        for cam_type in metric_collector:
            total = 0
            dice_sum = 0
            precision_sum = 0
            recall_sum = 0
            for img_path in metric_collector[cam_type]:
                intersections = np.array(metric_collector[cam_type][img_path]['intersection'])
                sum_preds = np.array(metric_collector[cam_type][img_path]['sum_pred'])
                sum_gts = np.array(metric_collector[cam_type][img_path]['sum_gt'])

                if len(sum_gts) > 0:
                    total += 1
                    dice = 2 * intersections.sum() / (sum_preds.sum() + sum_gts.sum())
                    if sum_preds.sum() == 0:
                        precision = 0
                    else:
                        precision = intersections.sum() / sum_preds.sum()
                    recall = intersections.sum() / sum_gts.sum()

                    dice_sum += dice
                    precision_sum += precision
                    recall_sum += recall
                else:
                    continue
            result[cam_type] = {}
            result[cam_type]['dice'] = dice_sum / total
            result[cam_type]['precision'] = precision_sum / total
            result[cam_type]['recall'] = recall_sum / total

    return result


def evaluate_cams(hparams):
    """ Generate and evaluate CAMs for different data splits and CAM thresholds."""
    assert hparams.batch_size == 1
    pl.seed_everything(hparams.trainseed, workers=True)
    # path names
    MODEL_DIR = Path(args.load_checkpoint).parent.parent
    OUT_DIR = MODEL_DIR.joinpath(hparams.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # add hparams from saved model
    with open(MODEL_DIR.joinpath("hparams.yaml"), 'r') as stream:
        parsed_yaml = yaml.safe_load(stream)
        for k, v in parsed_yaml.items():
            setattr(hparams, k, v)
    # backward compatibility
    if 'myo_input' not in hparams:
        setattr(hparams, 'myo_input', False)

    # prepare dataloaders
    dataset_train, dataset_val, dataset_test = load_dataset(hparams)

    train_loader = DataLoader(dataset_train,
                            batch_size=hparams.batch_size,
                            shuffle=False,
                            drop_last=False,
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

    # load model
    model = init_model(hparams)
    model.eval()
    if hparams.skip_train == True:
        split_names, loaders = ["Val", "Test"], [val_loader, test_loader]
    else:
        split_names, loaders = ["Train", "Val", "Test"], [train_loader, val_loader, test_loader]

    for split, data_loader in zip(split_names, loaders):
        for threshold in hparams.thresholds:
            metrics = evaluate_setting(data_loader, model, hparams, threshold, visualize=hparams.visualize, out_dir=OUT_DIR)
            print(f"{split}/{threshold=} : {metrics=}")

    return



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--visualize", action='store_true',
                        help="Plot fibrosis Pseudo-labels.")
    parser.add_argument("--skip_train", action='store_true',
                        help="Set flag to only use validation and test set.")
    parser.add_argument("--save", action='store_true',
                        help="Set flag to save the fibrosis pseudolabels.")
    parser.add_argument("--metrics_level", default='2D', choices=['2D', '3D'],
                        help="""Whether to compute metrics (Dice, precision,
                        recall) on a per-slice (2D) or per-stack (3D) level.
                        Default=2D.""")
    # choose dataset
    parser.add_argument("--dataset", type=str, default="deeprisk",
                        choices=['deeprisk', 'cifar10'],
                        help="Dataset to use.")
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
    parser.add_argument("--load_checkpoint", type=str, required=True,
                         help="Full path + filename of model checkpoint.")
    parser.add_argument("--myoseg_path", type=str, default=r"myocard_predictions/deeprisk_myocardium_predictions",
                         help="Relative path from data_path to directory with predicted myocardium segmentations.")
    parser.add_argument("--seg_labels_dir", type=str, default=r"fibrosis_labels_n=117",
                         help="Relative path from data_path to directory with ground truth fibrosis segmentations.")
    parser.add_argument("--out_dir", type=str, default=r"pseudolabels",
                        help="Output directory for fibrosis pseudolabels.")
    # model
    parser.add_argument("--singlestage", action='store_true',
                         help="Whether to use single-stage weak supervision, which didn't end up working (Feel free to ignore).")
    parser.add_argument("--model", type=str, default='drnd', 
                        choices=['convnext', 'resnet18', 'resnet50', 'resnet101', 'simple', 'drnc', 'drnc26', 'drnd', 'drnd22', 'drnd24', 'resnet38', 'vgg16'],
                        help="Model architecture to use.")
    # learning hyperparameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Don't change, code can currently only handle batch size 1.")
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
    # cam seeds hyperparameters
    parser.add_argument("--select_values", type=str, default="pos_only",
                        choices=["all", "pos_only", "sigmoid"],
                        help="""Whether to apply ReLU on Class Activation Maps
                        (pos_only), apply a Sigmoid (sigmoid) or leave values as is (all).
                        The CAM will be scaled to range(0,1) after this.""")
    parser.add_argument("--upsampling", type=str, default="bilinear",
                         choices=["no", "bilinear", "conv_transpose"],
                         help="Type of upsampling to use to bring CAMs to image resolution.")
    parser.add_argument("--myo_fullsize_mask", default=True,
                        help="""Set to True to do the masking out of CAMs
                        AFTER CAMs are upsampled.""")
    parser.add_argument("--myo_mask_threshold", default=0.4,
                        help="Threshold on mycardium prediction to generate mask for CAMs.")
    parser.add_argument("--no_myo_mask", action='store_true',
                        help="Set flag to NOT use myocardium mask on CAMs.")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[1.0],
                        help="List of CAMs thresholds to calculate metrics for.")
    parser.add_argument("--confidence_weighting", action="store_true",
                        help="""Set flag to, after having scaled CAMs in range(0,1),
                        multiply CAMs by the prediction of the classification model.""")
    parser.add_argument("--otsu_mask", action="store_true",
                        help="Set flag to use otsu thresholding (Recommended true)")
    parser.add_argument("--otsu_cam_threshold", type=float, default=0.0,
                        help="Perform otsu thresholding only where CAM > threshold.")
    parser.add_argument("--ambiguous_threshold", type=float, default=0.0,
                        help="CAM threshold for non-zero probabilities")
    parser.add_argument("--merge_cam", action="store_true",
                        help="""Set True to merge CAMs of several slightly
                        translated input images. Can improve CAM smoothness if CAM
                        resolution is low, but is also slower to run. """)
    parser.add_argument("--iters", type=int, default=5,
                        help="Number of iterations for denseCRF, 0 to skip.")
    parser.add_argument("--w1", type=float, default=0,
                        help="dCRF appearance weight (distance + intensity)")
    parser.add_argument("--alpha", type=float, default=3,
                        help="w1 standard deviation for distance.")
    parser.add_argument("--beta", type=float, default=3,
                        help="w1 standard deviation for intensity.")
    parser.add_argument("--w2", type=float, default=1,
                        help="dCRF smoothness weight (distance)")
    parser.add_argument("--gamma", type=float, default=3,
                        help="w2 standard deviation.")

    args = parser.parse_args()

    evaluate_cams(args)
