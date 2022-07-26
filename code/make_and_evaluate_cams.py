
import pytorch_lightning as pl
import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import yaml
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F
import skimage
from skimage.filters import threshold_multiotsu
import torchvision.transforms.functional as TF
import SimpleITK as sitk

from train_pl import load_dataset, init_model
from preprocessing import denormalize_transform, uncrop
from densecrf import densecrf

MYO_TEST_IDS = ["0075", "0172", "0338", "0380", "0411", "0435", "0507", "0567", "0634", "0642", "0673", "1017", "1042", "1166", "1199"]
ONLY_MYO_TEST = False
ONLY_CORRECT  = False

DENORM = denormalize_transform()

def dice_score(cam, label, threshold):
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
    assert cam.shape == label.shape, f"shapes {cam.shape} and {label.shape} don't match"
    assert cam.shape[0] == label.shape[0] == 1 # batch size 1 only
    cam, label = cam.squeeze(), label.squeeze()

    label = label > 0.0
    cam = cam >= threshold

    intersection = (label * cam).sum()
    sum_pred = cam.sum()
    sum_gt = label.sum()
    #dice = 2 * intersection / (cam.sum() + label.sum())
    #if cam.sum() == 0:
    #    precision = 0
    #else:
    #    precision = intersection / cam.sum()

    #recall = intersection / label.sum()
    #return {'dice' : dice, 'precision' : precision, 'recall' : recall}
    return {'intersection': intersection.item(), 'sum_pred' : sum_pred.item(), 'sum_gt' : sum_gt.item()}



def merge_cams(model, image, myo_seg=None, select_values="pos_only", upsampling="bilinear", max_translation=8, translation_step=2, fullsize_mask=False, cheat_gt=None):
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

            """fig, axs = plt.subplots(1, 5, figsize=(20, 6))
            fig.suptitle(f"Check merging {x=} {y=}")
            fig.colorbar(cm.ScalarMappable(cmap="coolwarm"), ax=axs.ravel().tolist())
            axs[0].imshow(shift_image[0][0], cmap="gray")
            axs[0].set_title("Reference image")
            axs[1].imshow(shift_image[0][0], cmap="gray")
            axs[1].imshow(shift_myo_seg[0][0], cmap="Blues", alpha=0.5, vmin=0, vmax=1)
            axs[1].set_title("Myo prediction")

            axs[2].imshow(shift_image[0][0], cmap="gray")
            axs[2].imshow(shift_cam[0][0].detach(), cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)
            axs[2].set_title("shifted CAM")

            axs[3].imshow(image[0][0], cmap="gray")
            axs[3].imshow(cam[0][0].detach(), cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)
            axs[3].set_title("CAM shifted back to original image")

            axs[4].imshow(shift_cam[0][0].detach(), cmap="Blues", vmin=0, vmax=1)
            axs[4].imshow(cam[0][0].detach(), cmap="Reds", alpha=0.5, vmin=0, vmax=1)
            axs[4].set_title("shifted and shifted back CAM")
            plt.show()"""

            cam_sum += cam
    merged_cam = cam_sum / (torch.amax(cam_sum, (2, 3), keepdims=True) + 1e-5)

    """fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Check merged result")
    fig.colorbar(cm.ScalarMappable(cmap="coolwarm"), ax=axs.ravel().tolist())
    axs[0].imshow(image[0][0], cmap="gray")
    axs[0].set_title("Reference image")
    axs[1].imshow(image[0][0], cmap="gray")
    axs[1].imshow(myo_seg[0][0], cmap="Blues", alpha=0.5, vmin=0, vmax=1)
    axs[1].set_title("Myo prediction")

    axs[2].imshow(image[0][0], cmap="gray")
    axs[2].imshow(merged_cam[0][0].detach(), cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)
    axs[2].set_title("CAM")
    plt.show()"""
    return merged_cam



def visualize_cam(img, myo_seg, fibrosis_seg, cam, prediction, dice, probs=None, pseudo=None, pseudo_dice=None):
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

def cam_to_prob(cam, cam_threshold=0.3, binarize=True, ambiguous_threshold=0.1, ambiguous_prob=0.5, background_prob=0):
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
    """ Only works for batch size 1!"""
    img = batch["img"]
    weak_label = batch["label"]
    myo_seg = batch['myo_seg']
    fibrosis_seg_label = batch["fibrosis_seg_label"]
    crop_corners = batch['crop_corners']
    # compute dice if label available
    if fibrosis_seg_label.amax() != 1 and fibrosis_seg_label.amax() > 0:
        print(f"{batch['img_path']}{batch['slice_idx']}:{fibrosis_seg_label.amax()}")
    if fibrosis_seg_label.amax() > 0.01 or (compute_no_gt == True and weak_label == 1):
        if no_myo_mask == True:
            myo_seg_input = None
        else:
            myo_seg_input = myo_seg

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

        pred = model.output2pred(model(img, myo_seg=myo_seg))

        if ONLY_CORRECT == True and pred < 0.5:
            return {}, {}, torch.zeros_like(img)

        if confidence_weighting == True:
            cam = cam * pred

        if otsu_mask == True:
            pseudo = cam.clone()
            pseudo = otsu_mask_cam(pseudo, img, otsu_cam_threshold)
        else:
            pseudo = cam.clone()

        if iters > 0:
            # denseCRF
            denorm_img = DENORM(img)
            img_int8 = (255*img).type(torch.uint8)
            probs = cam_to_prob(pseudo, cam_threshold=threshold, binarize=cam_prob_binarize, ambiguous_threshold=ambiguous_threshold)
            pseudo = densecrf(img_int8[0], probs[0], crf_params)

        pseudo_metrics = dice_score(pseudo, fibrosis_seg_label, threshold)
        pseudo_dice = pseudo_metrics['dice']

        metrics = dice_score(cam, fibrosis_seg_label, threshold)
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





def evaluate_setting(dataloader, model, hparams, threshold, visualize=False, out_dir=None):
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

    # save last mri pseudo labels
    if hparams.save == True:
        output_file = str(out_dir.joinpath(Path(prev_img_path).stem)) + "_pseudo-label.nrrd"
        pseudo_stack = np.concatenate(pseudo_stack, axis=0)[None, ...]
        pseudo_stack = sitk.JoinSeries([sitk.GetImageFromArray(array, isVector=False) for array in pseudo_stack])
        pseudo_stack.SetOrigin(origin)
        pseudo_stack.SetSpacing(spacing)
        compress = True
        sitk.WriteImage(pseudo_stack, output_file, compress)

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
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--skip_train", action='store_true')
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--metrics_level", default='2D', choices=['2D', '3D'])
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
    parser.add_argument("--load_checkpoint", type=str, required=True)
    parser.add_argument("--myoseg_path", type=str, default=r"myocard_predictions//deeprisk_myocardium_predictions")
    parser.add_argument("--seg_labels_dir", type=str, default=r"fibrosis_labels_n=117")
    parser.add_argument("--out_dir", type=str, default=r"pseudolabels")
    # model
    parser.add_argument("--singlestage", action='store_true')
    parser.add_argument("--model", type=str, default='drnd', choices=['convnext', 'resnet18', 'resnet50', 'resnet101', 'simple', "drnd", "drnc"])
    # learning hyperparameters
    parser.add_argument("--batch_size", type=int, default=1) # don't change
    # data augmentation
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
    # cam seeds hyperparameters
    parser.add_argument("--select_values", type=str, default="pos_only", choices=["all", "pos_only", "sigmoid"])
    parser.add_argument("--upsampling", type=str, default="bilinear", choices=["no", "bilinear", "conv_transpose"])
    parser.add_argument("--myo_fullsize_mask", default=True)
    parser.add_argument("--myo_mask_threshold", default=0.4)
    parser.add_argument("--no_myo_mask", action='store_true')
    parser.add_argument("--thresholds", nargs="*", type=float, default=[1.0])
    parser.add_argument("--confidence_weighting", action="store_true")
    parser.add_argument("--otsu_mask", action="store_true")
    parser.add_argument("--otsu_cam_threshold", type=float, default=0.0)
    parser.add_argument("--ambiguous_threshold", type=float, default=0.0)
    parser.add_argument("--merge_cam", action="store_true")
    parser.add_argument("--iters", type=int, default=5) # denseCRF iterations, zero for no denseCRF
    parser.add_argument("--w1", type=float, default=0) # appearance weight (distance +intensity)
    parser.add_argument("--alpha", type=float, default=3) # std distance
    parser.add_argument("--beta", type=float, default=3) # std intensity
    parser.add_argument("--w2", type=float, default=1) # smoothness weight (only distance)
    parser.add_argument("--gamma", type=float, default=3) # std intensity

    args = parser.parse_args()

    evaluate_cams(args)
