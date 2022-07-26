
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

from train_classifier_3d import load_dataset, init_model
from preprocessing import denormalize_transform, uncrop
from densecrf import densecrf

from make_and_evaluate_cams import dice_score, merge_cams, otsu_mask_cam, cam_to_prob

MYO_TEST_IDS = ["0075", "0172", "0338", "0380", "0411", "0435", "0507", "0567", "0634", "0642", "0673", "1017", "1042", "1166", "1199"]
ONLY_MYO_TEST = False
ONLY_CORRECT  = False

DENORM = denormalize_transform()


def visualize_cam(img, myo_seg, fibrosis_seg, cam, prediction, dice, probs=None, pseudo=None, pseudo_dice=None):
    # reformat depth to batch dimension from 3D to batched 2D
    B, C, D, H, W = img.shape
    img = img.view(B*D, C, H, W)
    myo_seg = myo_seg.view(B*D, C, H, W)
    fibrosis_seg = fibrosis_seg.view(B*D, C, H, W)
    cam = cam.view(B*D, C, H, W)
    if probs != None:
        probs = probs.view(B*D, C, H, W)
    if pseudo != None:
        pseudo = pseudo.view(B*D, C, H, W)



    grid_img = torchvision.utils.make_grid(img, nrow=4, normalize=True, padding=2).detach().cpu()
    grid_myo = torchvision.utils.make_grid(myo_seg, nrow=4, padding=2).cpu().detach()
    grid_myo = np.ma.masked_where(grid_myo <= 0, grid_myo)

    grid_fibrosis = torchvision.utils.make_grid(fibrosis_seg, nrow=4, padding=2).cpu().detach()
    grid_fibrosis = np.ma.masked_where(grid_fibrosis <= 0, grid_fibrosis)

    grid_cam = torchvision.utils.make_grid(cam, nrow=4, padding=2).cpu().detach()

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
    axs[1].imshow(grid_fibrosis[0,:,:], cmap="coolwarm", alpha=1.0, vmin=0, vmax=1)
    axs[1].set_title("Myo prediction and fibrosis ground truth")

    axs[2].imshow(grid_img[0,:,:], cmap="gray")
    axs[2].imshow(grid_cam[0,:,:], cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)
    axs[2].set_title("CAM")

    if pseudo != None:
        grid_probs = torchvision.utils.make_grid(probs, nrow=4, padding=2).cpu().detach()
        grid_probs = np.ma.masked_where(grid_probs <= 0, grid_probs)
        axs[3].imshow(grid_img[0,:,:], cmap="gray")
        axs[3].imshow(grid_probs[0,:,:], cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)
        axs[3].set_title("Probs for dCRF")

        grid_pseudo = torchvision.utils.make_grid(pseudo, nrow=4, padding=2).cpu().detach()
        grid_pseudo = np.ma.masked_where(grid_pseudo <= 0, grid_pseudo)
        axs[4].imshow(grid_img[0,:,:], cmap="gray")
        axs[4].imshow(grid_pseudo[0,:,:], cmap="coolwarm", alpha=1.0, vmin=0, vmax=1)
        axs[4].set_title("CAM+dCRF")

    for i in range(plots):
        axs[i].set_axis_off()


    plt.show()
    return


def evaluate_batch(batch, model, confidence_weighting=False,
                    threshold=0.3, iters=0, crf_params=None,
                    visualize=False, select_values="pos_only",
                    upsampling="conv_transpose", otsu_mask=True,
                    otsu_cam_threshold=0.0, cam_prob_binarize=False,
                    compute_no_gt=False, merge_cam=True,
                    ambiguous_threshold=0.1,
                    myo_fullsize_mask=False,
                    no_myo_mask=False,
                    cheat_dice=False):
    """ Only works for batch size 1!"""
    img = batch["img"]
    weak_label = batch["label"]
    myo_seg = batch['myo_seg']
    fibrosis_seg_label = batch["fibrosis_seg_label"]
    # remove padded slices
    labeled = ~weak_label.isnan().view(-1)
    img = img[:,:,labeled]
    myo_seg = myo_seg[:,:,labeled]
    fibrosis_seg_label = fibrosis_seg_label[:,:,labeled]
    # convert slice-level weak label to stack-level weak label
    weak_label = weak_label.nan_to_num().amax(dim=-1, keepdim=False)

    # compute dice if label available
    if fibrosis_seg_label.amax() > 0.01 or (compute_no_gt == True and weak_label == 1):

        # get prediction value
        pred = model.output2pred(model.forward(img, myo_seg)).view(-1)

        # make cam
        if merge_cam == True:
            max_translation = 8
        else:
            max_translation = 1

        if no_myo_mask == True:
            myo_seg_input = None
        else:
            myo_seg_input = myo_seg

        if cheat_dice == True:
            cheat_gt = fibrosis_seg_label
        else:
            cheat_gt = None


        cam = merge_cams(model, img, myo_seg=myo_seg_input,
                            select_values=select_values, upsampling=upsampling,
                            max_translation=max_translation, translation_step=1,
                            fullsize_mask=myo_fullsize_mask, cheat_gt=cheat_gt).detach()

        if ONLY_CORRECT == True and pred < 0.5:
            return {}, {}, torch.zeros_like(img)

        if confidence_weighting == True:
            cam = cam * pred

        pseudo = cam.clone()
        if otsu_mask == True:
            pseudo = otsu_mask_cam(pseudo, img, otsu_cam_threshold)

        if iters > 0:
            # denseCRF
            denorm_img = DENORM(img)
            img_int8 = (255*img).type(torch.uint8)
            probs = torch.zeros_like(pseudo)
            for s in range(pseudo.shape[2]):
                probs[:,:,s] = cam_to_prob(pseudo[:,:,s], cam_threshold=threshold, binarize=cam_prob_binarize, ambiguous_threshold=ambiguous_threshold)
                pseudo[:,:,s] = densecrf(img_int8[0,:,s], probs[0,:,s], crf_params)

        pseudo_metrics = dice_score(pseudo, fibrosis_seg_label, threshold)
        pseudo_dice = pseudo_metrics['dice']

        metrics = dice_score(cam, fibrosis_seg_label, threshold)
        dice = metrics['dice']

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
    metric_sums = {}
    metric_sums['cam'] = {"dice" : 0, "precision" : 0, "recall" : 0}
    metric_sums['pseudo'] = {"dice" : 0, "precision" : 0, "recall" : 0}
    total = 0

    crf_params = (hparams.w1, hparams.alpha, hparams.beta, hparams.w2, hparams.gamma, hparams.iters)

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
                                        cheat_dice=hparams.cheat_dice_posttraining)
        # track metrics
        if len(metrics) > 0:
            total += 1
        for m in metrics:
            metric_sums['cam'][m] += metrics[m]
        for m in pseudo_metrics:
            metric_sums['pseudo'][m] += pseudo_metrics[m]

        # save pseudo labels
        if hparams.save == True:
            B, C, D, H, W = pseudo.shape
            pseudo = pseudo.view(C, B*D, H, W)
            pseudo = uncrop(pseudo, batch['crop_corners'], batch['original_shape'], is_mask=True)
            output_file = str(out_dir.joinpath(Path(img_path).stem)) + "_pseudo-label.nrrd"
            C, D, H, W = pseudo.shape
            pseudo = sitk.JoinSeries([sitk.GetImageFromArray(pseudo_slice, isVector=False) for pseudo_slice in pseudo])

            origin = [x.item() for x in batch['origin']]
            spacing = [x.item() for x in batch['spacing']]
            pseudo.SetOrigin(origin)
            pseudo.SetSpacing(spacing)

            compress = True
            sitk.WriteImage(pseudo, output_file, compress)


    return {k: {m : sum /total for m, sum in d.items()} for k, d in metric_sums.items() if total > 0}



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
    assert hparams.classification_level == "3D"

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
    if hparams.change_myo_dilation != None:
        model.myo_mask_dilation = hparams.change_myo_dilation

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
    parser.add_argument("--cheat_dice_posttraining", action='store_true')
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
    parser.add_argument("--myoseg_path", type=str, default=r"myocard_predictions/deeprisk_myocardium_predictions")
    parser.add_argument("--seg_labels_dir", type=str, default=r"fibrosis_labels_n=117")
    parser.add_argument("--out_dir", type=str, default=r"pseudolabels")
    # model
    parser.add_argument("--model", type=str, default='drnd', choices=['drnd', 'drnd22', 'drnd24'])
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
    parser.add_argument("--upsampling", type=str, default="trilinear", choices=["no", "trilinear"])
    parser.add_argument("--no_myo_mask", action='store_true')
    parser.add_argument("--myo_fullsize_mask", default=True)
    parser.add_argument("--myo_mask_threshold", default=0.4)
    parser.add_argument("--change_myo_dilation", type=int, default=None)
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
    parser.add_argument("--gamma", type=float, default=3) # std distance


    args = parser.parse_args()

    evaluate_cams(args)
