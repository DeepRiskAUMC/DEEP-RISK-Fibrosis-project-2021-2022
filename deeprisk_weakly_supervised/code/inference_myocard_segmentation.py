""" Runs myocardium segmentation inference on some data.
For this you need at least a trained model and some (nifti format) data to do 
inference on. Ideally data is similar to the training data, so beware of
differences between MAG and PSIR data, which of these the model was
trained on, and what the data you're trying to run inference on is.

IDEA: maybe it's possible to convert data yourself from MAG to PSIR
or from PSIR to MAG, so that you can convert your data to the type
that the model was trained on.
Other possiblity would be to train models on both MAG and PSIR variants
of data, but that would obviously require retraining models.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torchvision
import torchvision.transforms.functional as TF
import yaml
from preprocessing import CropToLargestSquare, normalize_image
from torchvision import transforms
from tqdm import tqdm
from train_myo_segmentation import init_model


def load_model(hparams):
    # load params from checkpoint
    MODEL_DIR = Path(hparams.load_checkpoint).parent.parent
    with open(MODEL_DIR.joinpath("hparams.yaml"), 'r') as stream:
        parsed_yaml = yaml.safe_load(stream)
        for k, v in parsed_yaml.items():
            # convert some parameters that have an unlucky/bad name
            # really, this shouldn't be necessary if it was made well
            if k == 'bilinear':
                k, v = ("upsampling", "upsample") if v == True else ("upsampling", "convtrans")
            elif k == "loss_function_string":
                k = "loss_function"
            elif k == "model_name":
                k = 'model'
            setattr(hparams, k, v)

    setattr(hparams, 'dataset', 'deeprisk')
    # load model itself
    model = init_model(hparams)
    model.eval()
    return model



def inference_transforms(args):
    """Create transforms for inference, in order: crop -> resize -> normalize"""
    transforms_list = []
    if args.crop_shortest == True:
        transforms_list.append(CropToLargestSquare())
    if args.center_crop != None:
        transforms_list.append(transforms.CenterCrop((args.center_crop, args.center_crop)))
    if args.resize != None:
        transforms_list.append(transforms.Resize((args.resize, args.resize)))

    if args.image_norm == "per_image":
        transforms_list.append(normalize_image())
    elif args.image_norm == "global_agnostic" and args.dataset == 'deeprisk':
        transforms_list.append(transforms.Normalize((0.5), (0.5)))
    elif args.image_norm == "global_statistic" and args.dataset == 'deeprisk':
        transforms_list.append(transforms.Normalize(mean=[.57], std=[.06]))
    elif args.image_norm == "no_norm":
        pass
    else:
        raise ValueError
    return transforms.Compose(transforms_list)



def visualize_stack(img, *segmentations):
    """ Visualize the images and their segmentations for debugging purposes.
    number of arguments for segmentations is variable, each type will be shown
    in a different column, e.g. both myocardium and fibrosis segmentations.

    img and segmentations should be either (D, C, H, W) or (C, D, H, W),
    with number of channels C=1, which will be converted to (D, C, H, W).
    """
    #C, D, H, W = img.shape
    #img = img.view(D, C, H, W)
    #myo_seg = myo_seg.view(D, C, H, W)
    print(f"{img.shape=}")
    print(f"{img.mean()=}")
    img = img.squeeze().unsqueeze(1) # set channel dimension at dim 1
    grid_img = torchvision.utils.make_grid(img, nrow=4, normalize=True, padding=2).detach().cpu()
    num_rows = len(segmentations)+1
    fig, axs = plt.subplots(1, num_rows, figsize=(4*num_rows, 4))
    fig.colorbar(cm.ScalarMappable(cmap='bwr'), ax=axs.ravel().tolist())
    axs[0].imshow(grid_img[0,:,:], cmap='gray')
    axs[0].set_title("LGE Image")
    for i, segmentation in enumerate(segmentations):
        segmentation = segmentation.squeeze().unsqueeze(1) # set channel dimension at dim 1
        assert img.shape == segmentation.shape, f"{img.shape=} {segmentation.shape=}"
        grid_seg = torchvision.utils.make_grid(segmentation, nrow=4, padding=2).cpu().detach()
        grid_seg = np.ma.masked_where(grid_seg <= 0, grid_seg)
        axs[i+1].imshow(grid_img[0,:,:], cmap='gray', vmin=0, vmax=1)
        axs[i+1].imshow(grid_seg[0,:,:], cmap='bwr', alpha=0.8, vmin=0, vmax=1)

    for i in range(num_rows):
        axs[i].set_axis_off()
    plt.show()
    return


def infer(model, nifti_file, args):
    """ 
    Input:      trained segmentation model, mri (nifti) filename, dict with args
    Returns:    sitk image of predicted myocardium segmentation. Equal size to original
                nifti, confidence values between 0 and 1
    """
    # load nifti
    nifti_image = sitk.ReadImage(str(nifti_file))
    if args.verbose:
        print(f"Succesfully read {str(nifti_file)}")
    origin = nifti_image.GetOrigin()
    spacing = nifti_image.GetSpacing()
    nifti_array = sitk.GetArrayFromImage(nifti_image)
    original_shape = nifti_array.shape
    # make up time dimension spacing if not provided,
    # just to be consistent (even though LGE MRI is for one time only)
    if len(spacing) == 3 and len(original_shape) == 4:
        spacing = spacing + spacing[-1]
    if args.verbose:
        print(f'{original_shape=}')

    # preprocessing
    if len(nifti_array.shape) == 3:
        nifti_array = nifti_array[None, ...]
    # from (C, D, H, W) to (D, C, H, W)
    # so depth in batch dim, which makes sense for a 2D model
    nifti_tensor = torch.Tensor(nifti_array.astype('float')).permute(1, 0, 2, 3)
    pre_crop_nifti = nifti_tensor.clone()
    if args.verbose:
        print(f'{pre_crop_nifti.shape=}')
    # transform image, for example cropping, resizing, normalization
    nifti_tensor = args.transform(nifti_tensor)
    if args.verbose:
        print(f"Succesfully transformed")
    if args.verbose:
        print(f'{nifti_tensor.shape=}')

    # apply model
    myo_pred, _ = model(nifti_tensor)

    # now, start transforming prediction back to original shape
    # for which we need to do the reverse transforms in reverse order

    # resize back
    if args.resize != None and args.center_crop != None:
        myo_pred = TF.resize(myo_pred, size=(args.center_crop, args.center_crop))
    elif args.resize != None and args.crop_shortest == True:
        shortest_side = min(original_shape[-1], original_shape[-2])
        myo_pred = TF.resize(myo_pred, size=(shortest_side, shortest_side))

    # uncrop
    # a much clearer and commented uncrop() function can be found in preprocessing.py,
    # but some code would have to be added to create the correct input parameters here. 
    if args.center_crop != None or args.crop_shortest == True:
        og_center_x, og_center_y = pre_crop_nifti.shape[-2] // 2, pre_crop_nifti.shape[-1] // 2
        if args.center_crop != None:
            crop_size = args.center_crop
        elif args.crop_shortest == True:
            crop_size = min(pre_crop_nifti.shape[-2],  pre_crop_nifti.shape[-1])
        crop_center_x, crop_center_y = crop_size // 2, crop_size // 2
        # dont add padded crop values when uncropping
        crop_x_min = max(0, crop_center_x - og_center_x)
        crop_x_max =  min(og_center_x + crop_center_x, crop_size)
        crop_y_min = max(0, crop_center_y - og_center_y)
        crop_y_max =  min(og_center_y + crop_center_y, crop_size)

        og_x_min = crop_x_min - crop_center_x + og_center_x
        og_x_max = crop_x_max - crop_center_x + og_center_x
        og_y_min = crop_y_min - crop_center_y + og_center_y
        og_y_max = crop_y_max - crop_center_y + og_center_y
        temp = torch.zeros(size=pre_crop_nifti.shape)
        temp[..., og_x_min:og_x_max, og_y_min:og_y_max] = myo_pred[..., crop_x_min:crop_x_max, crop_y_min:crop_y_max]
        myo_pred = temp
        nifti_tensor = pre_crop_nifti

    if args.verbose:
        print(f'{pre_crop_nifti.shape=}, {myo_pred.shape=}')
        print(f'{len(spacing)=}, {spacing=}')
    if args.visualize == True:
        visualize_stack(nifti_tensor, myo_pred)

    # reformat to original shape and type
    myo_pred = myo_pred.permute(1, 0, 2, 3)
    if len(spacing) == 3:
        myo_pred = myo_pred.squeeze(0)
    myo_pred = myo_pred.detach().numpy()
    if args.verbose:
        print(f'{myo_pred.shape=}')


    assert myo_pred.shape == original_shape

    # create sitk image
    myo_pred_image = sitk.GetImageFromArray(myo_pred, isVector=False)
    myo_pred_image.SetOrigin(origin)
    myo_pred_image.SetSpacing(spacing)
    return myo_pred_image



def main(args):
    assert not ((args.resize != None) and args.center_crop == None and args.crop_shortest == False)
    # setup & check paths
    IN_DIR = Path(args.input_path)
    assert IN_DIR.exists()
    OUT_DIR = Path(args.output_path)
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CKPT = Path(args.load_checkpoint)
    assert MODEL_CKPT.exists()

    # load model
    model = load_model(args)
    args.transform = inference_transforms(args)

    # load data & inference & save
    nifti_files = [x for x in IN_DIR.glob(f'*{args.files_must_contain}*.{args.input_extension}')]
    for nifti_file in tqdm(nifti_files):
        output_file = OUT_DIR.joinpath(f'{Path(nifti_file.stem).stem}{args.suffix}.nrrd') # double stem for '.nii.gz' extension
        try:
            output_nifti = infer(model, nifti_file, args)
            # write image
            if args.dry_run == False:
                compress = True
                sitk.WriteImage(output_nifti, str(output_file), compress)
        except Exception as e:
            print('Something went wrong for', nifti_file)
            print(e)

    return



if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = ArgumentParser()

    # debugging arguments
    parser.add_argument('--dry_run', action='store_true',
                        help="Don't save results. Output_path still required, sorry.")
    parser.add_argument('--verbose', action='store_true',
                        help="Print extra stuff for debugging, such as image shapes")
    parser.add_argument('--visualize', action='store_true',
                        help="""Plot segmentations of each sequence.
                        Alternatively, you can open images and segmentations in something like MITK.""")
    # paths
    parser.add_argument('--load_checkpoint', type=str, required=True,
                        help="""Path pointing to trained model checkpoint. 
                        Pytorch lightning uses extension .ckpt by default, but can be anything.""")
    parser.add_argument('--input_path', type=str, required=True,
                        help="Path to directory where your input data is.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to directory where your output data will be.")
    # input & output filename details
    parser.add_argument('--files_must_contain', type=str, default="MAG",
                        help="""Can be used to filter data in the input_path.
                        For example in deep risk data, 'MAG' to only select MAG data, not PSIR.
                        Can also be used to filter through subfolder structures with wildcard '*'
                        For example, emidec images are in 'input_path\Casexyz\Images\Casexyz.nii.gz',
                        which can be found with '--files_must_contain *\Images\*' .
                        No need to flattern the input directory to run inference!
                        (although output will be flat, maybe add an option for that)""")
    parser.add_argument('--input_extension', type=str, default='mha',
                         choices=['mha', 'nrrd', 'nii', 'nii.gz'],
                        help="""Input extension of files. Listed choices might not be extensive,
                        everything that sitk.ReadImage() accepts should be fine.""")
    parser.add_argument('--suffix', type=str, default='_myo_pred',
                        help="""Suffix to add to output predictions. Use something different
                         if you dont want to overwrite earlier predictions.""")

    # preprocessing hyperparameters
    # for cropping, pick either no crop (default), or --crop_shortest,
    # or a size n crop with --center_crop n. They can't be combined.
    parser.add_argument('--crop_shortest', action='store_true',
                        help="Make a square center crop, where crop size is min(height, width).")
    parser.add_argument('--center_crop', type=int, default=None,
                        help="Make a square center crop of given size. Default=None for no crop. ")
    parser.add_argument('--resize', type=int, default=None,
                        help="Resize input images to given size. Default=None for no resizing.")
    parser.add_argument('--image_norm', default='per_image', type=str,
                         choices=['per_image', 'global_agnostic', 'global_statistic', 'no_norm'],
                         help="Type of image normalization. See preprocessing.py for details.")


    args = parser.parse_args()
    main(args)
