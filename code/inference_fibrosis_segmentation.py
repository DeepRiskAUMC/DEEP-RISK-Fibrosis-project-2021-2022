""" Runs fibrosis segmentation inference on some data.
For this you need at least a trained model and some (nifti format) data to do 
inference on.
Additionally, you need myocardium predictions if either:
    -   You do a region-of-interest (roi) crop, since roi is determined using myocardium
    -   The model requires myocardium predictions as input. This should be obvious from the
        name of the model. Otherwise just try it without, and python will get mad at you if 
        it's missing.
Myocardium predictions can be obtained by running inference_myocard_segmentation.py

Ideally data is similar to the training data, so beware of
differences between MAG and PSIR data, which of these the model was
trained on, and what the data you're trying to run inference on is.

IDEA: maybe it's possible to convert data yourself from MAG to PSIR
or from PSIR to MAG, so that you can convert your data to the type
that the model was trained on.
Other possiblity would be to train models on both MAG and PSIR variants
of data, but that would obviously require retraining models.
"""
import traceback
from argparse import ArgumentParser
from pathlib import Path

import SimpleITK as sitk
import torch
import torch.utils.data.dataloader as tudl
import torchvision.transforms.functional as TF
import yaml
from datasets import process_images_3d
from inference_myocard_segmentation import visualize_stack
from preprocessing import (compose_inference_transforms_with_segmentations,
                           get_array_from_nifti, uncrop)
from tqdm import tqdm
from train_fib_segmentation import init_model


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

    # load model itself
    model = init_model(hparams)
    model.eval()
    return model

def infer(model, image_file, pred_myo_file, pred_fib_file, args):
    """ 
    Input:      
        model:          Trained segmentation model
        image_file:     MRI (nifti) filename
        pred_myo_file:  myocardium prediction filename
        pred_fib_file:  fibrosis prediction filename. For the 'cascaded' segmentation model,
                        a preliminary fibrosis prediction is given as additional input for a 3D
                        segmentation model. This didn't really improve things. Set to None when
                        model is not of 'cascaded' kind.
        args:           All the parsed arguments.

    Returns:    sitk image of predicted myocardium segmentation. Confidence values between 0 and 1
                Should be equal size/shape to original nifti.
    """
    # start collecting the different types of images and their names
    images, image_names = [], []

    # load mri image nifti (should be first in images list, for normalization)
    img_array, spacing, origin = get_array_from_nifti(image_file, with_spacing=True, with_origin=True)
    original_shape = img_array.shape
    # make up time dimension spacing if not provided 
    #(otherwise niftis not readable with SimpleITK 2.1+ ???)
    if len(spacing) == 3 and len(original_shape) == 4:
        spacing = spacing + spacing[-1]
    elif len(spacing) == 3 and len(original_shape) == 3:
        img_array = img_array[None, ...]

    if args.verbose:
        print(f"{img_array.shape=}")
    images.append(img_array)
    image_names.append('img')

    # load fibrosis prediction if available
    if pred_fib_file != None:
        pred_fib = get_array_from_nifti(pred_fib_file)
        images.append(pred_fib)
        image_names.append('pred_fib')

    # load myocardium prediction if available (should be last in images list, for roi crop)
    if pred_myo_file != None:
        pred_myo = get_array_from_nifti(pred_myo_file)
        images.append(pred_myo)
        image_names.append('pred_myo')

    # 3D models require (batch, channel, depth, height, width), batch and channel == 1
    # 2D models require (depth, channel, height, width), so we put depth in the batch dimension 
    if '3D' in model.model_name:
        depth_to_batch = False
    else:
        depth_to_batch = True

    # apply cropping, resizing, normalization
    data = process_images_3d(images, image_names, args.transform, args.roi_crop,
                             depth_to_batch=depth_to_batch)
    # pretend like we're using a torch dataloader
    data = tudl.default_collate([data])

    # format model input
    pred_myo_input = ('stacked' in model.model_name) # stacked means we need myocardium inputs
    pred_fib_input = ('cascaded' in model.model_name) # cascaded means we need fibrosis inputs
    if (not pred_myo_input) and (not pred_fib_input):
        input = data['img'][0]
    elif pred_myo_input and not pred_fib_input:
        input = torch.stack([
                data['img'][0].squeeze(1),
                data['pred_myo'][0].view_as(data['img'][0]).squeeze(1)
                ], dim=1)
    elif pred_fib_input and not pred_myo_input:
        input = torch.stack([
                data['img'][0].squeeze(1),
                data['pred_fib'][0].view_as(data['img'][0]).squeeze(1)
                ], dim=1)
    elif pred_fib_input and pred_myo_input:
        input = torch.stack([
                data['img'][0].squeeze(1),
                data['pred_myo'][0].view_as(data['img'][0]).squeeze(1),
                data['pred_fib'][0].view_as(data['img'][0]).squeeze(1)
                ], dim=1)

    if args.verbose:
        print(f"{input.shape=}")

    if args.dataset == 'emidec' and '3D' in model.model_name:
        # for emidec the slice order is inverted,
        # so flip it to agree with the (training) deeprisk data
        # this is of course not necessary if the model is 2D 
        input = input.flip(dims=(-3,))

    # apply model
    pred_fib, attention_coefs = model.forward(input.float())

    if args.dataset == 'emidec' and '3D' in model.model_name:
        # reverse emidec predictions to original slice ordering to save
        pred_fib = pred_fib.flip(dims=(-3,))

    if args.verbose:
        print(f"{pred_fib.shape=}")

    
    if args.roi_crop in ["fitted", "fixed"]:
        # uncrop and resize from roi crop        
        assert args.center_crop == None and args.crop_shortest == False, "Pick either roi crop or center crop"
        if '3D' in model.model_name:
            pred_fib = pred_fib[0] # remove batch dimension
        else:
            pred_fib = pred_fib.permute(1, 0, 2, 3) # batch dimension back to depth
        if args.verbose:
            print(f"{pred_fib.shape=}, {original_shape=}")
        pred_fib = uncrop(pred_fib, data['crop_corners'], original_shape)
    else:
        # FIXME pre_crop_nifti is missing here
        # see inferece_myocard_segmentation.py for what it should look like
        # However, we probably want to use roi_crop anyway, since all fibrosis segmentation
        # models are trained with that. In that case we don't get in this else statement.

        # resize back
        if args.resize != None and args.center_crop != None:
            pred_fib = TF.resize(pred_fib, size=(args.center_crop, args.center_crop), antialias=True)
        elif args.resize != None and args.crop_shortest == True:
            shortest_side = min(original_shape[-1], original_shape[-2])
            pred_fib = TF.resize(pred_fib, size=(shortest_side, shortest_side), antialias=True)
        # uncrop
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
            temp[..., og_x_min:og_x_max, og_y_min:og_y_max] = pred_fib[..., crop_x_min:crop_x_max, crop_y_min:crop_y_max]
            pred_fib = temp
            nifti_tensor = pre_crop_nifti

    if args.verbose:
        print(f'{pred_fib.shape=}')
        print(f'{len(spacing)=}, {spacing=}')
    if args.visualize == True:
        if 'pred_myo' in data and not '3D' in model.model_name:
            # show mri image, myocardium prediction, and new fibrosis prediction
            visualize_stack(torch.from_numpy(img_array),
                            uncrop(data['pred_myo'][0].permute(1, 0, 2, 3), data['crop_corners'], pred_fib.shape),
                            pred_fib)
        else:
            # show mri image and new fibrosis prediction
            visualize_stack(torch.from_numpy(img_array), pred_fib)

    # reformat to original shape and type
    pred_fib = pred_fib.view(original_shape)
    pred_fib = pred_fib.detach().numpy()
    if args.verbose:
        print(f'{pred_fib.shape=}')

    assert pred_fib.shape == original_shape

    # create sitk image
    pred_fib_image = sitk.GetImageFromArray(pred_fib, isVector=False)
    pred_fib_image.SetOrigin(origin)
    pred_fib_image.SetSpacing(spacing)
    return pred_fib_image

def main(args):
    # setup & check paths
    IN_DIR = Path(args.input_path)
    assert IN_DIR.exists()

    MODEL_CKPT = Path(args.load_checkpoint)
    assert MODEL_CKPT.exists()
    MODEL_DIR = MODEL_CKPT.parent.parent

    if args.output_path == None:
        OUT_DIR = MODEL_DIR.joinpath(f"{args.dataset}{args.suffix}")
    else:
        OUT_DIR = Path(args.output_path)

    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=False)
    else:
        # made this so we don't accidentally overwrite results
        # don't forget to answer, otherwise you'll be waiting a long time
        answer = "???"
        while answer not in ["Y", "N"]:
            answer = input(f"{OUT_DIR=} already exists and has {len(list(OUT_DIR.glob('*')))} files, do you want to risk overwiting it? [Y/N]\n")
            if answer=="Y" or answer == "y":
                print("Overwriting previous fibrosis segmentations...\n")
            elif answer=="N" or answer=="n":
                print("Cancelled fibrosis inference\n")
                return
            else:
                print("Please type Y (yes) or N (no)\n")

    if args.pred_myo_path != None:
        PRED_MYO_DIR = Path(args.pred_myo_path)
        pred_myo_available = True
    else:
        pred_myo_available = False

    if args.pred_fib_path != None:
        PRED_FIB_DIR = Path(args.pred_fib_path)
        pred_fib_available = True
    else:
        pred_fib_available = False

    # load model
    model = load_model(args)
    num_imgs = 1
    if pred_myo_available:
        num_imgs += 1
    if pred_fib_available:
        num_imgs += 1
    args.transform = compose_inference_transforms_with_segmentations(image_norm=args.image_norm, center_crop=args.center_crop,
                                                                        input_size=args.resize, num_imgs=num_imgs)

    # load data & inference & save
    nifti_files = sorted([x for x in IN_DIR.glob(f'*{args.files_must_contain}*.{args.input_extension}')])
    for nifti_file in tqdm(nifti_files):
        if args.verbose:
            print(f"{nifti_file=}")
        if args.pred_myo_path == None:
            pred_myo_file = None
        else:
            pred_myo_file = [f for f in PRED_MYO_DIR.glob(f'*{Path(nifti_file.stem).stem}*')]
            if len(pred_myo_file) == 0:
                print(f"No myocardium prediction found for sequence {nifti_file}, skipping...")
                continue
            assert len(pred_myo_file) == 1, f"Multiple myocardium predictions found for sequence {nifti_file=}, {pred_myo_file=}"
            pred_myo_file = pred_myo_file[0]
        output_file = OUT_DIR.joinpath(f'{Path(nifti_file.stem).stem}{args.suffix}.nrrd') # double stem for '.nii.gz' extension
        if args.pred_fib_path == None:
            pred_fib_file = None
        else:
            pred_fib_file = [f for f in PRED_FIB_DIR.glob(f'*{Path(nifti_file.stem).stem}*')]
            if len(pred_fib_file) == 0:
                print(f"No fibrosis prediction found for sequence {nifti_file}, skipping...")
                continue
            assert len(pred_fib_file) == 1, f"Multiple fibrosis predictions found for sequence {nifti_file=}, {pred_fib_file=}"
            pred_fib_file = pred_fib_file[0]
        output_file = OUT_DIR.joinpath(f'{Path(nifti_file.stem).stem}{args.suffix}.nrrd') # double stem for '.nii.gz' extension

        try:
            output_nifti = infer(model, nifti_file, pred_myo_file, pred_fib_file, args)
            # write image
            if args.dry_run == False:
                compress = True
                sitk.WriteImage(output_nifti, str(output_file), compress)
        except Exception as e:
            print('Something went wrong for', nifti_file)
            print(e)
            traceback.print_exc()

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
    parser.add_argument('--dataset', type=str, required=True, choices=['deeprisk', 'emidec'],
                        help="Dataset we want to run on.")
    # paths
    parser.add_argument('--load_checkpoint', type=str, required=True,
                        help="""Path pointing to trained model checkpoint. 
                        Pytorch lightning uses extension .ckpt by default, but can be anything.""")
    parser.add_argument('--input_path', type=str, required=True,
                        help="Path to directory where your input mri image data is.")
    parser.add_argument('--pred_myo_path', type=str, default=None,
                        help=""" Path to directory where your myocardium predictions are.
                        Default=None for when you don't need myocardium predictions.""")
    parser.add_argument("--pred_fib_path", type=str, default=None,
                        help=""" Path to directory where your fibrosis predictions are.
                        Default=None for when you don't need fibrosis predictions, which is
                        almost always unless you're using a 'cascaded' type model.""")
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
    parser.add_argument('--suffix', type=str, default='_fib_pred',
                        help="""Suffix to add to output predictions. Use something different
                         if you dont want to overwrite earlier predictions.""")

    # preprocessing hyperparameters
    parser.add_argument("--roi_crop", type=str, default="fixed", choices=['fitted', 'fixed'],
                         help="Type of region of interest crop. See dataset.py for details.")
    parser.add_argument('--crop_shortest', action='store_true',
                        help="Make a square center crop, where crop size is min(height, width).")
    parser.add_argument('--center_crop', type=int, default=None,
                        help="Make a square center crop of given size. Default=None for no crop. ")
    parser.add_argument('--resize', type=int, default=224,
                        help="Resize input images to given size. None for no resizing.")
    parser.add_argument('--image_norm', default='per_image', type=str,
                         choices=['per_image', 'global_agnostic', 'global_statistic', 'no_norm'],
                         help="Type of image normalization. See preprocessing.py for details.")
                         

    args = parser.parse_args()
    main(args)
