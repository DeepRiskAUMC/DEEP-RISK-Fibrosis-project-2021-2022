from tqdm import tqdm
import SimpleITK as sitk
import torch
import numpy as np
from pathlib import Path
import torchvision
import torchvision.transforms.functional as TF
from argparse import ArgumentParser
from torchvision import transforms
import yaml
import torch.utils.data.dataloader as tudl
import traceback

from train_segmentation import init_model
from preprocessing import get_array_from_nifti, compose_inference_transforms_with_segmentations, uncrop
from datasets import process_images_3d
from inference_myocard_segmentation import visualize_stack

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

def infer(model, image_file, pred_myo_file, pred_fib_file, args):
    images, image_names = [], []
    img_array, spacing, origin = get_array_from_nifti(image_file, with_spacing=True, with_origin=True)
    original_shape = img_array.shape
    # make up time dimension spacing if not provided (otherwise niftis not readable with SimpleITK 2.1+ ???)
    if len(spacing) == 3 and len(original_shape) == 4:
        spacing = spacing + spacing[-1]
    elif len(spacing) == 3 and len(original_shape) == 3:
        img_array = img_array[None, ...]

    if args.verbose:
        print(f"{img_array.shape=}")
    images.append(img_array)
    image_names.append('img')
    if pred_fib_file != None:
        pred_fib = get_array_from_nifti(pred_fib_file)
        images.append(pred_fib)
        image_names.append('pred_fib')
    if pred_myo_file != None:
        pred_myo = get_array_from_nifti(pred_myo_file)
        images.append(pred_myo)
        image_names.append('pred_myo')

    if '3D' in model.model_name:
        depth_to_batch = False
    else:
        depth_to_batch = True
    data = process_images_3d(images, image_names, args.transform, args.roi_crop, depth_to_batch=depth_to_batch)
    data = tudl.default_collate([data])
    # inference
    if 'stacked' in model.model_name:
        assert 'pred_myo' in data.keys(), "Myocardium prediction is necessary for this model"
        input = torch.stack([data['img'][0].squeeze(1), data['pred_myo'][0].view_as(data['img'][0]).squeeze(1)], dim=1)
    else:
        input = data['img'][0]

    # format model input
    pred_myo_input = ('stacked' in model.model_name)
    pred_fib_input = ('cascaded' in model.model_name)
    if (not pred_myo_input) and (not pred_fib_input):
        input = data['img'][0]
    elif pred_myo_input and not pred_fib_input:
        input = torch.stack([data['img'][0].squeeze(1), data['pred_myo'][0].view_as(data['img'][0]).squeeze(1)], dim=1)
    elif pred_fib_input and not pred_myo_input:
        input = torch.stack([data['img'][0].squeeze(1), data['pred_fib'][0].view_as(data['img'][0]).squeeze(1)], dim=1)
    elif pred_fib_input and pred_myo_input:
        input = torch.stack([data['img'][0].squeeze(1), data['pred_myo'][0].view_as(data['img'][0]).squeeze(1), data['pred_fib'][0].view_as(data['img'][0]).squeeze(1)], dim=1)

    if args.verbose:
        print(f"{input.shape=}")

    if args.dataset == 'emidec' and 'cascaded' in model.model_name:
        # for emidec reverse slice order to agreee with deeprisk slice ordering
        input = input.flip(dims=(-3,))

    pred_fib, attention_coefs = model.forward(input.float())

    if args.dataset == 'emidec' and 'cascaded' in model.model_name:
        # reverse emidec to original slice ordering to save
        pred_fib = pred_fib.flip(dims=(-3,))

    if args.verbose:
        print(f"{pred_fib.shape=}")

    if args.roi_crop in ["fitted", "fixed"]:
        assert args.center_crop == None and args.crop_shortest == False, "Pick either roi crop or center crop"
        if '3D' in model.model_name:
            pred_fib = pred_fib[0] # remove batch dimension
        else:
            pred_fib = pred_fib.permute(1, 0, 2, 3) # batch dimension back to depth
        if args.verbose:
            print(f"{pred_fib.shape=}, {original_shape=}")
        pred_fib = uncrop(pred_fib, data['crop_corners'], original_shape)
    else:
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
            visualize_stack(torch.from_numpy(img_array), uncrop(data['pred_myo'][0].permute(1, 0, 2, 3), data['crop_corners'], pred_fib.shape), pred_fib,)
        else:
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
        answer = "???"
        while answer not in ["Y", "N"]:
            answer = input(f"{OUT_DIR=} already exists and has {len(list(OUT_DIR.glob('*')))} files, do you want to overwite it? [Y/N]\n")
            if answer=="Y":
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
    #num_imgs = 1 if args.pred_myo_path == None else 2
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
    parser.add_argument('--dry_run', action='store_true') # dry run does not save the results
    parser.add_argument('--verbose', action='store_true') # print shapes during conversion
    parser.add_argument('--visualize', action='store_true') # plot segmentations

    parser.add_argument('--dataset', type=str, required=True, choices=['deeprisk', 'emidec']) # no default to prevent accidental overwriting of dataset segmentations

    # paths
    parser.add_argument('--load_checkpoint', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--pred_myo_path', type=str, default=None)
    parser.add_argument("--pred_fib_path", type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    # intput & output filename details
    parser.add_argument('--files_must_contain', type=str, default="MAG") # pathlib can regex for folder structures, like emidec Casexyz/Images/Casexyz.nii.gz -> "*\Images\*"
    parser.add_argument('--input_extension', type=str, default='mha', choices=['mha', 'nrrd', 'nii', 'nii.gz']) # might not be an extensive list of acceptable input types
    parser.add_argument('--suffix', type=str, default='_fib_pred')

    # preprocessing hyperparameters
    parser.add_argument("--roi_crop", type=str, default="fixed", choices=['fitted', 'fixed'])
    parser.add_argument("--crop_shortest", action='store_true')
    parser.add_argument('--center_crop', type=int, default=None)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--image_norm', default='per_image', type=str, choices=['per_image', 'global_agnostic', 'global_statistic', 'no_norm'])


    args = parser.parse_args()
    main(args)
