from pathlib import Path
from argparse import ArgumentParser
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from nnunet.dataset_conversion.utils import generate_dataset_json

""" Code to convert a map with nifti's into the nnUnet file and folder format.
This code assumes we don't have ground truth segmentations, it only puts the images in the
correct format so nnUnet can preprocess and perform inference on the images.

Note that for this code to function, environment variables have to be set, as described here:
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md .
Full nnUnet documentation can be found here: https://github.com/MIC-DKFZ/nnUNet.

P.S we didn't end up using nnUnet, but left the code here in case it is needed in the future.
"""


def make_folders(*dirs):
    # from a list of directories, create all non-existing directories
    print(dirs)
    for dir in dirs:
        print(dir)
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)
    return


def write_images(src_dir, train_dir, labels_dir, test_dir):
    """ Write images from source directory into both the training and test directories.
    Dummy segmentation maps are written to the labels directory,
    since nnUnet requires labels but we don't know the ground truth.
    """
    # following nnUnet naming, "raw data" means without preprocessing
    raw_data = [f for f in src_dir.glob(f"*PSIR*.mha")]

    for i, f in tqdm(enumerate(raw_data)):
        try:
            # load image
            img = sitk.ReadImage(str(f))
            img_arr = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            # convert 4D to 3D
            # (i.e. remove the extra channel/time dimension, however you want to interpret it)
            img_arr = img_arr[0]
            spacing = tuple(list(spacing[:-1]))
            origin = tuple(list(origin[:-1]))

            new_img = sitk.GetImageFromArray(img_arr)
            new_img.SetSpacing(spacing)
            new_img.SetOrigin(origin)

            img_out_fname_tr = train_dir.joinpath(
                f"{f.stem}_{i:03}_0000.nii.gz")
            img_out_fname_ts = test_dir.joinpath(
                f"{f.stem}_{i:03}_0000.nii.gz")

            sitk.WriteImage(new_img, str(img_out_fname_tr))
            sitk.WriteImage(new_img, str(img_out_fname_ts))

            # make dummy segmentation map (unfortunately, nnUnet will not preprocess otherwise)
            seg_arr = np.ones_like(img_arr)
            seg = sitk.GetImageFromArray(seg_arr)
            seg.SetSpacing(spacing)
            seg.SetOrigin(origin)
            seg_fname = labels_dir.joinpath(f"{f.stem}_{i:03}.nii.gz")
            sitk.WriteImage(seg, str(seg_fname))

        except Exception as e:
            print('Something went wrong', f)
            print(e)


def main(args):
    # set path objects
    SRC_DIR = Path(args.src_dir)
    BASE_DIR = Path(os.environ['nnUNet_raw_data_base']).joinpath(
        'nnUNet_raw_data').joinpath(f"Task{args.task_id:03}_{args.task_name}")
    TRAIN_DIR = BASE_DIR.joinpath("imagesTr")
    LABELS_DIR = BASE_DIR.joinpath("labelsTr")
    TEST_DIR = BASE_DIR.joinpath("imagesTs")
    JSON_PATH = BASE_DIR.joinpath('dataset.json')

    make_folders(BASE_DIR, TRAIN_DIR, LABELS_DIR, TEST_DIR)
    # images are written in nnUnets training directory, such that nnUnet preprocessing is saved
    # these preprocessed images can then be used for the weak supervision training
    # (But we ended up using our own preprocessing, and segmentation models, bypassing the need for nnUnet)
    write_images(SRC_DIR, TRAIN_DIR, LABELS_DIR, TEST_DIR)

    generate_dataset_json(str(JSON_PATH), str(TRAIN_DIR), str(TEST_DIR), ('PSIR',),
                          labels={0: 'background', 1: 'aankleuring'}, dataset_name=args.task_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task_id", type=int, default=500,
                        help="Identifier for task. Id needs to be: 500 <= id <= 999 for custom tasks/datasets. See nnUnet documentation for more details.")
    parser.add_argument("--task_name", type=str,
                        default="MyocardSegmentation", help="Task name, see nnUnet documentation for more details.")
    parser.add_argument("--src_dir", type=str,
                        default=r"\\amc.intra\users\R\rcklein\home\deeprisk\weakly_supervised\data\all_niftis_n=657",
                        decription="Source directory containing the mri's in nifti format that you want to convert.")

    args = parser.parse_args()

    main(args)
