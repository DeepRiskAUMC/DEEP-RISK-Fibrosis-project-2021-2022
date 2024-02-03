""" Preprocessing/data transformation functions.

"""

import numpy as np

# import monai - for the image transformations
from monai.transforms import (
    EnsureChannelFirst, Compose, LoadImage, RandRotate90, 
    Resize, ScaleIntensityRange, ToTensor, AdjustContrast, NormalizeIntensity, HistogramNormalize,
    RandFlip, RandSpatialCrop, ResizeWithPadOrCrop, ThresholdIntensity, ScaleIntensity,  GaussianSmooth
)

### ROELS FUNCTION ###
def myoseg_to_roi(img, prediction, mask, margin_frac=0.0, fixed_size=None, pad_value=0, calc_center="mean", masked=False):
    # Returns a square region of interest crop of an image based on myocardium predictions,
    # and the coordinates where the crop was taken.
    # Inputs:
    #   - img:          image where the last 2 dimension correspond to height and width dimensions
    #   - prediction:   (Myocardium) prediciton segmentation, last 2 dimensions are height and width
    #   - margin_frac:  Only used when fixed_size=None. The margin to take around smallest crop 
    #                   containing the myocardium segmentation. Default=0.3.
    #   - fixed size:   Size of the square crop around the myocardium center. When None, take the 
    #                   smallest square around the myocardium prediction, i.e. the fitted roi.
    #                   Default=None.
    #   - pad value:    Values to use for padding (making square crops on rectangular means padding
    #                   is sometimes necessary). Default=0.
    #   - calc_center:  Method used to calculate the center of the heart based on the myocardium
    #                   predictions. Options: "mean" -> take the average (x, y) coordinates,
    #                   "outer" -> take the average between maximum (x, y) coordinates. 
    #                   Only gets used when fixed_size != None. Default="mean".
    if masked: 
        coords = np.argwhere(mask != 0.0)
    else:
        coords = np.argwhere(prediction > 0.5)

    if len(coords) < 1:
        # Found no myocard prediction in MRI, taking center crop
        x_min = 0
        x_max = img.shape[-2]
        y_min = 0
        y_max = img.shape[-1]
    else:
        assert len(coords) >= 1, "Found no myocard prediction in MRI"
        x_min = coords[:, -2].min()
        x_max = coords[:, -2].max()
        y_min = coords[:, -1].min()
        y_max = coords[:, -1].max()

    if calc_center == "mean":
        if len(coords) < 1:
            # Found no myocard prediction in MRI, taking center crop
            center = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
        else:
            center = coords.mean(axis=0).astype(int)
            center = center[-2], center[-1]
    elif calc_center == "outer":
        center = int((x_min + x_max) / 2), int((y_min + y_max) / 2)

    # save original center (can be changed by later padding)
    original_center = center
    # find square bbox size
    min_dist_to_border = min(center[0], img.shape[-2] - center[0], center[1], img.shape[-1] - center[1])
    if fixed_size != None:
        if not isinstance(fixed_size, int) or fixed_size <= 0:
            raise ValueError
        bbox_side = fixed_size

        # pad images in case fixed roi goes over border
        padding_width = max(0, fixed_size // 2 + 1 - min_dist_to_border)

        # handle 3d and 2d inputs (first dim is channel)
        if len(img.shape) == 3:
            padding_element = [(0, 0), (padding_width, padding_width), (padding_width, padding_width)]
        elif len(img.shape) == 4:
            padding_element = [(0, 0), (0, 0), (padding_width, padding_width), (padding_width, padding_width)]
        else:
            raise ValueError

        img = np.pad(img, padding_element, mode='constant', constant_values=pad_value)
        center = center[0] + padding_width, center[1] + padding_width
    else:
        max_bbox_side = max((x_max - x_min)*(1 + margin_frac),
                            (y_max - y_min)*(1 + margin_frac))
        bbox_side = min(max_bbox_side, min_dist_to_border*2)

    # cutout roi
    bb_x_min = int(center[0] - 0.5 * bbox_side)
    bb_x_max = int(center[0] + 0.5 * bbox_side)
    bb_y_min = int(center[1] - 0.5 * bbox_side)
    bb_y_max = int(center[1] + 0.5 * bbox_side)

    roi_img = img[..., bb_x_min:bb_x_max, bb_y_min:bb_y_max]
    roi_seg = prediction[..., bb_x_min:bb_x_max, bb_y_min:bb_y_max]
    if mask is not None:
        roi_mask = mask[..., bb_x_min:bb_x_max, bb_y_min:bb_y_max]
    else:
        roi_mask = []
    assert roi_img.shape[-2] == roi_img.shape[-1], f"Image is not square {roi_img.shape}, {bb_x_min}, {bb_x_max}, {bb_y_min}, {bb_y_max} center: {center}, img shape {img.shape}"

    # save corresponding original corners so crop can be reversed
    og_x_min = int(original_center[0] - 0.5 * bbox_side)
    og_x_max = int(original_center[0] + 0.5 * bbox_side)
    og_y_min = int(original_center[1] - 0.5 * bbox_side)
    og_y_max = int(original_center[1] + 0.5 * bbox_side)

    return roi_img, roi_seg, roi_mask, (og_x_min, og_x_max, og_y_min, og_y_max)


class NormImage(object):
    def __init__(self, perc):
        self.perc = perc

    def __call__(self, img):
        min_val, max_val = np.percentile(img, self.perc)
        img = ((img.astype(img.dtype) - min_val) / (max_val - min_val)).clip(0, 1)
        return img
    
    # Alternatively you can use the following from monai: monai.transforms.ScaleIntensityRange
    # this would be in case you want the whole data to be in the same range

    # Good source on other techniques and their results
    # https://medium.com/@susanne.schmid/image-normalization-in-medical-imaging-f586c8526bd1

            
def compose_transforms_aumc(args):
    """ If IMG channels is higher than one, only image normalization transforms are applied. 
        For this setting data augmentation transforms are defined in the dataloader. """
    
    # Define transforms for image
    if args.IMG == 1:
        train_transforms = Compose(
                [
                        EnsureChannelFirst(channel_dim="no_channel"),
                        HistogramNormalize(num_bins=256, min=0.0, max=1.0),
                        NormImage(perc=args.norm_perc),
                        RandFlip( spatial_axis=0),
                        RandFlip( spatial_axis=1),
                        RandRotate90(),
                        Resize(args.win_size, mode = "area"),
        
                        ToTensor(),
                ]
                )
        
        val_transforms = Compose(
            [
                    EnsureChannelFirst(channel_dim="no_channel"),
                    HistogramNormalize(num_bins=256, min=0.0, max=1.0),
                    NormImage(perc=args.norm_perc),
                    Resize(args.win_size, mode = "area"),
    
                    ToTensor(),
            ]
            )
    else:
        
        train_transforms = Compose(
                [
                        HistogramNormalize(num_bins=256, min=0.0, max=1.0),
                        NormImage(perc=args.norm_perc),
                ]
                )
    
        val_transforms = Compose(
                [
                        HistogramNormalize(num_bins=256, min=0.0, max=1.0),
                        NormImage(perc=args.norm_perc),
                ]
                )
        
    return train_transforms, val_transforms

def compose_transforms_emidec(args):
    if args.model_version == 'VAE_Multi':
        train_transforms = Compose(
            [
                    LoadImage(image_only=True),
                    EnsureChannelFirst(channel_dim="no_channel"),
                    HistogramNormalize(num_bins=256, min=0.0, max=1.0),
                    NormImage(perc=args.norm_perc),
                    # AdjustContrast(gamma = 5.0), # This transformation is great for EMIDEC
                    # ScaleIntensity(minv=0.0, maxv=1.0), # this is important!!
                    RandFlip( spatial_axis=0),
                    RandFlip( spatial_axis=1),
                    RandRotate90(),
                    Resize(args.win_size, mode = "area"),
                    ToTensor(),
            ]
            )
        val_transforms = Compose(
                [
                        LoadImage(image_only=True),
                        EnsureChannelFirst(channel_dim="no_channel"),
                        HistogramNormalize(num_bins=256, min=0.0, max=1.0),
                        NormImage(perc=args.norm_perc),
                        # AdjustContrast(gamma = 5.0), # This transformation is great for EMIDEC
                        # ScaleIntensity(minv=0.0, maxv=1.0), # this is important!!
                        Resize(args.win_size, mode = "area"),
                        ToTensor(),
                ]
                )
    else:
        train_transforms = Compose(
                [
                        LoadImage(image_only=True),
                        EnsureChannelFirst(channel_dim="no_channel"),
                        AdjustContrast(gamma = 5.0), # This transformation is great for EMIDEC
                        ScaleIntensity(minv=0.0, maxv=1.0), # this is important!!
                        RandFlip( spatial_axis=0),
                        RandFlip( spatial_axis=1),
                        RandRotate90(),
                        Resize(args.win_size, mode = "area"),
                        ToTensor(),
                ]
                )
        val_transforms = Compose(
                [
                        LoadImage(image_only=True),
                        EnsureChannelFirst(channel_dim="no_channel"),
                        AdjustContrast(gamma = 5.0),
                        ScaleIntensity(minv=0.0, maxv=1.0),
                        Resize(args.win_size, mode = "area"),
                        ToTensor(),
                ]
                )
    return train_transforms, val_transforms

if __name__ == "__main__":
    pass