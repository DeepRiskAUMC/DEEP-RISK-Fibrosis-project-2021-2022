""" Preprocessing/data transformation functions.

A recommendation:
Some of the issues when doing data augmentation, is that you want to do some
transforms on both image and segmentation, and some on only the image.
E.g. spatial transforms like rotations and resizing you want for both,
(but maybe you want to use a different type of resampling).
ColorJitter you only want for the image.
Some of the functions/classes here are designed to facilitate this,
like ApplyAll and ApplyFirst.
When starting on a new project, it's probably easier to use some other
library which has a more useful implementation, for example
monai which has dictionary transforms: 
https://docs.monai.io/en/stable/transforms.html#dictionary-transforms
"""
import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


def get_array_from_nifti(filename, with_spacing=False, with_origin=False):
    # Wrapper to get array from a nifti filename.
    # Returns image array, or tuple (array, spacing(optionally), origin(optionally))
    img = sitk.ReadImage(str(filename))
    result = sitk.GetArrayFromImage(img).astype(np.float32)
    if with_spacing == True and with_origin == True:
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        result = (result, spacing, origin)
    elif with_spacing == True:
        spacing = img.GetSpacing()
        result = (result, spacing)
    elif with_origin == True:
        origin = img.GetOrigin()
        result = (result, origin)
    return result


def myoseg_to_roi(img, prediction, margin_frac=0.3, fixed_size=None, pad_value=0, calc_center="mean"):
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

    assert prediction.shape[-2] == img.shape[-2], f"prediction shape {prediction.shape} does not match image shape {img.shape}"
    assert prediction.shape[-1] == img.shape[-1], f"prediction shape {prediction.shape} does not match image shape {img.shape}"

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
    assert roi_img.shape[-2] == roi_img.shape[-1], f"Image is not square {roi_img.shape}, {bb_x_min}, {bb_x_max}, {bb_y_min}, {bb_y_max} center: {center}, img shape {img.shape}"

    # save corresponding original corners so crop can be reversed
    og_x_min = int(original_center[0] - 0.5 * bbox_side)
    og_x_max = int(original_center[0] + 0.5 * bbox_side)
    og_y_min = int(original_center[1] - 0.5 * bbox_side)
    og_y_max = int(original_center[1] + 0.5 * bbox_side)

    return roi_img, (og_x_min, og_x_max, og_y_min, og_y_max)


def roi_crop_multiple_images(pred_myo, images, fixed_size=None, margin_frac=0.3, pad_value=0, calc_center="mean"):
    # Function that applies the same ROI cropping to different image types
    # image :   a list of different image types corresponding to the same sequence.
    # For parameter details, see myoseg_to_roi function.
    results = []
    crop_corners_list = []
    for image in images:
        image, crop_corners = myoseg_to_roi(
            image, pred_myo, fixed_size=fixed_size, margin_frac=margin_frac, pad_value=pad_value, calc_center=calc_center)
        results.append(image)
        crop_corners_list.append(crop_corners)

    pred_myo, myo_crop_corners = myoseg_to_roi(
        pred_myo, pred_myo, fixed_size=fixed_size)
    for cc in crop_corners_list:
        assert cc == myo_crop_corners, f"{myo_crop_corners=}, {cc=}, {crop_corners_list=}"
    results = tuple(results) + (pred_myo, crop_corners)
    return results


def uncrop(img, crop_corners, original_shape, is_mask=False):
    # Function to uncrop an cropped and/or resized image (usually a model prediction)
    # into the original spatial dimensions.
    # Requires the corners where the crop was taken and
    # the original shape/size of the image.
    x_min_og, x_max_og, y_min_og, y_max_og = crop_corners
    x_min_og, x_max_og, y_min_og, y_max_og = x_min_og[0].item(), x_max_og[0].item(), y_min_og[0].item(), y_max_og[0].item()

    assert x_max_og - x_min_og == y_max_og - y_min_og # we assume the cropped image is square
    crop_og_size = x_max_og - x_min_og

    # resize back to original size
    if is_mask == True:
        interpolation = transforms.InterpolationMode.NEAREST
    else:
        interpolation = transforms.InterpolationMode.BILINEAR
    img = TF.resize(img, size=crop_og_size, interpolation=interpolation)

    x_min_crop, x_max_crop, y_min_crop, y_max_crop = 0, crop_og_size, 0, crop_og_size
    # if x,y coords go over image border after resizing, remove parts of crop
    # (this mean padding was done during the cropping)
    if x_min_og < 0:
        x_min_crop -= x_min_og
        x_min_og = 0
    if x_max_og > original_shape[-2]:
        x_min_crop -= (x_max_og - original_shape[-2])
    if y_min_og < 0:
        y_min_crop -= y_min_og
        y_min_og = 0
    if y_max_og > original_shape[-1]:
        y_min_crop -= (y_max_og - original_shape[-1])

    # put the cropped image a zero-padded version of the original image size
    new_img = torch.zeros(original_shape)
    new_img[..., x_min_og:x_max_og, y_min_og:y_max_og] = img[..., x_min_crop:x_max_crop, y_min_crop:y_max_crop]

    return new_img


def normalize_image_func(img):
    # normalizes a 2d numpy image (1 channel) such that it has 0 mean and 1 std
    img -= img.mean()
    img /= img.std()
    return img


class set_image_range(object):
    """Class to set image pixel values in the desired value range."""
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, image):
        image -= image.min()
        image += self.min
        image /= image.max()
        image *= self.max
        return image

    def __repr__(self):
        return f"set_image_range(min={self.min},max={self.max})"


class normalize_image(object):
    """Gives each image 0 mean and 1 std"""

    def __call__(self, image):
        image -= image.mean()
        image /= image.std()
        return image

    def __repr__(self):
        return "normalize_img()"


def denormalize_transform(mean=[.57], std=[.06]):
    # Function to reverse the torchvision normalize transform.
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    return inv_normalize


class CropToLargestSquare(object):
    """Takes a rectangular image and returns the largest square center crop."""
    def __call__(self, image):
        crop_size = min(image.shape[-1], image.shape[-2])
        output = transforms.CenterCrop((crop_size, crop_size))(image)
        return output


class ApplyAllRandomAffine(transforms.RandomAffine):
    """Applies the same torchvision randomaffine transform to a list of images."""
    def __init__(self, degrees, interpolations, **kwargs):
        super().__init__(degrees, **kwargs)
        self.interpolations = interpolations

    def forward(self, imgs):
        """imgs (list of PIL image or Tensors): images that should get the same transformation (e.g. img and segmentation masks)"""
        fill = self.fill
        if isinstance(imgs[0], torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * TF.get_image_num_channels(imgs[0])
            else:
                fill = [float(f) for f in fill]

        img_size = TF.get_image_size(imgs[0])
        ret = self.get_params(self.degrees, self.translate,
                              self.scale, self.shear, img_size)
        return [TF.affine(img, *ret, interpolation=interpolation, fill=fill) for img, interpolation in zip(imgs, self.interpolations)]


class ApplyAllRandomCrop(transforms.RandomCrop):
    """Applies the same torchvision randomcrop transform to a list of images."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, imgs):
        padded_imgs = []
        for img in imgs:
            if self.padding is not None:
                img = TF.pad(img, self.padding, self.fill, self.padding_mode)

            _, height, width = TF.get_dimensions(img)
            # pad the width if needed
            if self.pad_if_needed and width < self.size[1]:
                padding = [self.size[1] - width, 0]
                img = TF.pad(img, padding, self.fill, self.padding_mode)
            # pad the height if needed
            if self.pad_if_needed and height < self.size[0]:
                padding = [0, self.size[0] - height]
                img = TF.pad(img, padding, self.fill, self.padding_mode)
            padded_imgs.append(img)

        i, j, h, w = self.get_params(padded_imgs[0], self.size)

        return [TF.crop(img, i, j, h, w) for img in padded_imgs]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"


class ApplyAllRandomHorizontalFlip(torch.nn.Module):
    """ Applies the same torchvision randomhorizontalflip transform to a list of images."""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, imgs):
        if torch.rand(1) < self.p:
            return [TF.hflip(img) for img in imgs]
        else:
            return imgs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class ApplyAllRandomVerticalFlip(torch.nn.Module):
    """Applies the same torchvions randomverticalflip transform to a list of images."""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, imgs):
        if torch.rand(1) < self.p:
            return [TF.vflip(img) for img in imgs]
        else:
            return imgs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class ApplyAllResize(transforms.Resize):
    """ Applies resize to a list of images
        interpolations is a list of interpolation modes : for regular images bilinear is preferred,
        for segmentation masks nearest might be better, but this does seem to reduce small patches.
    """

    def __init__(self, interpolations, **kwargs):
        super().__init__(**kwargs)
        self.interpolations = interpolations

    def forward(self, imgs):
        return [TF.resize(img, self.size, interpolation, self.max_size, self.antialias) for (img, interpolation) in zip(imgs, self.interpolations)]

    def __repr__(self) -> str:
        detail = f"(size={self.size}, interpolations={[interpolation.value for interpolation in self.interpolations]}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


class ApplyAll(object):
    """Wrapper to apply a transform to a list of inputs
        Don't use if transforms have a random component (RandomAffine, Flips),
        otherwise different modalities don't get the same transform.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, imgs):
        return [self.transform(img) for img in imgs]

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}"
        s += str(self.transform)
        return s


class ApplyFirst(object):
    """Wrapper to apply a transform only to the first of a list of images"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, imgs):
        result = [self.transform(imgs[0])]
        result.extend(imgs[1:])
        return result

    def __repr__(self):
        s = f"{self.__class__.__name__}"
        s += str(self.transform)
        return s


class DepthToBatch(object):
    """ torchvision can't handle 3D inputs  (B, C, D, H, W),
        so reshape to (B+D, C, H, W), apply transforms and reshape back
        Batched transforms will get the same random parameters."""

    def __init__(self, depth_size):
        self.depth_size = depth_size

    def __call__(self, imgs):
        if len(imgs.shape) == 5:
            B, C, D, H, W = imgs.shape
            assert D == self.depth_size
            return imgs.view(B+D, C, H, W)
        elif len(imgs.shape) == 4:
            C, D, H, W = imgs.shape
            assert D == self.depth_size, f"Dimensions {D} and {self.depth_size} don't match"
            return imgs.view(D, C, H, W)

    def __repr__(self):
        s = f"{self.__class__.__name__}"
        return s


class RecoverDepth(object):
    """ Reverse of DepthToBatch."""

    def __init__(self, depth_size):
        self.depth_size = depth_size

    def __call__(self, imgs):
        BD, C, H, W = imgs.shape
        return imgs.view(int(BD/self.depth_size), C, self.depth_size, H, W).squeeze(0)

    def __repr__(self):
        s = f"{self.__class__.__name__}"
        return s


def compose_transforms(hparams, split):
    """"Compose an object to transform images, based on values in hparams."""
    if split not in ["train", "val"]:
        raise NotImplementedError

    transforms_list = []
    # make all non-square images square
    # if a roi crop is done they're already square
    if hparams.dataset == "deeprisk" and hparams.no_roi_crop == True:
        transforms_list.append(transforms.CenterCrop(
            (hparams.center_crop, hparams.center_crop)))

    # resize to desired size
    transforms_list.append(transforms.Resize(
        (hparams.input_size, hparams.input_size), interpolation=transforms.InterpolationMode.BILINEAR))

    # set range between (0, 1) for ColorJitter Transforms
    if hparams.dataset == 'cifar10':
        transforms_list.append(transforms.ToTensor())
    elif hparams.image_norm == "per_image" and hparams.dataset == 'deeprisk':
        transforms_list.append(set_image_range(0, 1))

    # color jitter (train only)
    if split == "train":
        transforms_list.append(transforms.ColorJitter(
            brightness=hparams.brightness, contrast=hparams.contrast))

    # normalize
    if hparams.image_norm == "per_image":
        transforms_list.append(normalize_image())
    elif hparams.image_norm == "global_agnostic" and hparams.dataset == 'deeprisk':
        transforms_list.append(transforms.Normalize((0.5), (0.5)))
    elif hparams.image_norm == "global_statistic" and hparams.dataset == 'deeprisk':
        transforms_list.append(transforms.Normalize(mean=[.57], std=[.06]))
    elif hparams.image_norm == "global_agnostic" and hparams.dataset == 'cifar10':
        transforms_list.append(transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif hparams.image_norm == "global_statistic" and hparams.dataset == 'cifar10':
        raise NotImplementedError
    elif hparams.image_norm == "no_norm":
        pass

    # other data augmentation (train only)
    if split == "train":
        transforms_list.append(transforms.RandomApply([transforms.RandomAffine(hparams.rotate, scale=tuple(hparams.scale),
                                                                               shear=hparams.shear, interpolation=transforms.InterpolationMode.BILINEAR)],
                                                      hparams.randomaffine_prob))
        if hparams.hflip > 0:
            transforms_list.append(
                transforms.RandomHorizontalFlip(hparams.hflip))
        if hparams.vflip > 0:
            transforms_list.append(
                transforms.RandomVerticalFlip(hparams.vflip))
        if hparams.randomcrop != False:
            transforms_list.append(transforms.RandomApply([transforms.RandomCrop(
                size=(hparams.randomcrop, hparams.randomcrop))], hparams.randomcrop_prob))
        if any([p > 0 for p in hparams.randomerasing_probs]):
            for p in hparams.randomerasing_probs:
                transforms_list.append(transforms.RandomErasing(
                    p=p, scale=hparams.randomerasing_scale, ratio=hparams.randomerasing_ratio))

    return transforms.Compose(transforms_list)


def compose_transforms_with_segmentations(hparams, split, num_imgs=3, depth_size=0):
    """"Compose an object to transform multiple image types, based on values in hparams.
    Assumption is made that the first image type is an actual mri image,
    while the following image types are things like segmentation maps.
    Normalization and colorjitter is only applied to the first image type. """
    if split not in ["train", "val"]:
        raise NotImplementedError

    # bilinear interpolation for both image and segmentation masks 
    # (nearest on segmentation masks significantly reduces small fibrosis patches?)
    # don't forget that afterwards, segmentations are not fully binary 
    # (so choose some reasonable cutoff between 0 and 1)
    interpolations = [transforms.InterpolationMode.BILINEAR] * num_imgs
    transforms_list = []

    if depth_size != 0:
        transforms_list.append(ApplyAll(DepthToBatch(depth_size)))

    # make all non-square images square
    if hparams.dataset == "deeprisk" and hparams.no_roi_crop == True:
        transforms_list.append(ApplyAll(transforms.CenterCrop(
            (hparams.center_crop, hparams.center_crop))))

    # resize to desired size
    transforms_list.append(ApplyAllResize(
        size=(hparams.input_size, hparams.input_size), interpolations=interpolations))

    # set range between (0, 1) for ColorJitter Transforms
    if hparams.dataset == 'cifar10':
        transforms_list.append(ApplyFirst(transforms.ToTensor()))
    elif hparams.image_norm == "per_image" and hparams.dataset == 'deeprisk':
        transforms_list.append(ApplyFirst(set_image_range(0, 1)))

    # color jitter (train only)
    if split == "train":
        transforms_list.append(ApplyFirst(transforms.ColorJitter(
            brightness=hparams.brightness, contrast=hparams.contrast)))

    # normalize
    if hparams.image_norm == "per_image":
        transforms_list.append(ApplyFirst(normalize_image()))
    elif hparams.image_norm == "global_agnostic" and hparams.dataset == 'deeprisk':
        transforms_list.append(ApplyFirst(transforms.Normalize((0.5), (0.5))))
    elif hparams.image_norm == "global_statistic" and hparams.dataset == 'deeprisk':
        transforms_list.append(ApplyFirst(
            transforms.Normalize(mean=[.57], std=[.06])))
    elif hparams.image_norm == "global_agnostic" and hparams.dataset == 'cifar10':
        transforms_list.append(ApplyFirst(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))))
    elif hparams.image_norm == "global_statistic" and hparams.dataset == 'cifar10':
        raise NotImplementedError
    elif hparams.image_norm == "no_norm":
        pass

    # other data augmentation (train split only)
    if split == "train":
        transforms_list.append(transforms.RandomApply([ApplyAllRandomAffine(hparams.rotate, scale=tuple(hparams.scale),
                                                                            shear=hparams.shear, interpolations=interpolations)],
                                                      hparams.randomaffine_prob))
        if hparams.hflip > 0:
            transforms_list.append(ApplyAllRandomHorizontalFlip(hparams.hflip))
        if hparams.vflip > 0:
            transforms_list.append(ApplyAllRandomVerticalFlip(hparams.vflip))
        if hparams.randomcrop != False:
            transforms_list.append(transforms.RandomApply([ApplyAllRandomCrop(
                size=(hparams.randomcrop, hparams.randomcrop))], hparams.randomcrop_prob))
        if any([p > 0 for p in hparams.randomerasing_probs]):
            for p in hparams.randomerasing_probs:
                transforms_list.append(ApplyAll(transforms.RandomErasing(
                    p=p, scale=hparams.randomerasing_scale, ratio=hparams.randomerasing_ratio)))

    if depth_size != 0:
        transforms_list.append(ApplyAll(RecoverDepth(depth_size)))

    return transforms.Compose(transforms_list)


def compose_inference_transforms_with_segmentations(image_norm="per_image", center_crop=None,
                                                    input_size=None, num_imgs=3, depth_size=0,
                                                    dataset='deeprisk'):
    """"Similar to compose_transforms_with_segmentations, but only for the transforms
    necessary for inference (i.e. no random data augmentation).
    """
    # bilinear interpolation for both image and segmentation masks 
    # (nearest on segmentation masks can significantly reduce small fibrosis patches)
    interpolations = [transforms.InterpolationMode.BILINEAR] * num_imgs
    transforms_list = []

    if depth_size != 0:
        transforms_list.append(ApplyAll(DepthToBatch(depth_size)))

    if center_crop != None:
        transforms_list.append(
            ApplyAll(transforms.CenterCrop((center_crop, center_crop))))

    # resize to desired size
    if input_size != None:
        transforms_list.append(ApplyAllResize(
            size=(input_size, input_size), interpolations=interpolations))
    # normalize
    if image_norm == "per_image":
        transforms_list.append(ApplyFirst(normalize_image()))
    elif image_norm == "global_agnostic" and dataset == 'deeprisk':
        transforms_list.append(ApplyFirst(transforms.Normalize((0.5), (0.5))))
    elif image_norm == "global_statistic" and dataset == 'deeprisk':
        transforms_list.append(ApplyFirst(
            transforms.Normalize(mean=[.57], std=[.06])))
    elif image_norm == "global_agnostic" and dataset == 'cifar10':
        transforms_list.append(ApplyFirst(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))))
    elif image_norm == "global_statistic" and dataset != 'deeprisk':
        raise NotImplementedError
    elif image_norm == "no_norm":
        pass
    else:
        raise NotImplementedError

    if depth_size != 0:
        transforms_list.append(ApplyAll(RecoverDepth(depth_size)))

    return transforms.Compose(transforms_list)


if __name__ == "__main__":
    pass
