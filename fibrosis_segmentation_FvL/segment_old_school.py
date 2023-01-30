import os
from datetime import datetime
import sys
import argparse
import random
import cv2
import statistics
import torch
import numpy as np
from data_loading.load_data import load_data
from utils_functions.criterions import Diceloss, dice_score, Diceloss, hausdorff_distance, average_hausdorff_distance, get_TPR, get_TNR

def test(args, version_nr):
    test_loader = load_data(dataset=args.dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            fibrosis_model=args.method,
                            myocard_model_version=args.version_myocard_preds,
                            resize=args.resize,
                            size=args.size,
                            only_test=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    dice_scores = []
    if 'hausdorff' in args.metrics:
        hausdorff_scores = []
    if 'average_dist' in args.metrics:
        average_dist_scores = []
    if 'TPR' in args.metrics:
        TPR_scores = []
    if 'TNR' in args.metrics:
        TNR_scores = []

    for batch in test_loader:
        LGE_image, myo_pred, fibrosis_mask, pat_id, slice_nr = batch
        flattened_masked_values = []
        flattened_pixel_locations = []
        LGE_image, myo_pred, gt_mask = LGE_image.squeeze(), myo_pred.squeeze(), fibrosis_mask.squeeze()
        for i in range(myo_pred.shape[0]):
            for j in range(myo_pred.shape[1]):
                if myo_pred[i,j] == 255:
                    flattened_masked_values.append(LGE_image[i,j])
                    flattened_pixel_locations.append((i,j))
        if len(flattened_masked_values) == 0:
            prediction = np.zeros_like(LGE_image)
        else:
            flattened_masked_values = np.array(flattened_masked_values, dtype=np.float32).reshape((-1,1))
            if args.method == 'k-means':
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                _, labels, (centers) = cv2.kmeans(flattened_masked_values, args.n_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            elif args.method == 'GMM':
                gmm_classifier = cv2.ml.EM_create()
                gmm_classifier.setClustersNumber(args.n_classes)
                _, _, labels, _ = gmm_classifier.trainEM(flattened_masked_values)
                if len(labels) != len(flattened_masked_values):
                    raise ValueError(f'Number of labels and input values should be equal but got {len(labels)} and {len(flattened_masked_values)}')
            else:
                raise ValueError(f'Method {args.method} not recognized.')
            prediction = np.zeros_like(LGE_image)
            for i, label in enumerate(labels):
                if label == 1.0:
                    pixel_location = flattened_pixel_locations[i]
                    # print(pixel_location)
                    prediction[pixel_location[0], pixel_location[1]] = 1
            # raise ValueError('')
        dice_scores.append(dice_score(torch.from_numpy(prediction), gt_mask))
        saving_folder = os.path.join(args.output_path, args.dataset, 'fibrosis', 'old_school')
        os.makedirs(saving_folder, exist_ok=True)
        saving_folder = os.path.join(saving_folder, f"version_{version_nr}")
        os.makedirs(saving_folder, exist_ok=True)
        cv2.imwrite(os.path.join(saving_folder, f"prediction_{pat_id[0]}_slice{slice_nr.item()}.png"), prediction * 255)
        print_string = f"{pat_id[0]}, slice {slice_nr.item()}. Dice score 2: {dice_scores[-1]}"
        gt_mask = gt_mask.cpu().detach().numpy()
        if 'hausdorff' in args.metrics:
            hausdorff_score = hausdorff_distance(prediction, gt_mask)
            hausdorff_scores.append(hausdorff_score)
            print_string = print_string + f". Hausdorff distance: {hausdorff_score}"
        if 'average_dist' in args.metrics:
            average_dist_score = average_hausdorff_distance(prediction, gt_mask)
            average_dist_scores.append(average_dist_score)
            print_string = print_string + f". Average distance: {average_dist_score}"
        if 'TPR' in args.metrics:
            # print(pat_id[0] == 'DRAUMC0411')
            # print(slice_nr == 6)
            if pat_id[0] == 'DRAUMC0411' and slice_nr == 6:
                TPR = get_TPR(prediction, gt_mask, info=True)
            else:
                TPR = get_TPR(prediction, gt_mask)
            TPR_scores.append(TPR)
            print_string = print_string + f". TPR: {TPR}"
        if 'TNR' in args.metrics:
            TNR = get_TNR(prediction, gt_mask)
            TNR_scores.append(TNR)
            print_string = print_string + f". TNR: {TNR}"
        print(print_string)
    final_print_string = f'Mean dice score: {sum(dice_scores)/len(dice_scores)}. Min dice score: {min(dice_scores)}. Max dice score: {max(dice_scores)}. Median dice score: {statistics.median(dice_scores)}'
    if 'hausdorff' in args.metrics:
        final_print_string = final_print_string + f'. Mean hausdorff distance: {sum(hausdorff_scores)/len(hausdorff_scores)}'
    if 'average_dist' in args.metrics:
        final_print_string = final_print_string + f'. Mean average distance: {sum(average_dist_scores)/len(average_dist_scores)}'
    if 'TPR' in args.metrics:
        final_print_string = final_print_string + f'. Mean TPR: {sum(TPR_scores)/len(TPR_scores)}'
    if 'TNR' in args.metrics:
        final_print_string = final_print_string + f'. Mean TNR: {sum(TNR_scores)/len(TNR_scores)}'
    print(final_print_string)
    return saving_folder      


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--method', default='k-means', type=str,
                        help='What method to use for the segmentation',
                        choices=['k-means', 'GMM'])
    parser.add_argument('--n_classes', default=2, type=int,
                        help='Number of classes.')
    parser.add_argument('--dimension', default='2D', type=str,
                        help='What kind of model dimensions we want to use',
                        choices=['2D', '3D'])

    # Other hyperparameters
    parser.add_argument('--dataset', default='AUMC2D', type=str,
                        help='What dataset to use for the segmentation',
                        choices=['AUMC2D', 'AUMC3D', 'Myops']) 
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Minibatch size') 
    parser.add_argument('--resize', default='crop', type=str,
                        help='Whether to resize all images to 256x256 or to crop images to the size of the smallest image width and height',
                        choices=['resize', 'crop', 'none'])    
    parser.add_argument('--size', default=['168', '168'], nargs='+', type=str,
                        help='Shape to which the images need to be cropped. Elements of lists are Strings which are later converted to ints.')        
    parser.add_argument('--metrics', default='none', nargs='+', type=str,
                        help='Metrics (other than Dice score) to evaluate the segmentations on.')        
    parser.add_argument('--use_only_fib_slices', default='no', type=str,
                        help='If we want to only use the 2D slices that contain fibrosis. Only useful when using dataset AUMC2D',
                        choices=['yes', 'no'])
    parser.add_argument('--version_myocard_preds', default='6', type=int,
                        help='Model version of which the myocard predictions should be used')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--output_path', default='output', type=str,
                        help='Path to store the segmented images')

    args = parser.parse_args()

    #write prints to file
    version_nr = args.method
    file_name = f'test_segmentation_version_{version_nr}.txt'
    first_path = os.path.join(args.output_path, args.dataset, 'fibrosis', file_name)
    # second_path = os.path.join(args.output_path, f"version_{version_nr}", file_name)
    sys.stdout = open(first_path, "w")
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print(f"Dataset: {args.dataset} | method: {args.method} | classes: {args.n_classes} | resize: {args.resize} | seed: {args.seed} | version_no: {version_nr} | version_myocard_preds: {args.version_myocard_preds}")
    version_folder = test(args, version_nr)
    sys.stdout.close()
    os.rename(first_path, os.path.join(version_folder, file_name))
    sys.stdout = open("/dev/stdout", "w")
    print('Segmentation completed')