# Author: Georgia Doumou (g.doumou@lms.mrc.ac.uk)
# Co-author: Wenjia Bai (w.bai@imperial.ac.uk); Su Boyang (su.boyang@nhcs.com.sg)
# Date: 03/08/2018
import os
from os import rename
import pydicom as dicom
import glob
import shutil
import operator
import numpy as np
import nibabel as nib
from collections import OrderedDict, Counter
import math
import sys

def slice_spacing_dict(series_nr_path, series_type):
    slice_dict = {}
    slices_folders = os.listdir(series_nr_path)

    if series_type == 'CINE':
        for slice_no in slices_folders:
            n = 0
            for slice_name in os.listdir(os.path.join(series_nr_path, slice_no)):
                if n == 0:
                    slice_loc = round(float((slice_name).split('_')[1]))
                    if slice_loc in slice_dict:
                        slice_no = slice_dict[slice_loc]
                    slice_dict[slice_loc] = slice_no
                    n = 1
    if series_type == 'LGE':
        for slice_name in slices_folders:
            slice_loc = round(float(slice_name.split('_')[0]))
            slice_dict[slice_loc] = slice_name
    ordered_slice_dict = OrderedDict(sorted(slice_dict.items()))
    return ordered_slice_dict

def second_check(ordered_slice_dict):
    slice_locations = [float(location) for location in ordered_slice_dict.keys()]
    differences = [round(slice_locations[i] - slice_locations[i - 1]) for i in range(1, len(slice_locations))]
    differences_counter = Counter(differences)
    most_common_difference = differences_counter.most_common(1)[0][0]
    previous_location = slice_locations[0]
    second_location = slice_locations[1]

    first = True
    for location, _ in ordered_slice_dict.items():
        if first:
            difference = round(second_location - location)
            if difference != most_common_difference:
                ordered_slice_dict.pop(location)
                return second_check(ordered_slice_dict)
            else:
                previous_location = location
                first = False

        else:
            difference = round(location - previous_location)
            if difference != most_common_difference:
                ordered_slice_dict.pop(location)
                return second_check(ordered_slice_dict)
            else:
                previous_location = location

    return ordered_slice_dict

def reorder_dicoms(series_nr_path, ordered_slice_dict, series_type):
    ordered_slice_dict = second_check(ordered_slice_dict)
    if ordered_slice_dict:
        new_series_nr_path = series_nr_path + '_temp'

        if not os.path.exists(new_series_nr_path):
            os.makedirs(new_series_nr_path)

        for slice_loc, value in ordered_slice_dict.items():   
            if series_type == 'CINE':
                folder_series_nr_path = new_series_nr_path + '/' + str(value)
                if not os.path.exists(folder_series_nr_path):
                    os.makedirs(folder_series_nr_path)
                input_path = os.path.join(series_nr_path, value)            
                shutil.copytree(input_path, folder_series_nr_path, dirs_exist_ok=True)

            if series_type == 'LGE':
                input_path = os.path.join(series_nr_path, value)
                shutil.copy(input_path, new_series_nr_path)
                
        return new_series_nr_path, True
    else:
        return series_nr_path, False

def convert_lge(output_name, nii_output_dir, series_nr_path):
    os.chdir(series_nr_path)
    slices = sorted(os.listdir(series_nr_path), key=lambda x: float((x).split('_')[0]))
    slices_number = len(slices)
    # phases_number = len(os.listdir(os.path.join(series_nr_path, slice_folders[0])))
    # Conversion and merging
    file_name = slices[0]
    dicom_path = series_nr_path + '/' + file_name


    d = dicom.read_file(dicom_path)
    X = d.Columns
    Y = d.Rows
    T = 1
    dx = float(d.PixelSpacing[1])
    dy = float(d.PixelSpacing[0])
    # dz = d.SpacingBetweenSlices
    # dz = d.SliceThickness
    # try:
                    #dz = d.SpacingBetweenSlices

    #except:
                    #dz = d.SliceThickness

    Z = slices_number

    # The coordinate of the upper-left voxel
    pos_ul = np.array([float(x) for x in d.ImagePositionPatient])
    pos_ul[:2] = -pos_ul[:2]

    # Image orientation
    axis_x = np.array([float(x) for x in d.ImageOrientationPatient[:3]])
    axis_y = np.array([float(x) for x in d.ImageOrientationPatient[3:]])
    axis_x[:2] = -axis_x[:2]
    axis_y[:2] = -axis_y[:2]

    file_name = slices[1]
    dicom_path = series_nr_path + '/' + file_name

    d2 = dicom.read_file(dicom_path)
    pos_ul2 = np.array([float(x) for x in d2.ImagePositionPatient])
    pos_ul2[:2] = -pos_ul2[:2]
    axis_z = pos_ul2 - pos_ul
    axis_z = axis_z / np.linalg.norm(axis_z)

    try:
                    dz = d.SpacingBetweenSlices

    except:
                    dz = abs(d2.SliceLocation - d.SliceLocation)

    # Affine matrix which converts the voxel coordinate to world coordinate
    affine = np.eye(4)
    affine[:3,0] = axis_x * dx
    affine[:3,1] = axis_y * dy
    affine[:3,2] = axis_z * dz
    affine[:3,3] = pos_ul

    # check whether slice vec is orthogonal to iop vectors
    dv = np.dot((axis_z * dz), np.cross(axis_x, axis_y))
    qfac = np.sign(dv)
    if np.abs(qfac*dv - np.round(dz, 7)) > 1e-6:
        sys.stderr.write("Non-orthogonal volume!\n");
    
    # The 4D volume
    volume = np.zeros((X, Y, Z, T), dtype='float32')

    # Go through each slice
    for z in range(0,Z):
        # Read the images
        file_name = slices[z]
        dicom_path = series_nr_path + '/' + file_name
        d = dicom.read_file(dicom_path)
        volume[:, :, z, 0] = d.pixel_array.transpose()

    # Write the 4D volume
    filename = "{}.nii.gz".format(output_name)
    nim = nib.Nifti1Image(volume, affine)
    nim.header['sform_code'] = 1
    out_file = os.path.join(nii_output_dir, filename)
    print(out_file)
    nib.save(nim, out_file)
    return list(volume.shape)

def get_instance_nr(filename):
    # Assuming the filenames are of the format "num_value_name.dcm"
    return int(filename.split('_')[0])

def get_series_nr(instance_nr):
      return int(instance_nr.split('_')[0])


def convert_cine(output_name, nii_output_dir, series_nr_path):
    os.chdir(series_nr_path)
    slice_folders = sorted(os.listdir(series_nr_path), key=get_series_nr)
    slice_number = len(slice_folders)
    phases_number = len(os.listdir(os.path.join(series_nr_path, slice_folders[0])))

    # Conversion and merging
    dir_name = slice_folders[0]
    dicom_path = series_nr_path + '/' + dir_name
    dicom_name = dicom_path + '/' + os.listdir(dicom_path)[0]
    # dicom_name = dir_name + '/' + '*' + dir_name +'_001.dcm'
    dicom_file = glob.glob(dicom_name)[0]
    d = dicom.read_file(dicom_file)
    X = d.Columns
    Y = d.Rows
    try:
                    T = d.CardiacNumberOfImages
    except:
                    T = phases_number

    dx = float(d.PixelSpacing[1])
    dy = float(d.PixelSpacing[0])
    # dz = d.SpacingBetweenSlices
    # dz = d.SliceThickness
    # try:
                    #dz = d.SpacingBetweenSlices

    #except:
                    #dz = d.SliceThickness

    Z = slice_number

    # The coordinate of the upper-left voxel
    pos_ul = np.array([float(x) for x in d.ImagePositionPatient])
    pos_ul[:2] = -pos_ul[:2]

    # Image orientation
    axis_x = np.array([float(x) for x in d.ImageOrientationPatient[:3]])
    axis_y = np.array([float(x) for x in d.ImageOrientationPatient[3:]])
    axis_x[:2] = -axis_x[:2]
    axis_y[:2] = -axis_y[:2]

    # Read the dicom file at the second time point
    dicom_name = dicom_path + '/' + os.listdir(dicom_path)[1]
    dicom_file = glob.glob(dicom_name)[0]
    d2 = dicom.read_file(dicom_file)
    dt = (d2.TriggerTime - d.TriggerTime) * 1e-3


    # Read the dicom file at the second slice
    dir_name = slice_folders[1]
    dicom_path = series_nr_path + '/' + dir_name
    dicom_name = dicom_path + '/' + os.listdir(dicom_path)[0]
    
    dicom_file = glob.glob(dicom_name)[0]
    d2 = dicom.read_file(dicom_file)
    pos_ul2 = np.array([float(x) for x in d2.ImagePositionPatient])
    pos_ul2[:2] = -pos_ul2[:2]
    axis_z = pos_ul2 - pos_ul
    axis_z = axis_z / np.linalg.norm(axis_z)

    try:
                    dz = d.SpacingBetweenSlices

    except:
                    dz = abs(d2.SliceLocation - d.SliceLocation)
    
    # Affine matrix which converts the voxel coordinate to world coordinate
    affine = np.eye(4)
    affine[:3,0] = axis_x * dx
    affine[:3,1] = axis_y * dy
    affine[:3,2] = axis_z * dz
    affine[:3,3] = pos_ul

    # check whether slice vec is orthogonal to iop vectors
    dv = np.dot((axis_z * dz), np.cross(axis_x, axis_y))
    qfac = np.sign(dv)
    if np.abs(qfac*dv - np.round(dz, 7)) > 1e-6:
        sys.stderr.write("Non-orthogonal volume!\n");
    
    # The 4D volume
    volume = np.zeros((X, Y, Z, T), dtype='float32')

    # Go through each slice
    for z in range(0,Z):
        # Read the images
        for t in range(0, T):
            dir_name = slice_folders[z]
            dicom_path = series_nr_path + '/' + dir_name
            sorted_file_names = sorted(os.listdir(dicom_path), key=get_instance_nr)
            dicom_name = dicom_path + '/' + sorted_file_names[t]

            # dir_name = slice_folders[z]
            # suffix = '_{0:03d}.dcm'.format(t + 1)
            # dicom_name = dir_name + '/' + '*' + dir_name + suffix
            dicom_file = glob.glob(dicom_name)[0]
            d = dicom.read_file(dicom_file)
            volume[:, :, z, t] = d.pixel_array.transpose()

    # Write the 4D volume
    filename = "{}.nii.gz".format(output_name)
    nim = nib.Nifti1Image(volume, affine)
    nim.header['pixdim'][4] = dt
    nim.header['sform_code'] = 1
    out_file = os.path.join(nii_output_dir, filename)
    print(out_file)
    nib.save(nim, out_file)

    return list(volume.shape)

