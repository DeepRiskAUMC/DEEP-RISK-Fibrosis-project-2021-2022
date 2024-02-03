import os
import subprocess
import convert_utils as convert


class DicomToNiftiConverters:

    def __init__(self, nii_input_dir, nii_output_dir):
        self.nii_input_dir = nii_input_dir
        self.nii_output_dir = nii_output_dir

    def dcm2nii(nii_input_dir, nii_output_dir):
        '''
            Chris Rorden's dcm2nii :: 12 12 2012
            reading preferences file /Users/rorden/.dcm2nii/dcm2nii.ini
            Either drag and drop or specify command line options:
            dcm2nii
            OPTIONS:
            -a Anonymize [remove identifying information]: Y,N = Y
            -b load settings from specified inifile, e.g. '-b C:\set\t1.ini'
            -c Collapse input folders: Y,N = Y
            -d Date in filename [filename.dcm -> 20061230122032.nii]: Y,N = Y
            -e events (series/acq) in filename [filename.dcm -> s002a003.nii]: Y,N = Y
            -f Source filename [e.g. filename.par -> filename.nii]: Y,N = N
            -g gzip output, filename.nii.gz [ignored if '-n n']: Y,N = Y
            -i ID in filename [filename.dcm -> johndoe.nii]: Y,N = N
            -m manually prompt user to specify output format [NIfTI input only]: Y,N = Y
            -n output .nii file [if no, create .hdr/.img pair]: Y,N = Y
            -o Output Directory, e.g. 'C:\TEMP' (if unspecified, source directory is used)
            -p Protocol in filename [filename.dcm -> TFE_T1.nii]: Y,N = Y
            -r Reorient image to nearest orthogonal: Y,N
            -s SPM2/Analyze not SPM5/NIfTI [ignored if '-n y']: Y,N = N
            -v Convert every image in the directory: Y,N = Y
            -x Reorient and crop 3D NIfTI images: Y,N = N
            You can also set defaults by editing /Users/rorden/.dcm2nii/dcm2nii.ini
            EXAMPLE: dcm2nii -a y /Users/Joe/Documents/dcm/IM_0116
        '''
        # nii_output_dir = nii_output_dir + '\dcm2nii'

        if not os.path.exists(nii_output_dir):
            os.makedirs(nii_output_dir)
        

        output_name = '%s_%f' 

        # %f : input folder name, 
        # %i : patient ID, (DRAUMC0007)
        # %q : sequence name, (GR) # Not sure what this means
        # %s : series number, (15, 16, 17 ..)
        # %z : series description, (tfi2d1_15) # not sure what this means either

        # Create the dcm2niix command
        # python C:\\Users\\emquist\\miniconda3\\envs\\parsing_mri\\Scripts\\
        command = f"dcm2niix -b n -ba n -z y -f {output_name} -o {nii_output_dir} {nii_input_dir}" # -m y to group, doesnt do anything

        # Run the command as a subprocess
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check if the command was successful
        if result.returncode == 0:
            print('Conversion completed successfully.')
            print('Output:', result.stdout.decode('utf-8'))
        else:
            print('Conversion failed.')
            print('Error:', result.stderr.decode('utf-8'))
        pass

    def dicom2nifti(nii_input_dir, nii_output_dir):
        """
            usage: dicom2nifti [-h] [-G] [-I] [-S] [-r] [-o RESAMPLE_ORDER] [-p RESAMPLE_PADDING] [-M] [-C] [-R] input_directory output_directory

            dicom2nifti, convert dicom files into nifti format.

            positional arguments:
            input_directory       directory containing dicom files, can be nested
            output_directory      directory to store the nifti files

            options:
            -h, --help            show this help message and exit
            -G, --allow-gantry-tilting
                                    allow the conversion of gantry tilted data (this will be reflected in the affine matrix only unless resampling is enabled)
            -I, --allow-inconsistent-slice-increment
                                    allow the conversion of inconsistent slice increment data (this will result in distorted images unless resampling is enabled)
            -S, --allow-single-slice
                                    allow the conversion of a single slice (2D image)
            -r, --resample        resample gantry tilted data to an orthogonal image or inconsistent slice increment data to a uniform image
            -o RESAMPLE_ORDER, --resample-order RESAMPLE_ORDER
                                    order of the spline interpolation used during the resampling (0 -> 5) [0 = NN, 1 = LIN, ....]
            -p RESAMPLE_PADDING, --resample-padding RESAMPLE_PADDING
                                    padding value to used during resampling to use as fill value
            -M, --allow-multiframe-implicit
                                    allow the conversion of multiframe data with implicit vr transfer syntax (this is not guaranteed to work)
            -C, --no-compression  disable gzip compression and write .nii files instead of .nii.gz
            -R, --no-reorientation
                                    disable image reorientation (default: images are reoriented to LAS orientation)
        
        """


        # nii_output_dir = nii_output_dir + '\dicom2nifti'
        if not os.path.exists(nii_output_dir):
            os.makedirs(nii_output_dir)
        print(nii_output_dir)
        # command = f"python H:\\Coding_env\\conda\\envs\\my_env\\Scripts\\dicom2nifti -I -S -r -o 1000 {nii_input_dir} {nii_output_dir}"
        
        command = f"python C:\\Users\\emquist\\miniconda3\\envs\\parsing_mri\\Scripts\\dicom2nifti -I -S -r {nii_input_dir} {nii_output_dir}"
        # Run the command as a subprocess
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
        print(result.stderr.decode('utf-8'))
    

    def cardio_nifti(output_name, nii_input_dir, nii_output_dir, series_type):
        series_nr_path = nii_input_dir
        convert_method = 'nan'
        nii_shape = 'nan'
        if series_type == 'LGE':
            if not os.path.isfile(series_nr_path):
                slices_folder = os.listdir(series_nr_path)
                if len(slices_folder) > 2:
                    ordered_slice_spacing_dict = convert.slice_spacing_dict(series_nr_path, series_type)
                    if len(ordered_slice_spacing_dict) > 2:
                        series_nr_path, new = convert.reorder_dicoms(series_nr_path, ordered_slice_spacing_dict, series_type)
                        if new:
                            try:
                                nii_shape = convert.convert_lge(output_name, nii_output_dir, series_nr_path)
                                convert_method =  'cardio_nifti'
                            except Exception as ex:
                                print(ex)
                                # DicomToNiftiConverters.dicom2nifti(series_nr_path, nii_output_dir)
                                nii_shape = [0,0,0, 0]
                                convert_method = ex
                        else:
                            convert_method = 'none'
                            nii_shape = [0, 0, 0, 0]
            #                 DicomToNiftiConverters.dicom2nifti(nii_input_dir, nii_output_dir)
            #                 nii_shape = [0,0,1, 1]
            #                 convert_method = 'dicom_2_niffti'

                    else:
                        convert_method = 'none'
                        nii_shape = [0, 0, 0, 0]
            #             DicomToNiftiConverters.dicom2nifti(nii_input_dir, nii_output_dir)
            #             nii_shape = [0,0,1, 1]
            #             convert_method = 'dicom_2_niffti'
                else:
                    convert_method = 'none'
                    nii_shape = [0, 0, 0, 0]
            #         DicomToNiftiConverters.dicom2nifti(nii_input_dir, nii_output_dir)
            #         nii_shape = [0,0,1, 1]
            #         convert_method = 'dicom_2_niffti'      
            else:
                convert_method = 'none'
                nii_shape = [0, 0, 0, 0]
            #         DicomToNiftiConverters.dicom2nifti(nii_input_dir, nii_output_dir)
            #         nii_shape = [0,0,len(slices_folder), 1]
            #         convert_method = 'dicom_2_niffti'

        if series_type == 'CINE':
            slices_folder = os.listdir(series_nr_path)
            if len(slices_folder) > 2:
                ordered_slice_spacing_dict = convert.slice_spacing_dict(series_nr_path, series_type)

                if len(ordered_slice_spacing_dict) > 2:
                    series_nr_path, new = convert.reorder_dicoms(series_nr_path, ordered_slice_spacing_dict, series_type)
                    if new:
                        try:
                            nii_shape = convert.convert_cine(output_name, nii_output_dir, series_nr_path)
                            convert_method = 'cine_cardio_nifti'
                            
                        except Exception as ex:
                            print(ex)
                            # DicomToNiftiConverters.dcm2nii(nii_input_dir, nii_output_dir)
                            convert_method = ex
                            nii_shape = [0,0, 1, len(os.listdir(series_nr_path +'/'+ os.listdir(series_nr_path)[0]))]
                    else:
                        convert_method = 'none'
                        nii_shape = [0, 0, 0, 0]
            #             DicomToNiftiConverters.dcm2nii(nii_input_dir, nii_output_dir)
            #             convert_method = 'dcm2nii'
            #             nii_shape = [0,0, 1, len(os.listdir(series_nr_path +'/'+ slices_folder[0]))]
                else:
                    convert_method = 'none'
                    nii_shape = [0, 0, 0, 0]
            #         DicomToNiftiConverters.dcm2nii(nii_input_dir, nii_output_dir)
            #         convert_method = 'dcm2nii'
            #         nii_shape = [0,0, 1, len(os.listdir(series_nr_path +'/'+ slices_folder[0]))]
            else:
                convert_method = 'none'
                nii_shape = [0, 0, 0, 0]
            #     DicomToNiftiConverters.dcm2nii(nii_input_dir, nii_output_dir)
            #     nii_shape = [0,0, 1, len(os.listdir(series_nr_path +'/'+ slices_folder[0]))]
            #     convert_method = 'dcm2nii'

        return convert_method, nii_shape



