import os
import numpy as np
from emidec_data_features.utils_data_process import *
# from emidec_utils.utils_interpretable_features import myocardialmass, myocardial_thickness

def emidec_data_load_feature_extract(training_path, cropped=False): 
    # emidec_columns = ["GBS", "SEX", "AGE", "TOBACCO", "OVERWEIGHT", "ART_HYPERT", "DIABETES", "FHCAD", "ECG", "TROPONIN", "KILLIP", "FEVG", "NTProNBP"]

    if cropped:
        path_cases = os.path.join(training_path, 'myo_lv/binMask')
        cases = [('_').join(f.split('_')[:2]) for f in os.listdir(path_cases)]
    else:
        info_files = [f for f in os.listdir(training_path) if f.endswith('.txt')] # these are info files
        info_files.sort()
        cases =  [f for f in os.listdir(training_path) if not f.endswith('.txt')]# the others are the patient folders

    ####  CLINICAL INFORMATION ##################
    # clinical_info_training = load_clinical_info(training_path, info_files, emidec_columns, folders)


    # # Preprocess the values
    # cleanup_nums = {"SEX":     {" F": 1, " M": 0},
    #             "OVERWEIGHT": {" N": 0, " Y": 1},
    #             "ART_HYPERT": {" N": 0, "N":0, " Y": 1},
    #             "DIABETES": {" N": 0, " Y": 1},
    #             "FHCAD" : {" N": 0, " Y": 1},
    #             "ECG" : {" N": 0, " Y": 1}}


    # clinical_info_training = clinical_info_training.replace(cleanup_nums)

    healthy_cohort =  [param for param in cases if param.split("_")[1].startswith('N')] # the ones start with N (after _) are healthy
    minf_cohort = [param for param in cases if param.split("_")[1].startswith('P')] # the ones start with P are MINF
    nb_healthy =  len(healthy_cohort)
    nb_minf = len(minf_cohort)
    print(f"There are {nb_healthy} healthy and {nb_minf} heart attack patients in the dataset.")

    if cropped: 
        images_minf, Label_minf  = load_preprocessed_files(training_path, minf_cohort, cohort_name='MINF')
        images_nor, Label_nor = load_preprocessed_files(training_path, healthy_cohort, cohort_name='NOR')
    else:
        images_minf, Label_minf = preprocess_files(training_path, minf_cohort, cohort_name='MINF', Train=True)
        images_nor, Label_nor = preprocess_files(training_path, healthy_cohort, cohort_name='NOR', Train=True)

    ### all cases from emidec training dataset
    images_train = images_nor + images_minf 
    labels_train = Label_nor + Label_minf 

    indices = np.arange(len(images_train))

    return images_train, labels_train, indices #, clinical_info_training
