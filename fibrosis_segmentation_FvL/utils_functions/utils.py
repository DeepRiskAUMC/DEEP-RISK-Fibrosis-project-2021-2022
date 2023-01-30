import os
import cv2
from collections import OrderedDict
import SimpleITK as sitk

def get_model_version_no(log_dir, segment_task='myocard', cross_validate=False, method='lightning_logs'):
    if cross_validate:
        folder_path = os.path.join(log_dir, 'cross_validation', segment_task, method)
    else:
        folder_path = os.path.join(log_dir, segment_task, method)
    obj_names = os.listdir(folder_path)
    highest_nr = -1
    for fn in obj_names:
        number = fn.split('_')[-1]
        if number.split('.')[-1] == 'txt':
            continue
        number = int(number)
        if number > highest_nr:
            highest_nr = number
    # print(data_paths)
    return highest_nr+1

# def get_model_version_no_classification(log_dir, model_name='CNN', method='lightning_logs'):
def get_model_version_no_classification(logging_dir, method='lightning_logs'):
    folder_path = os.path.join(logging_dir, method)
    try:
        obj_names = os.listdir(folder_path)
    except:
        os.mkdir(folder_path)
        return 0
    highest_nr = -1
    for fn in obj_names:
        number = fn.split('_')[-1]
        if number.split('.')[-1] == 'txt':
            continue
        number = int(number)
        if number > highest_nr:
            highest_nr = number
    # print(data_paths)
    return highest_nr+1

def get_data_paths(data_dir):
    """
    Get get image data paths
    :param data_dir: the root data directory
    :return: data paths
    """
    data_paths = []
    obj_names = next(os.walk(data_dir))[2]
    for fn in obj_names:
        path = os.path.join(data_dir, fn)
        data_paths.append(path)
    data_paths = sorted(data_paths)
    return data_paths

def resize_image(img, size):
    img_new = cv2.resize(img, dsize=size, interpolation=cv2.INTER_LINEAR)

def get_smallest_size(data_folder):
    train_folder = os.path.join(data_folder, 'train', 'LGE_niftis')
    test_folder = os.path.join(data_folder, 'test', 'LGE_niftis')
    train_paths = get_data_paths(train_folder)
    test_paths = get_data_paths(test_folder)
    all_paths = train_paths + test_paths

    smallest_height, smallest_width = 1e6, 1e6
    for img_path in all_paths:
        LGE_image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        _, h, w = LGE_image.squeeze().shape
        if h < smallest_height:
            smallest_height = h
        if w < smallest_width:
            smallest_width = w
    return smallest_height, smallest_width

def load_pretrained_layers(new_model, pretrained_model):
    pretrained_dict = pretrained_model.state_dict()
    model_dict = new_model.state_dict()

    # print('pretrained_dict')
    # print(pretrained_dict.keys())
    # print('model_dict')
    # print(model_dict.keys())

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    not_loaded = [k for k in model_dict if k not in pretrained_dict]
    print('Modules for which the weights were not loaded:', not_loaded)
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    new_model.load_state_dict(pretrained_dict)
    return new_model

def rename_pretrained_layers(old_state_dict):
    new_state_dict = OrderedDict()
    for param_tensor, values in old_state_dict.items():
        if param_tensor.split('.')[0] == 'model' and param_tensor.split('.')[1] != 'model':
            continue
        else:
            new_state_dict[param_tensor] = values
    return new_state_dict

def get_print_statement(args):
    if 'CNN' in args.model:
        print_statement = f"Dataset: {args.dataset} | model: {args.model} | layers_conv: {args.hidden_layers_conv} | layers_lin: {args.hidden_layers_lin} | resize: {args.resize} | normalize: {args.normalize} | augmentation: {args.transformations} | segmentation checkpoint: {args.fib_checkpoint} | loss_function: {args.loss_function} | use_val_weights: {args.use_val_weights} | lr: {args.lr} | batch_size: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed}"
    elif 'encoder' in args.model:
        print_statement_a = f"Dataset: {args.dataset} | prediction task: {args.prediction_task} | model: {args.model} | flatten_or_maxpool: {args.flatten_or_maxpool} | use_MRI_features: {args.use_MRI_features} | layers_lin: {args.hidden_layers_lin} | resize: {args.resize} | normalize: {args.normalize} | augmentation: {args.transformations}"
        print_statement_b = f"loss_function: {args.loss_function} | use_val_weights: {args.use_val_weights}| lr: {args.lr} | batch_size: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed}"
        print_statement_c = f"segmentation checkpoint: {args.fib_checkpoint}"
        print_statement = print_statement_a + "\n" + print_statement_b + "\n" + print_statement_c
    elif 'GNN' in args.model:
        print_statement_a = f"Model: {args.model} | labels: {args.prediction_task} | node_attributes: {args.node_attributes} | num_myo: {args.num_myo} | num_fib: {args.num_fib} | edges_per_node: {args.edges_per_node} | hidden_channels_gcn: {args.hidden_channels_gcn}"
        print_statement_b = f"num_gcn_layers: {args.num_gcn_layers} | update_pos: {args.update_pos} | dropout: {args.dropout} | dist_info: {args.dist_info} | hidden_units_fc: {args.hidden_units_fc} | kernel_size: {args.kernel_size} | distance_measure: {args.distance_measure}"
        print_statement_c = f"loss_function: {args.loss_function} | use_val_weights: {args.use_val_weights} | lr: {args.lr} | batch_size: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed}"
        print_statement_d = f"batch_norm: {args.batch_norm} | instance_norm: {args.instance_norm} |probs_path: {args.probs_path} | myo_feat_path: {args.myo_feat_path} | fib_feat_path: {args.fib_feat_path}"
        print_statement = print_statement_a + "\n" + print_statement_b + "\n" + print_statement_c + "\n" + print_statement_d    
    elif 'densenet_padding' in args.model:
        print_statement = f"Dataset: {args.dataset} | model: {args.model} | dataset: {args.dataset} | labels: {args.prediction_task} | dropout: {args.dropout} | resize: {args.resize} | normalize: {args.normalize} | augmentation: {args.transformations} | myocard checkpoint: {args.myo_checkpoint} | loss_function: {args.loss_function} | lr: {args.lr} | batch_size: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed}"
    elif 'multi_input' in args.model:
        print_statement = f"Dataset: {args.dataset} | model: {args.model} | labels: {args.prediction_task} | resize: {args.resize} | normalize: {args.normalize} | augmentation: {args.transformations} | segmentation checkpoint: {args.fib_checkpoint} | densenet checkpoint: {args.densenet_checkpoint} | loss_function: {args.loss_function} | lr: {args.lr} | batch_size: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed}"
    return print_statement

if __name__ == '__main__':
    smallest_height, smallest_width = get_smallest_size('L:\\basic\diva1\Onderzoekers\DEEP-RISK\DEEP-RISK\CMR DICOMS\Roel&Floor\sample_niftis\labels\labels_model_testing')
    print(smallest_height, smallest_width)