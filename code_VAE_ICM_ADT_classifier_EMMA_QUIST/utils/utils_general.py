import os
import json
import random
import torch
import argparse
import numpy as np

def check_dir(path_dir):
    if os.path.isdir(path_dir):
        pass
    else:
       os.makedirs(path_dir)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def save_defaults_to_file(args):
    file_name = args.checkpoint_dir + 'defaults.json'
    args.device = str(args.device)
    defaults = vars(args)
    with open(file_name, 'w') as file:
        json.dump(defaults, file, indent=2)  

def load_args_from_json(json_file):
    with open(json_file, 'r') as file:
        args_dict = json.load(file)
    return argparse.Namespace(**args_dict)

def save_ckp(state,checkpoint_dir):
    f_path = checkpoint_dir + 'checkpoint.pth'
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer, sceduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    sceduler.load_state_dicht(checkpoint['sceduler'])
    return model, optimizer, checkpoint['epoch']

def early_stopping(counter, train_loss, validation_loss, min_delta):
    if (validation_loss - train_loss) > min_delta:
        counter += 1
        if counter % 10 == 0 or counter == 25:
            print('early stopping counter at:', counter)
    return counter



    