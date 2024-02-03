import os
import os.path
import argparse
from pathlib import Path
from multiprocessing import freeze_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from utils.utils_general import *
from data.data_loading_funcs import load_latent_classifier_dataset, oversample_smote
from models.vae_multi import VAE_Multi, Latent_Classifier, initialize_weights
from train_test_latent_classifier import *


def main_latent_calssifier(mlp_args):
    # DataLoaders   
    best_type = 'recon_loss'
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    set_seed(mlp_args.seed)

    vae_args = load_args_from_json(mlp_args.vae_dir + 'defaults.json')

    test_vae = VAE_Multi(vae_args).to(mlp_args.device)
    test_vae.load_state_dict(torch.load(mlp_args.vae_dir + 'best_{}_model.pth'.format(best_type)))

    mlp_args.logdir = mlp_args.vae_dir + mlp_args.label + '_' + mlp_args.classifier + '/'
    if mlp_args.classifier == 'deep_classifier' or mlp_args.classifier == 'small_classifier'or mlp_args.classifier == 'smaller_classifier':
        mlp_args.logdir = mlp_args.logdir + str(mlp_args.seed) + '_' + mlp_args.classifier + '_batch_32_' + str(round(mlp_args.lr, 6))  + '_conv_' + str(mlp_args.conv) + '_' + str(mlp_args.max_epochs) + '_' + str(mlp_args.mlp_dropout)+ '_over_' + '_weight_' + str(mlp_args.times_weights)  +'/'
    writer = SummaryWriter(mlp_args.logdir)

    latent_full_dataset = load_latent_classifier_dataset(mlp_args, train=True)
    latent_full_loader = DataLoader(latent_full_dataset,
                            batch_size=mlp_args.latent_batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=mlp_args.num_workers)

    latent_test_dataset = load_latent_classifier_dataset(mlp_args, train=False)
    latent_test_loader = DataLoader(latent_test_dataset,
                            batch_size=mlp_args.latent_batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=mlp_args.num_workers)

    mus_train, labels_train, mus_test, labels_test = obtain_latents(mlp_args, test_vae, latent_full_loader, latent_test_loader)
    
    # Convert lists to tensors
    train_latent = torch.stack(mus_train).detach().cpu()
    train_labels = torch.stack(labels_train).detach().cpu()
    test_latent = torch.stack(mus_test).detach().cpu()
    test_labels = torch.stack(labels_test).detach().cpu()
    if not mlp_args.conv:
        m = nn.AvgPool3d(3, stride=2)
        train_latent = m(train_latent).view(len(train_labels), -1)
        test_latent = m(test_latent).view(len(test_labels), -1)

    if mlp_args.classifier == 'deep_classifier' or mlp_args.classifier == 'small_classifier' or mlp_args.classifier == 'smaller_classifier':
        # Create a TensorDataset
        
        train_val_dataset = TensorDataset(train_latent, train_labels)
        test_dataset = TensorDataset(test_latent, test_labels)
        # Define the train and validation split (adjust the test_size as needed)
        train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.2, random_state=mlp_args.seed)
        if mlp_args.oversample:
            train_dataset = oversample_smote(train_dataset, mlp_args.seed)
        train_loader = DataLoader(train_dataset,
                                batch_size=mlp_args.class_batch_size,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=mlp_args.num_workers)
        val_loader = DataLoader(val_dataset,
                                batch_size=mlp_args.latent_batch_size,
                                shuffle=False,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=mlp_args.num_workers)
        test_loader = DataLoader(test_dataset,
                            batch_size=mlp_args.latent_batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=mlp_args.num_workers)

        if mlp_args.classifier == 'deep_classifier':
            model = Latent_Classifier(mlp_args, n_out=2, sigmoid=False, conv=mlp_args.conv)    
        model.apply(initialize_weights)

        if mlp_args.class_weights:
            y_for_weights = train_labels.numpy()
            # Calculate class weights to determine SMOTE sampling strategy
            mlp_args.class_weights = compute_class_weight('balanced', classes=np.unique(y_for_weights), y=y_for_weights) 
            print('class weights', mlp_args.class_weights)
            mlp_args.class_weights = torch.tensor(mlp_args.class_weights)
            mlp_args.class_weights[1] = mlp_args.class_weights[1]*mlp_args.times_weights
        test_loss, precision, recall, f1_score, accuracy, auc = train_deep_classifier(mlp_args, train_loader, val_loader, test_loader, model, writer, n_out=2)
    if mlp_args.classifier == 'SVM':
        model  = None
        test_loss, precision, recall, f1_score, accuracy, auc = train_test_svm(mlp_args, train_latent, train_labels, test_latent, test_labels, model, writer, n_out=2)
    return test_loss, precision, recall, f1_score, accuracy, auc



def main():
    label = 'AppropriateTherapy'
    model_type = 'vae_mlp'
    mlp_parser = argparse.ArgumentParser(description="Arguments for training the VAE model")
    mlp_parser.add_argument("--num_workers", type=int, default=24)
    # reproducability
    mlp_parser.add_argument("--seed", type=int, default=42)
    mlp_parser.add_argument("--best_model", type=str, default='auc_loss')
    if label == 'AppropriateTherapy' or label == 'Mortality':
        mlp_parser.add_argument("--base_dir", type=str, default="../ICM_LGE_clin_myo_mask_labeled_filterd_1y")
    else:
        mlp_parser.add_argument("--base_dir", type=str, default="../LGE_clin_myo_mask_1_iter_highertresh")
    if model_type == 'vae':
        mlp_parser.add_argument('--vae_dir', type=str, default="../EXP_VAE/AUMC_VAE_Multi/vae_optimal_sigma/seed_3_lr_0.0001_e_500_l_64_alpha_1.00_beta_3.00/")
    else:
        mlp_parser.add_argument('--vae_dir', type=str, default= '../FINAL_FOR_LATENTS/AUMC_VAE_Multi/vae_mlp_optimal_sigma_ICM/seed_28_lr_0.0001_e_500_l_64_alpha_500.00_beta_3.00_do_0.00/')
    mlp_parser.add_argument("--image_name", type=str, default="LGE")
    mlp_parser.add_argument('--class_weights', type=bool, default=True)
    mlp_parser.add_argument('--times_weights', type=int, default=4)
    mlp_parser.add_argument('--oversample', type=bool, default=True)
    mlp_parser.add_argument('--conv', type=bool, default=True)
    mlp_parser.add_argument('--norm_perc', type=tuple, default=(1, 95))
    mlp_parser.add_argument('--hist', type=int, default=256)
    mlp_parser.add_argument("--model_type", type=str, default=model_type, choices=['ae, vae, vae_mlp'])
    mlp_parser.add_argument("--masked", type=bool, default=True)
    mlp_parser.add_argument("--roi_crop", type=str, default="fitted", choices=["fixed", "fitted"])
    mlp_parser.add_argument("--label", type=str, default=label)
    mlp_parser.add_argument("--external_val", type=str, default='False')
    mlp_parser.add_argument('--classifier', type=str, default='deep_classifier')
    mlp_parser.add_argument('--svm_search', type=bool, default=False)
    mlp_parser.add_argument("--win_size", type=int, default=(64,64,64)) # Dimensions of the input to the net
    mlp_parser.add_argument("--IMG", type=int, default=1)

    mlp_parser.add_argument("--latent_size", type=int, default=64)
    mlp_parser.add_argument("--blocks", type=list, default=(1, 2, 4, 8))
    mlp_parser.add_argument('--mlp_dropout', type=float, default=0.1)
    mlp_parser.add_argument('--dropout', type=float, default=0.0)
    mlp_parser.add_argument("--max_epochs", type=int, default=200)
    mlp_parser.add_argument("--latent_batch_size", type=int, default=16)
    mlp_parser.add_argument("--class_batch_size", type=int, default=32)
    mlp_parser.add_argument("--lr", type=float, default=5e-5)
    mlp_parser.add_argument("--device", type=str, default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    mlp_args = mlp_parser.parse_args()
    seeds = [3, 14, 28, 42, 56]#3, 14, 28
    first_write = True
    for seed in seeds:
        label = 'AppropriateTherapy'
        if label == 'AppropriateTherapy' or label == 'Mortality':
            mlp_parser.base_dir = "../ICM_LGE_clin_myo_mask_labeled_filterd_1y"
        else:
            mlp_parser.base_dir = "../LGE_clin_myo_mask_1_iter_highertresh"
        mlp_args.label = label
        mlp_args.lr = 5e-5
        mlp_args.max_epochs = 200
        mlp_args.times_weights = 2
        mlp_args.classifier = 'SVM'
        mlp_args.seed = seed
        mlp_args.class_weights = True
        test_loss, precision, recall, f1_score, accuracy, auc = main_latent_calssifier(mlp_args)
        metrics = {'precision':precision, 'recall':recall, 'f1':f1_score, 'accuracy':accuracy, 'auc':auc}
        dict_dir = 'ADT_VAE_do_0.1_results.csv'
        header_data_frame = pd.DataFrame(metrics, index=[str(seed)])
        if first_write:
            header_data_frame.to_csv(dict_dir, index=True, sep=';', mode='a', header=True)
        else:
            header_data_frame.to_csv(dict_dir, index=True, sep=';', mode='a', header=False)
        first_write = False
    print(test_loss, precision, recall, f1_score, accuracy, auc)




if __name__ == "__main__":
    freeze_support()
    main()