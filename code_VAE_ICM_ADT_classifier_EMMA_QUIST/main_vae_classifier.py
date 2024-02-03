import torch
import argparse

import numpy as np
import pandas as pd
from multiprocessing import freeze_support
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from models.vae_one import VAE_One
from models.vae_multi import VAE_Multi, initialize_weights
from attri_vae.attri_vae_model.model import ConvVAE
from data.data_loading_funcs import *

from utils.utils_general import *
from utils.utils_reconstruction_evaluation import aumc_write_batch, emidec_write_batch


from train_test_vae_classifier import vae_train, vae_test, save_ckp
from attri_vae.attri_vae_train_test_funcs import attri_test, attri_train


def main_train(args):
    set_seed(args.seed)

    saving_dir = args.dataset + '_' + args.model_version + '/' + args.model_type + '_' + args.version 
    if args.model_type == 'vae_mlp':
        saving_dir = saving_dir + '_' + args.label 
    if args.debugging:
        saving_dir = 'debug/' + saving_dir
    saving_dir =  saving_dir + '/seed_%s_lr_%.4f_e_%d_l_%s_alpha_%.2f_beta_%.2f_do_%.2f/'%(args.seed, args.lr, args.max_epochs, args.latent_size, args.alpha, args.beta, args.dropout) + '/'

    args.logdir = args.logdir + saving_dir

    writer = SummaryWriter(args.logdir)

    print('Device: ', args.device)
    print(f"Parameter List: \n beta = {args.beta}\n epoch number is {args.max_epochs} and latent dimension is {args.latent_size} ")
    args.checkpoint_dir = args.logdir

    check_dir(args.checkpoint_dir)
    save_defaults_to_file(args)

    start_epoch = 0

    best_test_loss = np.finfo('f').max
    best_acc = np.finfo('f').min
    best_auc = np.finfo('f').min
    best_test_loss_epoch = -1
    img_size = args.win_size[0]

    if args.model_version == 'VAE_One':
        train, test = vae_train, vae_test
        model = VAE_One(args, img_size=img_size).to(args.device)
        # weights already initialized 
    if args.model_version == 'VAE_Multi':
        train, test = vae_train, vae_test
        model = VAE_Multi(args).to(args.device)
        model.apply(initialize_weights) ### Initializa the weights
    if args.model_version == 'AttriVAE':
        train, test = attri_train, attri_test
        img_size = args.win_size[0]
        model = ConvVAE(image_channels=args.IMG, h_dim=args.HDIM, latent_size= args.latent_size, n_filters_ENC=args.n_filters_ENC, n_filters_DEC=args.n_filters_DEC, img_size=img_size).to(args.device)  
        model.apply(initialize_weights) ### Initializa the weights

    # DataLoaders    
    if args.dataset == 'AUMC':
        train_dataset, val_dataset = load_dataset_aumc(args, train=True)
        write_image_batch = aumc_write_batch
    if args.dataset == 'EMIDEC':
        train_dataset, val_dataset  = load_dataset_emidec(args, train=True)
        write_image_batch = emidec_write_batch

    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=args.num_workers)
    

    # Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min',
                                                        factor=0.2,
                                                        patience=20,
                                                        min_lr=1e-6)

    if args.load_checkpoint:
        resume_path = args.checkpoint_dir + "checkpoint.pth"
        print('=> loading checkpoint %s' % args.load_checkpoint)
        checkpoint = torch.load(resume_path)
        start_epoch = checkpoint['epoch'] + 1
        best_test_loss = checkpoint['best_test_loss']
        best_test_loss_epoch = checkpoint['best_test_loss_epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print('=> loaded checkpoint %s' % args.load_checkpoint)

    for epoch in range(start_epoch, args.max_epochs):
        if args.model_type == 'vae':
            train_images, train_recons, train_metrics = train(args, model, train_loader, optimizer)
            test_images, test_recons, test_metrics = test(args, model, val_loader) 
        if args.model_type == 'vae_mlp':
            train_images, train_recons, train_metrics  = train(args, model, train_loader, optimizer)
            test_images, test_recons, test_metrics = test(args, model, val_loader) 

        scheduler.step(test_metrics['beta_test_loss'])

        if args.log:
            if args.model_type == 'vae' or args.model_type == 'vae_mlp':
                writer.add_scalar("Beta (epoch)", args.beta, epoch)
                writer.add_scalar("KL Loss training (epoch)", train_metrics['train_kl'], epoch)
                writer.add_scalar("Recon Loss training (epoch)", train_metrics['train_recon'], epoch)
                writer.add_scalar("KL Loss validation (epoch)", test_metrics['test_kl'], epoch)
                writer.add_scalar("Recon Loss validation (epoch)", test_metrics['test_recon'], epoch)
            if args.model_type == 'vae_mlp':
                writer.add_scalar("MLP loss training (epoch)", train_metrics['train_class'], epoch)
                writer.add_scalar("MLP loss Validation (epoch)", test_metrics['test_class'], epoch) 
                writer.add_scalar("Acc training", train_metrics['train_acc'], epoch)
                writer.add_scalar("Acc validation", test_metrics['test_acc'], epoch)
                writer.add_scalar("ROC training", train_metrics['train_roc'], epoch)
                writer.add_scalar("ROC validation", test_metrics['test_roc'], epoch)

            if args.version == 'sigma' or args.version == 'optimal_sigma':
                writer.add_scalar("train_log_sigma", train_metrics['train_sigma'], epoch)
            writer.add_scalar("Beta Loss training (epoch)", train_metrics['beta_train_loss'], epoch)
            writer.add_scalar("Beta Loss validation (epoch)", test_metrics['beta_test_loss'], epoch)
            writer.add_scalar("Pure Loss training (epoch)", train_metrics['pure_train_loss'], epoch)
            writer.add_scalar("pure Loss validation (epoch)", test_metrics['pure_test_loss'], epoch)
            writer.add_scalar("SSIM score training (epoch)", train_metrics['ssim_score'], epoch)
            writer.add_scalar("SSIM validation (epoch)", test_metrics['test_ssim_score'], epoch)

            # Create a grid of images
            write_image_batch(train_images.detach().cpu().numpy(), train_recons.detach().cpu().numpy(), 4, 'train', writer, epoch, img_size)
            write_image_batch(test_images.detach().cpu().numpy(), test_recons.detach().cpu().numpy(), 4, 'validate', writer, epoch, img_size)
        
        if args.model_type == 'vae':
            print('Epoch [%d/%d] Train KL loss: %.3f Train Recon loss: %.3f' % (epoch + 1, args.max_epochs, train_metrics['train_kl'], train_metrics['train_recon']))
            print('Epoch [%d/%d] Validation KL loss: %.3f Val Recon loss: %.3f' % (epoch + 1, args.max_epochs, test_metrics['test_kl'], test_metrics['test_recon']))
        if args.model_type == 'vae_mlp':
            print('Epoch [%d/%d] Train KL loss: %.3f Train Recon loss: %.3f Train MLP loss: %.3f' % (epoch + 1, args.max_epochs, train_metrics['train_kl'], train_metrics['train_recon'], train_metrics['train_class']))
            print('Epoch [%d/%d] Validation KL loss: %.3f Val Recon loss: %.3f Val MLP loss: %.3f' % (epoch + 1, args.max_epochs, test_metrics['test_kl'], test_metrics['test_recon'], test_metrics['test_class']))
            print(f"Training Accuracy (%) {train_metrics['train_acc']} AUC: {train_metrics['train_roc']} in epoch {epoch+1}")
            print(f"Validation Accuracy (%) {test_metrics['test_acc']} AUC: {test_metrics['test_roc']} in epoch {epoch+1}")
        print('Epoch [%d/%d] train loss: %.3f validation loss: %.3f' % (epoch + 1, args.max_epochs, train_metrics['beta_train_loss'], test_metrics['beta_test_loss']))

        ############### Save checkpoint 
        checkpoint = { 'epoch': epoch + 1, 'best_test_loss': best_test_loss, 'best_test_loss_epoch': best_test_loss_epoch, 'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
        save_ckp(checkpoint, args.checkpoint_dir)

        ### Save the model with the best loss ###########################
        if args.model_type == 'vae_mlp':
            if test_metrics['test_roc'] > best_auc:
                best_auc = test_metrics['test_roc']
                best_test_loss_epoch = epoch+1
                print(f"Best loss is achieved {best_auc} in the epoch: {best_test_loss_epoch}")
                torch.save(model.state_dict(), args.checkpoint_dir + "best_auc_loss_model.pth")
            if test_metrics['test_acc'] > best_acc:
                best_acc = test_metrics['test_acc']
                best_metric_epoch = epoch + 1
                print(f"Best accuracy is achieved {best_acc} in the epoch: {best_metric_epoch}")
                torch.save(model.state_dict(), args.checkpoint_dir +  "best_acc_model.pth")

        if test_metrics['test_recon'] < best_test_loss:
            best_test_loss = test_metrics['test_recon']
            best_test_loss_epoch = epoch+1
            print(f"Best loss is achieved {best_test_loss} in the epoch: {best_test_loss_epoch}")
            torch.save(model.state_dict(), args.checkpoint_dir + "best_recon_loss_model.pth")

    writer.close()
    if args.model_type == 'vae_mlp':
        evaluate(args, "auc_loss")
        args.first_write = False
        evaluate(args, "acc")
    evaluate(args, "recon_loss")
    args.first_write = False
    print("OUTCOME")
    print(f"Best loss of {best_test_loss} was achieved in epoch {best_test_loss_epoch}")

def evaluate(args, type): 

    args.eval_dir = args.logdir + type + '/'
    check_dir(args.eval_dir)
    if args.dataset == 'AUMC':
        if args.external_val:
            args.base_dir = '../processed_emidec_dataset' 
            test_dataset  = load_dataset_emidec(args, train=False, external_val=True)
            args.base_dir = '../LGE_clin_myo_mask_1_iter_highertresh'
            write_image_batch = emidec_write_batch
        else:
            test_dataset = load_dataset_aumc(args, train=False)
            write_image_batch = aumc_write_batch
    if args.dataset == 'EMIDEC':
        if args.external_val:
            args.base_dir = '../LGE_clin_myo_mask_1_iter_highertresh' 
            test_dataset = load_dataset_aumc(args, train=False,  external_val=True)
            args.base_dir = '../emidec_processed_dataset' 
            write_image_batch = aumc_write_batch
        else:
            test_dataset  = load_dataset_emidec(args, train=False)
            write_image_batch = emidec_write_batch


    test_loader = DataLoader(test_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=True,
                        pin_memory=True,
                        num_workers=args.num_workers)

    img_size = args.win_size[0]
    if args.model_version == 'VAE_One':
        test = vae_test
        model = VAE_One(args, img_size=args.img_size).to(args.device)
    if args.model_version == 'VAE_Multi':
        model = VAE_Multi(args).to(args.device)
        test = vae_test
    if args.model_version == 'AttriVAE':
        test = attri_test
        model = ConvVAE(image_channels=args.IMG, h_dim=args.HDIM, latent_size= args.latent_size, n_filters_ENC=args.n_filters_ENC, n_filters_DEC=args.n_filters_DEC, img_size=img_size).to(args.device)  
    

    model.load_state_dict(torch.load(args.checkpoint_dir + "best_{}_model.pth".format(type)))
    if args.model_type == 'vae':
        test_images, test_recons, metrics = test(args, model, test_loader, test=True) 
    if args.model_type == 'vae_mlp':
        test_images, test_recons, metrics = test(args, model, test_loader, test=True) 

    print("test_loss:", metrics['beta_test_loss'])
    print("Mutual Information:", metrics['mi'])
    print("MMD Score:", metrics['mmd'])
    if args.log:
        writer = SummaryWriter(args.eval_dir)
        writer.add_scalar("beta Loss test", metrics['beta_test_loss'], 0)
        writer.add_scalar("pure Loss test", metrics['pure_test_loss'], 0)
        if args.model_type == 'vae' or args.model_type == 'vae_mlp':
            writer.add_scalar("KL Loss test", metrics['test_kl'], 0)
            writer.add_scalar("Recon Loss test", metrics['test_recon'], 0)
        if args.model_type == 'vae_mlp':
            writer.add_scalar("MLP loss test", metrics['test_class'], 0) 
            writer.add_scalar("Acc test", metrics['test_acc'], 0)
            writer.add_scalar("ROC test", metrics['test_roc'], 0)
            writer.add_scalar("F1 test", metrics['test_f1'], 0)
            writer.add_scalar("Recall test", metrics['test_recall'], 0)
            writer.add_scalar("Precision test", metrics['test_precision'], 0)
            writer.add_scalar("acc_form_matri test", metrics['test_acc_form_matrix'], 0)

        writer.add_scalar("SSIM", metrics['test_ssim_score'], 0)
        writer.add_scalar("MSE", metrics['mse'], 0)
        writer.add_scalar("Mutual Information", metrics['mi'], 0)
        writer.add_scalar("MMD", metrics['mmd'], 0)
        # Create a grid of images
        write_image_batch(test_images.detach().cpu().numpy(), test_recons.detach().cpu().numpy(), 1, 'test', writer, 0, args.win_size[0])
    if args.write_results:
        dict_dir = args.save_results_path + 'results.csv'
        header_data_frame = pd.DataFrame(metrics, index=[args.eval_dir])
        if args.first_write:
            header_data_frame.to_csv(dict_dir, index=True, sep=';', mode='a', header=True)
        else:
            header_data_frame.to_csv(dict_dir, index=True, sep=';', mode='a', header=False)
    del model

def main():
    dataset = 'AUMC' #choices= ['EMIDEC', 'AUMC']
    model_version = 'VAE_Multi' # choices=['VAE_Multi', 'VAE_One', 'AttriVAE']
    label = 'ICM'
    logdir = '/Experiemnt_x/'
    parser = argparse.ArgumentParser(description="Arguments for training the VAE model")
    # hardware
    parser.add_argument("--gpus", default=0)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--device", type=str, default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    # reproducability
    parser.add_argument("--logdir", type=str, default='../' + logdir )
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--load_check_dir", type=str, default="")
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument("--debugging", type=bool, default=False)
    parser.add_argument('--search', type=bool, default=True)
    parser.add_argument("--log", type=bool, default=True)
    parser.add_argument("--write_results", type=bool, default=True)
    parser.add_argument("--save_results_path", type=str, default='../RESULTS/' + logdir)
    parser.add_argument('--first_write', type=bool, default=True)

    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--external_val', type=bool, default=True)
    parser.add_argument('--class_weights', type=bool, default=False)
    parser.add_argument('--oversample', type=bool, default=False)
    if dataset == 'AUMC':
        parser.add_argument("--train_ratio", type=float, default=0.75) 
        if label == 'AppropriateTherapy' or label == 'Mortality':
            parser.add_argument("--base_dir", type=str, default="../ICM_LGE_clin_myo_mask_labeled_filterd_1y")
        else:
            parser.add_argument("--base_dir", type=str, default="../LGE_clin_myo_mask_1_iter_highertresh")
        parser.add_argument("--image_name", type=str, default="LGE")
        parser.add_argument('--norm_perc', type=tuple, default=(1, 95))
        parser.add_argument('--hist', type=int, default=256)
        parser.add_argument("--roi_crop", type=str, default="fitted", choices=["fixed", "fitted"])
        parser.add_argument("--masked", type=bool, default=True)
        parser.add_argument("--label", type=str, default=label, choices=['ICM', 'NICM', 'HCM', 'DCM', 'MyocardialInfarction', 'AppropriateTherapy', 'Mortality'])
    if dataset == 'EMIDEC':
        parser.add_argument("--train_ratio", type=float, default=0.75) 
        parser.add_argument('--norm_perc', type=tuple, default=(1, 95))
        parser.add_argument('--hist', type=int, default=256)
        parser.add_argument("--label", type=str, default=label)
        parser.add_argument("--image_name", type=str, default="LGE")
        parser.add_argument("--base_dir", type=str, default="../emidec_processed_dataset")
        parser.add_argument("--roi_crop", type=str, default="fitted", choices=["fixed", "fitted"])
        parser.add_argument("--masked", type=bool, default=True)

    
    # learning hyperparameters
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--is_L1", type=bool, default=False)
    parser.add_argument("--annealing", type=bool, default=False)
    
    parser.add_argument('--model_version', type=str, default=model_version) 
    if model_version == 'VAE_Multi' or model_version == 'VAE_One':
        # model
        parser.add_argument("--model_type", type=str, default='vae_mlp', choices=['vae, vae_mlp'])
        parser.add_argument('--version', type=str, default='optimal_sigma', choices=['mse', 'gaussian', 'sigma', 'beta', 'optimal_sigma'])
        parser.add_argument("--win_size", type=int, default=(64,64,64)) # Dimensions of the input to the net
        parser.add_argument("--IMG", type=int, default=1)
        parser.add_argument("--blocks", type=list, default=(1, 2, 4, 8))

        # VAE params
        parser.add_argument('--sample', type=bool, default=False)
        parser.add_argument('--upsample', type=bool, default=True)
        parser.add_argument("--latent_size", type=int, default=64)
        parser.add_argument("--bottleneck_channel", type=int, default=0, help="number of channels before unflatten,only used in RES VAE") #if IMG ==2 then 2, if IMG ==1 then 1
        parser.add_argument("--n_filters", type=int, default=16)
        parser.add_argument('--final_dropout', type=bool, default=0.0)
        parser.add_argument('--dropout', type=bool, default=0.0)

        # learning hyperparameters
        parser.add_argument("--recon_param", type=float, default= 1)
        parser.add_argument("--ssim_indicator", type=int, default=1)
        parser.add_argument("--ssim_scalar", type=int, default=1)
        parser.add_argument("--alpha", type=float, default=1.0, help='alpha * mlp_loss')
        parser.add_argument("--beta", type=float, default= 3, help='beta parameter for KL-term in understanding beta-VAE')  
    
    
    if model_version == 'AttriVAE':
        """ Attri-VAE parameters """
        parser.add_argument("--model_type", type=str, default='vae_mlp')
        parser.add_argument("--win_size", type=int, default=(80,80,80)) # Dimensions of the input to the net
        parser.add_argument("--IMG", type=int, default=1)
        parser.add_argument('--num_classes', type=int, default=1)
        parser.add_argument("--HDIM", type=int, default=96, help="dim of the FC layer before the latent space (mu and sigma)") # LVAE + MLP = h_dims = [96, 48, 24] #
        parser.add_argument("--latent_size", type=int, default=64)
        parser.add_argument("--unflatten_channel", type=int, default=2, help="number of channels before unflatten") #if IMG ==2 then 2, if IMG ==1 then 1
        parser.add_argument("--dim_start_up_decoder", type=int, default=[5,5,5], help="in unflatten (inside the decoder)") # the dimensions will be [1, unflatten_channel, dim_startup_decoder]
        parser.add_argument("--n_filters", type=int, default=16, help='for AUMC dataset 8 filters')
        parser.add_argument("--n_filters_ENC", type=int, default=[8, 16, 32, 64, 2])
        parser.add_argument("--n_filters_DEC", type=int, default=[64, 32, 16, 8, 4, 2])
        parser.add_argument("--recon_param", type=float, default= 1.0)
        parser.add_argument('--version', type=str, default='beta')
        parser.add_argument("--alpha", type=float, default=1.0, help='alpha * mlp_loss')
        parser.add_argument("--beta", type=float, default= 2.0, help='beta parameter for KL-term in understanding beta-VAE')  # [0.02, 0.001, 0.0001], -> (0.55, 0.0275, 0.00275)  # this values was taken from biffi et. al. = beta value of the beta-VAE 

    args = parser.parse_args()

    main_train(args)

if __name__ == "__main__":
    freeze_support()
    main()