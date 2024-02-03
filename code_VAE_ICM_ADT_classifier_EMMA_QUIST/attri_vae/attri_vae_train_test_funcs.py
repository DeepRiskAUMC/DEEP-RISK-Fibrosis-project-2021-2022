### Training #####
import torch
from pytorch_msssim import ssim

from attri_vae.attri_vae_model.loss_functions import *
from utils.utils_general import *
from utils.utils_reconstruction_evaluation import get_all_recon_metrics
from utils.utils_classifier_evaluation import roc_cm, auroc_accuracy


def attri_train(args, epoch, model, train_loader, optimizer, use_AR_LOSS=False):
    model.train()
    kl_last = 0
    recon_last =0
    train_loss = 0
    mlp_last = 0
    ssim_score_total = 0
    metrics = {}
    iter = 0
    output_list = []
    label_list = []
    # for batch_idx, (data, label, rad_, clinic, fname) in enumerate(train_loader):
    for batch_idx, (data, label) in enumerate(train_loader):
        iter = iter + 1
        data = data.to(args.device)
        label = label.to(args.device)
        # rad_ = rad_.to(args.device)
        # clinic_ = clinic.to(args.device)
        optimizer.zero_grad()
        # 1. Forward #############################################################################
        recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(data)
        #########################################################
        #z_sampled_es : sampled using reparam trick implemented from mu and logvar
        #z_tilde : sampled using torch reparam trick from z_dist
        ########################################################
        
        # 2. Loss ################################################################################
        recon_loss = reconstruction_loss(recon_batch, data, args.recon_param, dist = 'gaussian')
        mlp_loss = mlp_loss_function(label, out_mlp, args.alpha)
        
        kl_loss1, kl_loss2 = KL_loss(mu, logvar, z_dist, prior_dist, args.beta, c=0.0)
        loss = recon_loss + mlp_loss +  kl_loss2

        # 2.1 Weight Regularization #############################################################
        ## L1 Regularization 
        if (args.is_L1==True):
      
            l1_crit = torch.nn.L1Loss(reduction="sum")
            weight_reg_loss = 0
            for param in model.parameters():
                weight_reg_loss += l1_crit(param,target=torch.zeros_like(param))

            fctr = 0.00005
            loss += fctr * weight_reg_loss
        else:
            pass

        # 3. Backward ###########################################################################
        loss.backward()
        # 4. Update #############################################################################
        optimizer.step()

        ssim_score = ssim(data, recon_batch, data_range=1, nonnegative_ssim=True)

        train_loss += loss
        kl_last += kl_loss2
        recon_last += recon_loss
        mlp_last += mlp_loss
        ssim_score_total += ssim_score

        output_list.append(out_mlp)
        label_list.append(label)


    metrics['beta_train_loss'] = train_loss/iter
    metrics['pure_train_loss'] = 0
    metrics['train_recon'] = recon_loss/iter
    metrics['train_kl'] = kl_last/iter
    metrics['ssim_score'] = ssim_score_total/iter


    train_outputs = torch.concat(output_list)
    train_labels = torch.concat(label_list)
    acc, roc_total = auroc_accuracy(train_labels, train_outputs, roc=True)
    metrics['train_class'] = mlp_last/iter
    metrics['train_acc'] = acc
    metrics['train_roc'] = roc_total


    return data, recon_batch, metrics
        

### Validating ####
def attri_test(args, epoch, model, test_loader, test=False):
    model.eval()
    test_loss = 0
    iter = 0
    kl_value = 0
    recon_value = 0
    mlp_value = 0
    test_ssim_score_total = 0
    metrics = {}
    test_output_list = []
    test_label_list = []
    total_mmd = 0
    total_mi = 0
    total_mse = 0
    with torch.no_grad():
        for batch_idx, (data_test, label) in enumerate(test_loader):
            iter = iter + 1
            
            data_test = data_test.to(args.device)
            label = label.to(args.device)
            recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(data_test)
            
            recon_loss = reconstruction_loss(recon_batch, data_test,args.recon_param, dist = 'gaussian')
            kl_loss1, kl_loss2 = KL_loss(mu, logvar, z_dist, prior_dist, args.beta, c=0.0)
            mlp_loss = mlp_loss_function(label, out_mlp, args.alpha)
            loss_ = recon_loss + mlp_loss +  kl_loss2

            test_loss += loss_

            test_ssim_score = ssim(data_test, recon_batch, data_range=1, nonnegative_ssim=True)


            kl_value += kl_loss2
            recon_value += recon_loss
            mlp_value += mlp_loss
            test_ssim_score_total += test_ssim_score

            test_output_list.append(out_mlp)
            test_label_list.append(label)

            if test:
                mmd, mi, _, mse = get_all_recon_metrics(data_test, recon_batch, args.win_size)
                total_mmd += mmd
                total_mse += mse
                total_mi += mi
                
    test_outputs = torch.concat(test_output_list)
    test_labels = torch.concat(test_label_list)
    # if args.version == 'beta':
    metrics['beta_test_loss'] = test_loss/iter 
    metrics['pure_test_loss'] = 0
    metrics['test_recon'] = recon_value/iter
    metrics['test_kl'] = kl_value/iter
    metrics['test_ssim_score'] = test_ssim_score_total/iter 
    
    if test:
        metrics['mmd'] = total_mmd/iter
        metrics['mi'] = total_mi/iter
        metrics['mse'] = total_mse/iter
        precision, recall, f1_score, accuracy = roc_cm(args, test_outputs, test_labels, sigmoid_applied=True)
        metrics['test_precision'] = precision
        metrics['test_recall'] = recall
        metrics['test_f1_score'] = f1_score
        metrics['test_accuracy'] = accuracy
    else:
        metrics['test_acc'] = roc_cm(args, test_outputs, test_labels, sigmoid_applied=True)


    _, test_roc_total = auroc_accuracy(test_labels, test_outputs, roc=True)
    metrics['test_class'] = mlp_value/iter
    metrics['test_roc'] = test_roc_total

    return data_test, recon_batch, metrics