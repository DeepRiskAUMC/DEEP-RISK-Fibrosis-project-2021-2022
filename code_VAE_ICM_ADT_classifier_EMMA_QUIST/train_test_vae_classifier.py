
import torch
import torch.nn as nn
from pytorch_msssim import ssim
from torch.autograd import Variable

from utils.utils_general import *
from models.loss_functions import *
from utils.utils_classifier_evaluation import auroc_accuracy, roc_cm
from utils.utils_reconstruction_evaluation import get_all_recon_metrics


def vae_train(args, model, train_loader, optimizer):
    model.train()
    beta_loss_total = 0
    pure_loss_total = 0
    kl_total = 0
    recon_total = 0
    mlp_total = 0
    ssim_score_total = 0
    iter = 0
    roc_total = 0
    metrics = {}
    mlp = False
    output_list = []
    label_list = []
    for images, label in train_loader:

        iter = iter + 1
        images = Variable(images.to(args.device, dtype=torch.float))
        
        if args.model_type == 'vae_mlp':
            mlp = True
            label = Variable(label).to(args.device)
            label = torch.nn.functional.one_hot(label, num_classes=2)

        optimizer.zero_grad()

        output_dict = model(images)

        if mlp:
            class_prob = output_dict['pred']

        recon_batch, mu, logvar =  output_dict['img'],  output_dict['mu'],  output_dict['log_var']


        scale_factor = 1/(args.batch_size)

        if args.version == 'beta':
            recon_loss = reconstruction_loss(recon_batch, images)
        elif args.version == 'optimal_sigma':
            recon_loss, log_sigma = sigma_reconstruction_loss(recon_batch, images)
        kl_loss = KL_loss(mu, logvar)
        
        kl_loss = kl_loss * scale_factor
        recon_loss = recon_loss * scale_factor

        beta_kl_loss = args.beta * kl_loss
        param_recon_loss = args.recon_param * recon_loss

        pure_loss = recon_loss +  kl_loss
        beta_vae_loss = param_recon_loss +  beta_kl_loss

        ssim_score = ssim(images, recon_batch, data_range=1, nonnegative_ssim=True)

        if mlp:
            if type(args.class_weights) != bool:
                args.class_weights = (args.class_weights).to(args.device)
            mlp_loss = mlp_loss_function(label, class_prob, args.class_weights)
            alpha_mlp_loss = args.alpha * mlp_loss

            beta_vae_loss = beta_vae_loss + alpha_mlp_loss
            pure_loss = pure_loss + mlp_loss

            mlp_total += alpha_mlp_loss.item() 

        ## L1 Regularization 
        if (args.is_L1==True):
      
            l1_crit = nn.L1Loss(reduction="sum")
            weight_reg_loss = 0
            for param in model.parameters():
                weight_reg_loss += l1_crit(param,target=torch.zeros_like(param))

            fctr = 0.00005
            beta_vae_loss += fctr * weight_reg_loss

    
        beta_vae_loss.backward()
        optimizer.step()

        beta_loss_total += beta_vae_loss.item()
        pure_loss_total += pure_loss.item()
        recon_total += recon_loss.item() 
        ssim_score_total += ssim_score.item()
        kl_total += kl_loss.item() 

        if mlp:
            output_list.append(class_prob)
            label_list.append(label)

    metrics['beta_train_loss'] = beta_loss_total/iter
    metrics['pure_train_loss'] = pure_loss_total/iter
    metrics['train_recon'] = recon_total/iter
    metrics['train_kl'] = kl_total/iter
    metrics['ssim_score'] = ssim_score_total/iter

    if mlp:
        train_outputs = torch.concat(output_list)
        train_labels = torch.concat(label_list)
        acc, roc_total = auroc_accuracy(train_labels, train_outputs)
        metrics['train_class'] = mlp_total/iter
        metrics['train_acc'] = acc
        metrics['train_roc'] = roc_total
    if args.version == 'sigma' or args.version == 'optimal_sigma':
        metrics['train_sigma'] = log_sigma

    return images, recon_batch, metrics
        

def vae_test(args, model, test_loader, test=False):
    model.eval()
    iter = 0
    test_beta_loss_total = 0
    test_pure_loss_total = 0
    test_ssim_score_total = 0
    test_kl_total = 0
    test_recon_total = 0
    test_mlp_total = 0 
    test_acc_total = 0
    test_roc_total = 0
    total_mi = 0
    total_mmd = 0
    total_mse = 0
    mlp = False
    metrics = {}
    output_list = []
    label_list = []
    with torch.no_grad():
        for images_test, label_test in test_loader:
            iter = iter + 1
            images_test = Variable(images_test.to(args.device, dtype=torch.float))

            if args.model_type == 'vae_mlp':
                mlp = True
                label_test = Variable(label_test).to(args.device)
                label_test = torch.nn.functional.one_hot(label_test, num_classes=2)

            output_dict = model(images_test)
            if mlp:
                class_prob = output_dict['pred']

            recon_batch, mu, logvar =  output_dict['img'],  output_dict['mu'],  output_dict['log_var']  

            scale_factor = 1/(args.batch_size)

            if args.version == 'beta':
                recon_loss = reconstruction_loss(recon_batch, images_test)
            elif args.version == 'optimal_sigma':
                recon_loss, log_sigma = sigma_reconstruction_loss(recon_batch, images_test)

            kl_loss = KL_loss(mu, logvar)
            kl_loss = kl_loss * scale_factor
            recon_loss = recon_loss * scale_factor

            beta_kl_loss = args.beta * kl_loss
            param_recon_loss = args.recon_param * recon_loss


            test_pure_loss = recon_loss +  kl_loss
            test_beta_vae_loss = beta_kl_loss +  param_recon_loss

            test_ssim_score = ssim(images_test, recon_batch, data_range=1, nonnegative_ssim=True)

            if mlp:
                mlp_loss = mlp_loss_function(label_test, class_prob)
                alpha_mlp_loss = mlp_loss * args.alpha
                test_beta_vae_loss = test_beta_vae_loss + alpha_mlp_loss
                test_pure_loss = test_pure_loss + mlp_loss
                test_mlp_total += alpha_mlp_loss.item() 

                output_list.append(class_prob)
                label_list.append(label_test)
 

            test_beta_loss_total += test_beta_vae_loss.item()
            test_pure_loss_total += test_pure_loss.item()
            test_recon_total += recon_loss.item() 
            test_ssim_score_total += test_ssim_score.item()
            test_kl_total += kl_loss.item() 
            
            if test:
                mmd, mi, _, mse = get_all_recon_metrics(images_test, recon_batch, args.win_size)
                total_mmd += mmd
                total_mse += mse
                total_mi += mi
                
    metrics['beta_test_loss'] = test_beta_loss_total/iter 
    metrics['pure_test_loss'] = test_pure_loss_total/iter 
    metrics['test_recon'] = test_recon_total/iter
    metrics['test_kl'] = test_kl_total/iter
    metrics['test_ssim_score'] = test_ssim_score_total/iter 
    
    if test:
        metrics['mmd'] = total_mmd/iter
        metrics['mi'] = total_mi/iter
        metrics['mse'] = (total_mse/iter).item()
    if mlp:
        test_outputs = torch.concat(output_list)
        test_labels = torch.concat(label_list)
        if test:
            precision, recall, f1_score, matrics_acc = roc_cm(args, test_outputs, test_labels)
            metrics['test_acc_form_matrix'] = matrics_acc
            metrics['test_f1'] = f1_score
            metrics['test_recall'] = recall
            metrics['test_precision'] = precision

        test_acc_total, test_roc_total = auroc_accuracy(test_labels, test_outputs)
        metrics['test_class'] = test_mlp_total/iter
        metrics['test_acc'] = test_acc_total.item()
        metrics['test_roc'] = test_roc_total.item()

    if args.version == 'sigma' or args.version == 'optimal_sigma':
        if not test:
            metrics['test_sigma'] = log_sigma.detach().cpu()

    return images_test, recon_batch, metrics
    

