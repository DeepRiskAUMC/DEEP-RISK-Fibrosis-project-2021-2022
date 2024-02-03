import os
import os.path
import argparse
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
import cv2
from torch.nn import functional as F
from multiprocessing import freeze_support

from torch.utils.data import DataLoader

from collections import OrderedDict

import numpy as np

import matplotlib.pyplot as plt



import monai
#from monai.networks.utils import eval_mode
from monai.transforms import (ScaleIntensity
)

from utils.utils_general import *
from data.data_loading_funcs import load_latent_classifier_dataset, oversample_smote
from models.vae_multi import VAE_Multi
from train_test_latent_classifier import *


"""
code used from https://github.com/iremcetin/Attention_generation
"""
# gradcam.py

# Overlay Original image and GradCAM heatmap
def superimposed_image_def(cam_img, original_img, alpha = 0.6):
    original_img = cv2.merge((original_img, original_img, original_img))
    cam_img = (cam_img - cam_img.min()) / (
               cam_img.max() - cam_img.min()
    ) *255
    
    # Convert to Heatmap ---- JET COLORMAP

    cam_img = cv2.applyColorMap(np.uint8(cam_img), cv2.COLORMAP_JET)

    original_img = np.uint8(
        (original_img - original_img.min())
        / (original_img.max() - original_img.min())
        * 255
    )

###### Superimpose Heatmap on Image Data #####################################
   
    superimposed_image = cv2.addWeighted(original_img, alpha, cam_img, (1-alpha), 0.0)

    return superimposed_image, cam_img, original_img
class PropBase(object):

    def __init__(self, model, target_layer, cuda=True):
        self.model = model
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        self.model.eval()
        self.target_layer = target_layer
        self.outputs_backward = OrderedDict()
        self.outputs_forward = OrderedDict()
        self.set_hook_func()

    def set_hook_func(self):
        raise NotImplementedError


    # set the target class as one others as zero. use this vector for back prop added by Lezi
    def encode_one_hot_batch(self, z, mu, logvar, mu_avg, logvar_avg):# icetin: why this returns mu??????
        one_hot_batch = torch.FloatTensor(z.size()).zero_() # all zeros
       
        return mu # icetin: when this returns one_hot_batch score_fc =0 otherwise different score values for each case...
        # [maybe] return one_hotbatch when you train with normal and test with abnormal ==> abnormality detection : NO: results are all black
        # return mu o.w.

    def forward(self, x): # icetin: run the VAE and return the outputs
        #self.preds = self.model(x) # icetin: ? No need
        output_dict = self.model(x)
        self.image_size = x.size(-1) 
        class_prob = output_dict['pred']
        sigmoid = torch.nn.Sigmoid()
        self.out_mlp  = sigmoid(class_prob)
        recon_batch, self.mu, self.logvar =  output_dict['img'],  output_dict['mu'],  output_dict['log_var']
        # recon_batch, self.mu, self.logvar, self.out_mlp = self.model(x) 
        return recon_batch, self.mu, self.logvar, self.out_mlp

    # icetin: back prop the one_hot signal
    def backward(self, mu, logvar, mu_avg, logvar_avg):
        self.model.zero_grad()
        z = self.model.encoder.sample(mu, logvar).cuda() # use the mu and logvar from forward pass and sample z ( then what is the z's dim???)
        print(f" dimension of the latent sample: {z.size()}") # [1, 64]
        one_hot = self.encode_one_hot_batch(z, mu, logvar, mu_avg, logvar_avg)# this returns mu

        if self.cuda:
            one_hot = one_hot.cuda()
        #flag = 2 # icetin: why??? -- never goes into first condition (flag==1) but in the paper--> attention maps are calculated using relu???
        flag=1
        if flag == 1:
            self.score_fc = torch.sum(F.relu(one_hot * mu))
            print(f" flag==1 and score_fc is : {self.score_fc}") #icetin: check the calculated score
        else:
            self.score_fc = torch.sum(one_hot)
            print(f" flag !=1 and score_fc is : {self.score_fc}")
        self.score_fc.backward(retain_graph=True)

    def get_conv_outputs(self, outputs, target_layer): # icetin: this function outputs the selected conv layer's output : Feature maps
        for key, value in outputs.items():
            for module in self.model.named_modules():
                # print(module)
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        
                        return value
        raise ValueError('invalid layer name: {}'.format(target_layer))

class GradCAM(PropBase):

    def set_hook_func(self):
        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_out[0].cpu()

        def func_f(module, input, f_output):
            self.outputs_forward[id(module)] = f_output

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)
            module[1].register_forward_hook(func_f)

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()

    def compute_gradient_weights(self):
        self.grads = self.normalize(self.grads) #icetin:  grads.size() = [1,2,5,5,5] [batch_size, selected_layers_output_channel, 5,5,5]
        print(f" grads inside compute_gradient_weights: {self.grads.size()}")
        self.map_size = self.grads.size()[2:] # map_size = [1,2, 5,5,5]
        #self.map_size = self.grads.size()[1:]
        print(f" map_size inside compute_gradient_weights: {self.map_size}")
        self.weights = nn.AvgPool3d(self.map_size)(self.grads) # 
        print(f" size of weights (compute_gradient_weights) after AvgPool3d {self.weights.size()}")
        #self.weights = nn.AvgPool3d(self.map_size[2])(self.grads)

    def generate(self):
        # get gradient
        self.grads = self.get_conv_outputs(
            self.outputs_backward, self.target_layer)
        # compute weithts based on the gradient
        self.compute_gradient_weights()

        # get activation
        self.activation = self.get_conv_outputs(
            self.outputs_forward, self.target_layer)

        self.weights.volatile = False
        print(f"SIZE OF ACTIVATION  : {self.activation.size()}")
        #self.activation = self.activation[:, :, :, :] # icetin: activation = [1,2,5,5,5] - first one is the batch_size, second one is the # of feature maps in the selected layer
        #self.activation = self.activation[None, :,:,:,:,:] #icetin: [1,1,2,5,5,5 ] #  this does not work 

        #self.weights = self.weights[:, :, :, :] #icetin: weights = [2,1,1,1]
        #self.weights = self.weights[:, None, :,:,:,:] icetin: not working
        print(f"SIZE OF WEIGHTS : {self.weights.size()}")
        #torch.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor
        #icetin:
        #gcam = F.conv3d(self.activation, weight = (self.weights.cuda().unsqueeze(0)), stride=1, padding=0, groups=len(self.weights))
        print(f" groups = len weights : {len(self.weights)}")
        #icetin: no need # self.weights = self.weights.squeeze(dim=0)
        gcam = F.conv3d(self.activation, weight = (self.weights.cuda()), stride=1, padding=0, groups=len(self.weights)) # groups = # feature maps?
        print(f"gcam size after F.conv3d before squeeze: {gcam.size()}")
        #icetin: gcam = gcam.squeeze(dim=0)
        
        gcam = F.upsample(gcam, (self.image_size, self.image_size, self.image_size), mode="trilinear") #icetin: Upsample the attention map to the size of the image
        print(f"gcam size after upsample {gcam.size()}")
        gcam = torch.abs(gcam)

        return gcam


def default_normalizer(x) -> np.ndarray:
    """
    A linear intensity scaling by mapping the (min, max) to (1, 0).

    N.B.: This will flip magnitudes (i.e., smallest will become biggest and vice versa).
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    scaler = ScaleIntensity(minv=0.0, maxv=1.0)
    x = [scaler(x) for x in x]
    return np.stack(x, axis=0)

def backward_hook_(module, grad_input, grad_output):
  global gradients # refers to the variable in the global scope
  print('Backward hook running...')
  gradients = grad_output
  # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
  print(f'Gradients size: {gradients[0].size()}') 
  # We need the 0 index because the tensor containing the gradients comes
  # inside a one element tuple.

def forward_hook_(module, args, output):
  global activations # refers to the variable in the global scope
  print('Forward hook running...')
  activations = output
  # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
  print(f'Activations size: {activations.size()}')

def main_latent_calssifier(mlp_args):
    # DataLoaders   
    best_type = 'recon_loss'
    mlp_args.seed = 18
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    set_seed(mlp_args.seed)

    vae_args = load_args_from_json(mlp_args.vae_dir + 'defaults.json')

    test_vae = VAE_Multi(vae_args).to(mlp_args.device)
    test_vae.load_state_dict(torch.load(mlp_args.vae_dir + 'best_{}_model.pth'.format(best_type)))
    test_vae.eval()

    # defines two global scope variables to store our gradients and activations

    mlp_args.logdir = mlp_args.vae_dir + mlp_args.label + '_' + mlp_args.classifier + '/'
    if mlp_args.classifier == 'deep_classifier' or mlp_args.classifier == 'small_classifier'or mlp_args.classifier == 'smaller_classifier':
        mlp_args.logdir = mlp_args.logdir + str(mlp_args.seed) + '_' + mlp_args.classifier + '_batch_32_' + str(round(mlp_args.lr, 6))  + '_conv_' + str(mlp_args.conv) + '_' + str(mlp_args.max_epochs) + '_' + str(mlp_args.mlp_dropout)+ '_over_' + '_weight_' + str(mlp_args.times_weights)  +'/'
    # writer = SummaryWriter(mlp_args.logdir)

    latent_full_dataset = load_latent_classifier_dataset(mlp_args, train=True)
    latent_full_loader = DataLoader(latent_full_dataset,
                            batch_size=8,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=mlp_args.num_workers)
    latent_test_dataset = load_latent_classifier_dataset(mlp_args, train=False)

    latent_test_loader = DataLoader(latent_test_dataset,
                            batch_size=mlp_args.latent_batch_size,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=mlp_args.num_workers)

    # backward_hook = backward_hook_
    # # forward_hook = forward_hook_
    # m = nn.AvgPool3d(3, stride=2)
    # # batch_x =8
    # latent_z_flat = []
    # latent_z_pool = []
    # targets = []
    # for i, batch_x in enumerate(latent_full_loader):
    #     data = batch_x[0]

    #     data = data.cuda().float() 
    #     z, mu, logvar = test_vae.encoder(data)
    #     z_pool = m(z).view(8, -1)
    #     z = z.detach().cpu().numpy()

    #     latent_z_flat.append(z.reshape(8, -1))   
        
    #     latent_z_pool.append(z_pool.detach().cpu().numpy())

    #     targets.append(batch_x[1].detach().cpu().numpy())

    # latent_z_flat = np.asanyarray(latent_z_flat).reshape((len(latent_z_flat)*8,4096))
    # latent_z_pool = np.asanyarray(latent_z_pool).reshape((len(latent_z_pool)*8,64))

    # model_tsne = TSNE(n_components=2, n_iter=12000, random_state=False) # show the latent space in 2 dimension
    # #z_states = latent_z.detach().cpu().numpy() # from tensor: gpu to cpu:numpy
    # z_embed_flat = model_tsne.fit_transform(latent_z_flat)
    # z_embed_pool = model_tsne.fit_transform(latent_z_pool)
    # classes = np.asanyarray(targets)

    # fig = plt.figure()
    # plt.scatter(z_embed_flat[:, 0], z_embed_flat[:,1], c=classes, s=10)
    # plt.colorbar()
    # plt.title("Latent space vis. with T-SNE flat") 
    # fig.savefig('latent_vis_T-SNE_flat.png')
    
    # fig = plt.figure()
    # plt.scatter(z_embed_pool[:, 0], z_embed_pool[:,1], c=classes, s=10)
    # plt.colorbar()
    # plt.title("Latent space vis. with T-SNE pool") 
    # fig.savefig('latent_vis_T-SNE_pool.png')


    target_layer = "encoder.down_res.1" # target layer to be visualized # down_res.1

    
    # Visualizations/plots will be saved here : PNG
    save_plot = "Plots/" 
    # images and their corresponding attention maps will be saved here : NIFTI
    result_path =  "GRAD/"
    
    if os.path.isdir(result_path):
        pass
    else:
        os.makedirs(result_path)
    if os.path.isdir(save_plot):
        pass
    else:
        os.makedirs(save_plot)

    gcam = GradCAM(test_vae, target_layer=target_layer, cuda=True)   
    
    img_batch, label = next(iter(latent_full_loader))
    img_example = img_batch[0].unsqueeze(0).to(mlp_args.device)
    mu_avg, logvar_avg = 0, 1
    # Visualize Results and Gradcam images
    n_examples = len(latent_test_dataset)
    visualize_ex = 10
    subplot_shape = [3, visualize_ex]
    fig, axes = plt.subplots(*subplot_shape, figsize=(50,15), facecolor='white', gridspec_kw={'width_ratios': [1] * visualize_ex})
    plt.subplots_adjust(hspace=0.01)
    items = range(0, len(latent_test_dataset)) #*
    #items = [25, 30]
    example = 0
    vis_ex = 0
    #for batch_idx, (x, _) in enumerate(test_loader):
    for imgs, labels in latent_test_loader:
        for i in range(imgs.shape[0]):
            # data = latent_test_dataset[item]
            x, label = imgs[i].to(mlp_args.device).unsqueeze(0), labels[i]              
            filename =  'img_'+str(i) + str(label)
            the_slice = x.shape[-1] // 2 +5
            img = x.detach().cpu().numpy() # use this to save
            #######################################
            ## Save the image #####################################################
            #
            toSave_img = nib.Nifti1Image(img.reshape(
                            [64, 64, 64, 1]),
                            img_example.affine)
            nib.save(toSave_img, result_path + str(i)+'_img.nii.gz')
            print("image saved!")
            #######################################################################


            print(img.shape)
            img = img.reshape(64, 1, 64, 64)
            img_ = img[the_slice][0]
            
            ### ATTENTION GENERATION ###############################################        
            test_vae.eval()
            print("Attention map is being generated...")      
            recon_batch, mu, logvar, out_mlp = gcam.forward(x)
            prediction = out_mlp.detach().cpu().numpy()
            print(f"out_mlp: {prediction}")
            test_vae.zero_grad()
            gcam.backward(mu, logvar, mu_avg, logvar_avg)
            gcam_map = gcam.generate() 
            print("Done!")
            #### DONE!!! ###########################################################

            gcam_map = default_normalizer(gcam_map.detach().cpu().data.numpy())       

            print("Results are saved!")
            # ########################################################################     
            gcam_map_slc = gcam_map.reshape(64, 1, 64, 64)[the_slice][0]       
            cam_and_img, gcam_map_, image_to_vis= superimposed_image_def(gcam_map_slc, img_)


            ## Visualize and save attention maps  ##

            ### Visualize the attention maps and the images ########################
            if vis_ex < visualize_ex and label ==1: ### To visualize MINF cases

                for row, (im, title) in enumerate(zip(
                    [img_, gcam_map.reshape(64, 1, 64, 64)[the_slice][0], cam_and_img],
                    ["IMG_label_" + str(label.item()), "GCAM", "Result"],
                )):
                    #cmap = 'gray' if row == 0  else 'jet'
                    ax = axes[row, vis_ex]
                    if isinstance(im, torch.Tensor):
                        im = im.cpu().detach()
                    if row==0:
                        im_show = ax.imshow(im, cmap="gray")
                    elif row==1 :
                        im_show = ax.imshow(im, cmap="jet")
                    elif row ==2:
                        ax.imshow( gcam_map.reshape(64, 1, 64, 64)[the_slice][0], alpha= 0.5, cmap="jet")
                        ax.imshow( img_,alpha=0.5, cmap="gray")
                    else :
                        im_show = ax.imshow(np.squeeze(im))

                    ax.set_title(title, fontsize=25)
                    ax.axis('off')           
                    fig.colorbar(im_show, fraction=0.046, ax=ax)

                vis_ex += 1
                name_plt = save_plot + "AUMC" + "_extra_seed_16_" + target_layer
                plt.savefig( name_plt +".png", dpi=300, bbox_inches='tight')

        if example == n_examples:
           break


        if example == n_examples:
          break


def main():
    label = 'ICM'
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
        # mlp_parser.add_argument("--base_dir", type=str, default="../emidec_processed_dataset")
    if model_type == 'vae':
        mlp_parser.add_argument('--vae_dir', type=str, default="../EXP_VAE/AUMC_VAE_Multi/vae_optimal_sigma/seed_3_lr_0.0001_e_500_l_64_alpha_1.00_beta_3.00/")
    else:
        mlp_parser.add_argument('--vae_dir', type=str, default= '../FINAL_FOR_LATENTS/AUMC_VAE_Multi/vae_mlp_optimal_sigma_ICM/seed_28_lr_0.0001_e_500_l_64_alpha_500.00_beta_3.00_do_0.00/')
        # mlp_parser.add_argument('--vae_dir', type=str, default= '../home/empquist/VAE_EM/EXP_EXTERNAL/EMIDEC_VAE_Multi/vae_mlp_optimal_sigma_MI/seed_3_lr_0.0001_e_500_l_64_alpha_500.00_beta_3.00_do_0.00/')
    mlp_parser.add_argument("--model_version", type=str, default='VAE_Multi')
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
    mlp_parser.add_argument('--classifier', type=str, default='SVM')
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

    main_latent_calssifier(mlp_args)

if __name__ == "__main__":
    freeze_support()
    main()