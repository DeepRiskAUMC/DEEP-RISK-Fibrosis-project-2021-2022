

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity 


def aumc_create_image_grid(input, reconstruction, rows, cols, img_size):
    input = input.reshape(img_size, 1, img_size, img_size)
    reconstruction = reconstruction.reshape(img_size, 1, img_size, img_size)
    batch_size, channels, height, width = input.shape
    factor = width//16
    grid = np.zeros((height * rows, width * cols * 2))
    for i in range(rows):
        for j in range(cols):
            grid[i*height:(i+1)*height, j*width:(j+1)*width] = input[(i*cols + j)*factor][0]
            grid[i*height:(i+1)*height, (j*width) + (width * cols): ((j+1)*width) + (width * cols)] = reconstruction[(i*cols + j)*factor][0]
    return grid


def emidec_create_image_grid(input, reconstruction, rows, cols, img_size):
    input = np.transpose(input).reshape(img_size, 1, img_size, img_size)
    reconstruction = np.transpose(reconstruction).reshape(img_size, 1, img_size, img_size)
    batch_size, channels, height, width = input.shape
    factor = width//16
    grid = np.zeros((height * rows, width * cols * 2))
    for i in range(rows):
        for j in range(cols):
            grid[i*height:(i+1)*height, j*width:(j+1)*width] = input[(i*cols + j)*factor][0]
            grid[i*height:(i+1)*height, (j*width) + (width * cols): ((j+1)*width) + (width * cols)] = reconstruction[(i*cols + j)*factor][0]
    return grid

def aumc_write_batch(input, reconstruction, step_size, test_or_train:str, writer, epoch, img_size):
    for batch_index in range(0, input.shape[0], step_size):
        if input.shape[1] > 1:
            grid = aumc_create_image_grid(input[batch_index][0], reconstruction[batch_index][0], rows=4, cols=4, img_size=img_size)
            grid_mask = aumc_create_image_grid(input[batch_index][1], reconstruction[batch_index][1], rows=4, cols=4, img_size=img_size)
            writer.add_image(f'mask_batch_{batch_index}_{test_or_train}', grid_mask, dataformats='HW', global_step=epoch)
        else:
            grid = aumc_create_image_grid(input[batch_index], reconstruction[batch_index], rows=4, cols=4, img_size=img_size)
        # Add the image grid to TensorBoard
        writer.add_image(f'batch_{batch_index}_{test_or_train}', grid, dataformats='HW', global_step=epoch)

def emidec_write_batch(input, reconstruction, step_size, test_or_train:str, writer, epoch, img_size):
    for batch_index in range(0, input.shape[0], step_size):
        grid = emidec_create_image_grid(input[batch_index], reconstruction[batch_index], rows=4, cols=4, img_size=img_size)
        # Add the image grid to TensorBoard
        writer.add_image(f'batch_{batch_index}_{test_or_train}', grid, dataformats='HW', global_step=epoch)

def calculate_mutual_information(hgram):
     # Mutual information for joint histogram
     # Convert bins counts to probability values
     pxy = hgram / float(np.sum(hgram))
     px = np.sum(pxy, axis=1) # marginal for x over y
     py = np.sum(pxy, axis=0) # marginal for y over x
     px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
     # Now we can do the calculation using the pxy, px_py 2D arrays
     nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
     return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def compute_mmd(samples_A, samples_B, gamma=1.0):
    K_aa = rbf_kernel(samples_A, samples_A, gamma=gamma)
    K_bb = rbf_kernel(samples_B, samples_B, gamma=gamma)
    K_ab = rbf_kernel(samples_A, samples_B, gamma=gamma)

    mmd = np.sqrt(np.mean(K_aa) + np.mean(K_bb) - 2 * np.mean(K_ab))
    return mmd

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.T), torch.mm(y, y.T), torch.mm(x, y.T)
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(xx.shape),
                  torch.zeros(xx.shape))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)



    return torch.mean(XX + YY - 2. * XY)

def get_all_recon_metrics(input, recons, img_size):
    total_mi = 0
    total_mmd = 0
    total_ssim = 0
    total_mse = 0

    for batch_index in range(input.shape[0]):
        input_3d, recon_3d = input[batch_index][0].reshape(img_size), recons[batch_index][0].reshape(img_size)
        # # Reshape images and calculate MMD
        
        flat_input, flat_outputs = input_3d.detach().cpu().numpy().ravel(), recon_3d.detach().cpu().numpy().ravel()
        hist_2d, x_edges, y_edges = np.histogram2d(flat_input, flat_outputs, bins=20)
        total_mi += calculate_mutual_information(hist_2d)
        total_ssim += structural_similarity(flat_input, flat_outputs, data_range=flat_input.max() - flat_input.min())
        total_mse += F.mse_loss(recon_3d, input_3d)
        
        input_samples = input_3d.reshape(input_3d.shape[0], -1).detach().cpu() #.numpy()
        output_samples = recon_3d.reshape(recon_3d.shape[0], -1).detach().cpu() #.numpy()
        total_mmd += MMD(input_samples, output_samples, kernel='rbf').item()

    mean_mmd = total_mmd/input.shape[0]
    mean_mi = total_mi/input.shape[0]
    mean_ssim = total_ssim/input.shape[0]
    mean_mse = total_mse/input.shape[0]

    return mean_mmd, mean_mi, mean_ssim, mean_mse
