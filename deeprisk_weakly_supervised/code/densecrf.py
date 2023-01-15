import denseCRF
import numpy as np
import torch


def densecrf(img, prob, param):
    """
    input parameters:
        img    : a numpy array of shape [C, H, W].
               type of I should be np.uint8, and the values are in [0, 255]
        prob    : a probability map of shape [K, H, W], where K is the number of classes
               type of P should be np.float32
        param: a tuple giving parameters of CRF (w1, alpha, beta, w2, gamma, it), where
                w1    :   weight of bilateral term, e.g. 10.0
                alpha :   spatial distance std, e.g., 80
                beta  :   rgb value std, e.g., 15
                w2    :   weight of spatial term, e.g., 3.0
                gamma :   spatial distance std for spatial term, e.g., 3
                it    :   iteration number, e.g., 5
    output parameters:
        out  : a tensor of shape [H, W], where pixel values represent class indices.
    """
    assert img.shape[1] == img.shape[2] == prob.shape[1] == prob.shape[2]
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if torch.is_tensor(prob):
        prob = prob.detach().cpu().numpy()

    # convert image to 3-channel
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    elif img.shape[0] == 3:
        pass
    else:
        raise NotImplementedError

    # put binary classification probs in shape [C,H,W], C=2
    num_classes = prob.shape[0]
    if num_classes == 1:
        prob = np.array([(1 - prob[0]), prob[0]])

    # swap dimensions (channel and class last)
    img = np.transpose(img, (1, 2, 0))
    prob = np.transpose(prob, (1, 2, 0))
    # inference
    out = denseCRF.densecrf(img, prob, param)
    return torch.from_numpy(out)[None, None, ...]
