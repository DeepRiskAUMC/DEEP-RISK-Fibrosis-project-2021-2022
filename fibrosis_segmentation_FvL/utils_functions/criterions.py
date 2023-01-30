from turtle import forward
import torch
import math
import numpy as np
import torch.nn.functional as f
import skimage.metrics as skmetrics
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from utils_functions.average_hausdorff import avg_hausdorff_distance, reg_hausdorff_distance

class Diceloss(torch.nn.Module):
    def init(self):
        super(Diceloss, self).init()
    def forward(self,pred, target):
        if torch.max(target).item() not in [0.0, 1.0] or torch.min(target).item() not in [0.0, 1.0]:
            raise ValueError(f'GT image must be binary but got minimum value of {torch.min(target).item()} and maximum of {torch.max(target).item()}')
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

class WeightedDiceLoss(torch.nn.Module):
    def init(self):
        super(WeightedDiceLoss, self).init()
    def forward(self,pred, myo_target, fib_target):
        if torch.max(myo_target).item() not in [0.0, 1.0] or torch.min(myo_target).item() not in [0.0, 1.0] or torch.max(fib_target).item() not in [0.0, 1.0] or torch.min(fib_target).item() not in [0.0, 1.0]:
            raise ValueError(f'GT image must be binary but got minimum value of {torch.min(myo_target).item()}/{torch.min(fib_target).item()} and maximum of {torch.max(myo_target).item()}/{torch.max(fib_target).item()}')
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = myo_target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        dice_score_myo = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
        tflat = fib_target.contiguous().view(-1)
        iflat = iflat[tflat==1.0]
        tflat = tflat[tflat==1.0]
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        dice_score_fib = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
        return 1 - (0.5 * dice_score_myo + 0.5 * dice_score_fib)

class AdaptiveWeightedDiceLoss(torch.nn.Module):
    def init(self):
        super(AdaptiveWeightedDiceLoss, self).init()
    def forward(self,pred, myo_target, fib_target, alpha):
        if torch.max(myo_target).item() not in [0.0, 1.0] or torch.min(myo_target).item() not in [0.0, 1.0] or torch.max(fib_target).item() not in [0.0, 1.0] or torch.min(fib_target).item() not in [0.0, 1.0]:
            raise ValueError(f'GT image must be binary but got minimum value of {torch.min(myo_target).item()}/{torch.min(fib_target).item()} and maximum of {torch.max(myo_target).item()}/{torch.max(fib_target).item()}')
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = myo_target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        dice_score_myo = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
        tflat = fib_target.contiguous().view(-1)
        iflat = iflat[tflat==1.0]
        tflat = tflat[tflat==1.0]
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        dice_score_fib = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
        return 1 - ((1-alpha) * dice_score_myo + alpha * dice_score_fib)

class Dice_WBCE_loss(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        # self.WCE_loss = torch.nn.CrossEntropyLoss(weight=weights)
        self.WCE_loss = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        self.dice_loss = Diceloss()
    def forward(self, pred, target):
        return self.WCE_loss(pred, target) + self.dice_loss(pred, target)

class L1loss(torch.nn.Module):
    def init(self):
        super(L1loss, self).init()
    def forward(self, pred, target, device=None):
        loss = f.l1_loss(pred, target, reduction='none')
        return loss.sum()

class MSEloss(torch.nn.Module):
    def init(self):
        super(MSEloss, self).init()
    def forward(self, pred, target, device=None):
        # print('pred1 type', pred.type())
        # print('target1 type', target.type())
        loss = f.mse_loss(pred, target, reduction='none')
        return loss.sum()

class WeightedMSEloss(torch.nn.Module):
    def init(self):
        super(WeightedMSEloss, self).init()
    def forward(self, pred, target, device=None):
        weights = torch.Tensor([[1.0,1.0,1.0,1.0]])
        weights = weights.type_as(pred)
        ymin_pred, ymax_pred, xmin_pred, xmax_pred = pred.squeeze()
        ymin_real, ymax_real, xmin_real, xmax_real = target.squeeze()
        if ymin_real < ymin_pred:
            weights[:,0] = 2.0
        if ymax_pred < ymax_real:
            weights[:,1] = 2.0
        if xmin_real < xmin_pred:
            weights[:,2] = 2.0
        if xmax_pred < xmax_real:
            weights[:,3] = 2.0
        unweighted_loss = f.mse_loss(pred, target, reduction='none')
        weighted_loss = torch.mul(unweighted_loss, weights)
        return weighted_loss.sum()
    
class IoUloss(torch.nn.Module):
    def __init__(self, loss=True, generalized=False):
        super().__init__()
        self.generalized = generalized
        self.loss = loss

    def forward(self, pred, target, device=None):
        pred = pred.squeeze()
        target = target.squeeze()

        #make sure x1 < x2 and y1 < y2
        p_x1 = min(pred[2], pred[3])
        p_x2 = max(pred[2], pred[3])
        p_y1 = min(pred[0], pred[1])
        p_y2 = max(pred[0], pred[1])

        g_y1, g_y2, g_x1, g_x2 = target

        #calculate area of bounding boxes
        area_g = (g_x2 - g_x1) * (g_y2 - g_y1)
        area_p = (p_x2 - p_x1) * (p_y2 - p_y1)

        #calculate intersection
        i_x1 = max(p_x1, g_x1)
        i_x2 = min(p_x2, g_x2)
        i_y1 = max(p_y1, g_y1)
        i_y2 = min(p_y2, g_y2)

        if i_x2 > i_x1 and i_y2 > i_y1:
            area_i = (i_x2 - i_x1) * (i_y2 - i_y1)
        else:
            area_i = 0
        
        # Finding the coordinates of smallest enclosing box c
        c_x1 = min(p_x1, g_x1)
        c_x2 = max(p_x2, g_x2)
        c_y1 = min(p_y1, g_y1)
        c_y2 = max(p_y2, g_y2)

        area_c = (c_x2 - c_x1) * (c_y2 - c_y1)

        union_area = area_p + area_g - area_i

        IoU = area_i / union_area
        GIoU = IoU - (area_c - union_area)/area_c

        if self.generalized:
            if self.loss:
                return 1 - GIoU
            else:
                return GIoU
        else:
            if self.loss:
                return 1-IoU
            else:
                return IoU

def dice_coef(img, img2):
        if not ((img==0) | (img==1)).all() or not ((img2==0) | (img2==1)).all():
            raise ValueError("Images need to be binary")
        if img.shape != img2.shape:
            raise ValueError("Shape mismatch: img and img2 have to be of the same shape.")
        else:
            
            lenIntersection=0
            
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if ( np.array_equal(img[i][j],img2[i][j]) ):
                        lenIntersection+=1
             
            lenimg=img.shape[0]*img.shape[1]
            lenimg2=img2.shape[0]*img2.shape[1]  
            value = (2. * lenIntersection  / (lenimg + lenimg2))
        return value

def smoothed_dice_score(img, img2):
    img, img2 = img.squeeze(), img2.squeeze()
    if not ((img==0) | (img==1)).all() or not ((img2==0) | (img2==1)).all():
            raise ValueError("Images need to be binary")
    if img.shape != img2.shape:
        raise ValueError("Shape mismatch: img and img2 have to be of the same shape.")
    else:
        dice_loss = Diceloss()(img, img2)
        return 1 - dice_loss

def dice_score(pred, target):
    pred, target = pred.squeeze(), target.squeeze()
    if not ((pred==0) | (pred==1)).all() or not ((target==0) | (target==1)).all():
            raise ValueError("Images need to be binary")
    if pred.shape != target.shape:
        raise ValueError("Shape mismatch: img and img2 have to be of the same shape.")
    else:
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        if A_sum + B_sum == 0:
            return 1.0
        else:
            return (2. * intersection) / (A_sum + B_sum)

def neg_dice_score(pred, target):
    pred, target = pred.squeeze(), target.squeeze()
    if not ((pred==0) | (pred==1)).all() or not ((target==0) | (target==1)).all():
            raise ValueError("Images need to be binary")
    if pred.shape != target.shape:
        raise ValueError("Shape mismatch: img and img2 have to be of the same shape.")
    else:
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        if B_sum == 0:
            return -1.0
        else:
            return (2. * intersection) / (A_sum + B_sum)

def dice_score_2d(img, img2):
    img, img2 = img.squeeze(), img2.squeeze()
    if img.dim() < 3 or img2.dim() < 3:
        raise ValueError("images are already 2D. use the 'dice_score' function instead for 2d images")
    if not ((img==0) | (img==1)).all() or not ((img2==0) | (img2==1)).all():
            raise ValueError("Images need to be binary")
    if img.shape != img2.shape:
        raise ValueError("Shape mismatch: img and img2 have to be of the same shape.")
    else:
        dice_scores = []
        for i in range(img.shape[0]):
            dice_loss = Diceloss()(img[i], img2[i])
            dice_scores.append(1-dice_loss)
        return dice_scores

def dice_score_3(prediction, gt):
    if not ((prediction==0) | (prediction==1)).all() or not ((gt==0) | (gt==1)).all():
            raise ValueError("Images need to be binary")
    if prediction.shape != gt.shape:
        raise ValueError("Shape mismatch: img and img2 have to be of the same shape.")
    else:
        prediction, gt = prediction.cpu().detach().numpy(), gt.cpu().detach().numpy()
        img1 = gt.flatten()
        img2 = prediction.flatten()
        conf_matrix = confusion_matrix(img1, img2).ravel()
        if len(conf_matrix) == 1:
            return 0
        else:
            try:
                tn, fp, fn, tp = conf_matrix
            except:
                raise ValueError('confusion_matrix:', confusion_matrix(img1, img2).ravel())
        smooth = 1
        dice = (2 * tp + smooth)/(2 * tp + fp + fn + smooth)
        if math.isnan(dice):
            return 0
        return dice
        
def get_accuracy(predictions, labels):
    # noinspection PyUnresolvedReferences
    return (labels == predictions).astype(float).mean()

def hausdorff_distance(img1, img2):
    if not ((img1==0) | (img1==1)).all() or not ((img2==0) | (img2==1)).all():
            raise ValueError("Images need to be binary")
    if img1.shape != img2.shape:
        raise ValueError("Shape mismatch: img and img2 have to be of the same shape.")
    else:
        img1, img2 = np.array(img1, dtype=bool), np.array(img2, dtype=bool)
        # print(img1.squeeze().shape, img2.squeeze().shape)
        # print(img1.dtype, img2.dtype)
        hausdorff = reg_hausdorff_distance(img1.squeeze(), img2.squeeze())
        return hausdorff

def average_hausdorff_distance(img1, img2):
    img1, img2 = np.array(img1, dtype=bool), np.array(img2, dtype=bool)
    avg_hausdorff_dist = avg_hausdorff_distance(img1.squeeze(), img2.squeeze())
    # print(avg_hausdorff_dist)
    return avg_hausdorff_dist

def get_TPR(prediction, gt, info=False):
    img1 = gt.flatten()
    img2 = prediction.flatten()
    conf_matrix = confusion_matrix(img1, img2).ravel()
    if len(conf_matrix) == 1:
        return 0
    else:
        try:
            tn, fp, fn, tp = conf_matrix
        except:
            raise ValueError('confusion_matrix:', confusion_matrix(img1, img2).ravel())
        if info:
            print(f'true positive: {tp}, true negative: {tn}. false positive: {fp}. false negative: {fn}')
            print(img1)
            gt_ones = np.sum(img1)
            own_tp = np.sum(img1 * img2)
            own_tn = np.bincount(img1 + img2)[0]
            own_fp = np.bincount(img2)[1] - own_tp
            own_fn = np.bincount(img2)[0] - own_tn
            print(f'true positive: {own_tp}, true negative: {own_tn}. false positive: {own_fp}. false negative: {own_fn}. ones in GT: {gt_ones}')
    TPR = tp/(tp+fn)
    if math.isnan(TPR):
        return 0
    return TPR

def get_TNR(prediction, gt):
    img1 = gt.flatten()
    img2 = prediction.flatten()
    conf_matrix = confusion_matrix(img1, img2).ravel()
    if len(conf_matrix) == 1:
        return 1
    else:
        try:
            tn, fp, fn, tp = conf_matrix
        except:
            raise ValueError('confusion_matrix:', confusion_matrix(img1, img2).ravel())
    return tn/(tn+fp)

def get_PPV(prediction, target):
    img1 = target.flatten()
    img2 = prediction.flatten()
    conf_matrix = confusion_matrix(img1, img2).ravel()
    if len(conf_matrix) == 1:
        return 1
    else:
        try:
            tn, fp, fn, tp = conf_matrix
        except:
            raise ValueError('confusion_matrix:', confusion_matrix(img1, img2).ravel())
    return tp/(tp+fp)

def get_NPV(prediction, target):
    img1 = target.flatten()
    img2 = prediction.flatten()
    conf_matrix = confusion_matrix(img1, img2).ravel()
    if len(conf_matrix) == 1:
        return 1
    else:
        try:
            tn, fp, fn, tp = conf_matrix
        except:
            raise ValueError('confusion_matrix:', confusion_matrix(img1, img2).ravel())
    return tn/(tn+fn)

def get_ROCAUC(prediction, gt):
    roc_score = roc_auc_score(gt, prediction)
    return roc_score

def evaluate_regression(y_true, y_pred):
    """
    Function to evaluate predictions
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: dictionary containing results
    """
    results = {}
    for metric in [mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score]:
        results[metric.__name__] = f'{metric(y_true, y_pred):.4f}'
    return results

class AUC_like_loss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, y_true, y_pred):
        device = y_true.device
        y_true, y_pred = y_true.reshape(-1, 1), y_pred.reshape(-1,1)
        preds_matrix = torch.sigmoid(y_pred @ y_pred.T)
        true_matrix = y_true @ torch.ones(y_true.shape, device=device).T
        true_matrix2 = torch.ones(y_true.shape, device=device) @ y_true.T
        subtraction = true_matrix - true_matrix2
        max_value = torch.maximum(subtraction, torch.tensor([0], device=device))
        reduce_mean = torch.mean(preds_matrix * max_value)
        cost = - reduce_mean
        # cost = - torch.reduce_mean(torch.sigmoid(y_pred @ y_pred.T) * np.maximum(y_true @ np.ones(y_true.shape).T - np.ones(y_true.shape) @ y_true.T, 0))
        return cost

def epoch_update_gamma(y_true, y_pred, epoch=-1, delta=2):
        """
        Calculate gamma from last epoch's targets and predictions.
        Gamma is updated at the end of each epoch.
        y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        """
        sub_sample_size = 2000.0
        pos = y_pred[y_true==1]
        neg = y_pred[y_true==0] # yo pytorch, no boolean tensors or operators?  Wassap?
        # subsample the training set for performance
        cap_pos = pos.shape[0]
        cap_neg = neg.shape[0]
        pos = pos[torch.rand_like(pos) < sub_sample_size/cap_pos]
        neg = neg[torch.rand_like(neg) < sub_sample_size/cap_neg]
        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]
        pos_expand = pos.view(-1,1).expand(-1,ln_neg).reshape(-1)
        neg_expand = neg.repeat(ln_pos)
        diff = neg_expand - pos_expand
        Lp = diff[diff>0] # because we're taking positive diffs, we got pos and neg flipped.
        ln_Lp = Lp.shape[0]-1
        diff_neg = -1.0 * diff[diff<0]
        diff_neg = diff_neg.sort()[0]
        ln_neg = diff_neg.shape[0]-1
        ln_neg = max([ln_neg, 0])
        left_wing = int(ln_Lp*delta)
        left_wing = max([0,left_wing])
        left_wing = min([ln_neg,left_wing])
        default_gamma = torch.tensor(0.2, dtype=torch.float).cuda()
        if diff_neg.shape[0] > 0 :
            gamma = diff_neg[left_wing]
        else:
            gamma = default_gamma # default=torch.tensor(0.2, dtype=torch.float).cuda() #zoink
        L1 = diff[diff>-1.0*gamma]
        if epoch > -1 :
            return gamma
        else :
            return default_gamma


def roc_star_loss(_y_true, y_pred, gamma, _epoch_true, epoch_pred):
        """
        Nearly direct loss function for AUC.
        See article,
        C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
        https://github.com/iridiumblue/articles/blob/master/roc_star.md
            _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
            y_pred: `Tensor` . Predictions.
            gamma  : `Float` Gamma, as derived from last epoch.
            _epoch_true: `Tensor`.  Targets (labels) from last epoch.
            epoch_pred : `Tensor`.  Predicions from last epoch.
        """
        #convert labels to boolean
        y_true = (_y_true>=0.50)
        epoch_true = (_epoch_true>=0.50)

        # if batch is either all true or false return small random stub value.
        if torch.sum(y_true)==0 or torch.sum(y_true) == y_true.shape[0]: return torch.sum(y_pred)*1e-8

        pos = y_pred[y_true]
        neg = y_pred[~y_true]

        epoch_pos = epoch_pred[epoch_true]
        epoch_neg = epoch_pred[~epoch_true]

        # Take random subsamples of the training set, both positive and negative.
        max_pos = 1000 # Max number of positive training samples
        max_neg = 1000 # Max number of positive training samples
        cap_pos = epoch_pos.shape[0]
        epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos/cap_pos]
        epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg/cap_pos]

        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]

        # sum positive batch elements agaionst (subsampled) negative elements
        if ln_pos>0 :
            pos_expand = pos.view(-1,1).expand(-1,epoch_neg.shape[0]).reshape(-1)
            neg_expand = epoch_neg.repeat(ln_pos)

            diff2 = neg_expand - pos_expand + gamma
            l2 = diff2[diff2>0]
            m2 = l2 * l2
        else:
            m2 = torch.tensor([0], dtype=torch.float).cuda()

        # Similarly, compare negative batch elements against (subsampled) positive elements
        if ln_neg>0 :
            pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
            neg_expand = neg.repeat(epoch_pos.shape[0])

            diff3 = neg_expand - pos_expand + gamma
            l3 = diff3[diff3>0]
            m3 = l3*l3
        else:
            m3 = torch.tensor([0], dtype=torch.float).cuda()

        if (torch.sum(m2)+torch.sum(m3))!=0 :
            res2 = torch.sum(m2)/max_pos+torch.sum(m3)/max_neg
        else:
            res2 = torch.sum(m2)+torch.sum(m3)

        res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

        return res2

if __name__ == '__main__':
    points_a = (3, 0)
    points_b = (6, 0)
    shape = (7, 4)
    image_a = np.zeros(shape, dtype=bool)
    image_b = np.zeros(shape, dtype=bool)
    image_a[points_a] = True
    image_a[4,0] = True
    image_a[3,2] = True
    image_b[points_b] = True
    image_b[3,2] = True
    TPR = get_TPR(image_a, image_b)
    TNR = get_TNR(image_a, image_b)
    print(TPR, TNR)