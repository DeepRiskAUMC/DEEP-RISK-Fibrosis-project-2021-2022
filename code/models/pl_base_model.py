import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import matplotlib.pyplot as plt
import io
import numpy as np
import itertools
import torchvision
import matplotlib.cm as cm
import math
from torchvision.transforms import transforms
import cv2

class MyDiceOld(torchmetrics.Metric):
    def __init__(self, threshold=0.5, dist_sync_on_step=False, smoothing=0):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold
        self.smoothing = smoothing
        self.add_state('sum_dice_scores', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, f"shapes {preds.shape} and {target.shape} don't match"
        preds = preds > self.threshold
        target = target > 0.0
        # calculate dice separately for each image in batch
        intersection = (preds * target).sum(dim=(1, 2, 3))
        dice = (2 * intersection +  self.smoothing) / (preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smoothing)

        self.sum_dice_scores += dice.sum()
        self.total += dice.numel()

    def compute(self):
        return self.sum_dice_scores / self.total


class MyDice(torchmetrics.Metric):
    def __init__(self, threshold=0.5, dist_sync_on_step=False, smoothing=0, name='Dice_2D'):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = name
        self.threshold = threshold
        self.smoothing = smoothing
        self.add_state('sum_dice_scores', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, f"shapes {preds.shape} and {target.shape} don't match"
        # remove padded slices
        B, C, H, W = preds.shape
        preds = preds.view(B, -1)
        target = target.view(B, -1)

        # remove slices/mris without valid target label
        labeled =  (target.amin(dim=(1), keepdim=False) > -0.01)
        if labeled.sum() == 0:
            return
        preds = preds[labeled]
        target = target[labeled]

        preds = preds > self.threshold
        target = target > 0.0

        # without smoothing, dice is undefined when both pred and target empty
        if self.smoothing == 0:
            non_empty = (preds.amax(dim=(1), keepdim=False) != 0) * (target.amax(dim=(1), keepdim=False) != 0)
            if non_empty.sum() == 0:
                return
            preds = preds[non_empty]
            target = target[non_empty]

        # calculate dice separately for each image in batch
        intersection = (preds * target).sum(dim=(1))
        dice = (2 * intersection +  self.smoothing) / (preds.sum(dim=(1)) + target.sum(dim=(1)) + self.smoothing)

        self.sum_dice_scores += dice.sum()
        self.total += dice.numel()

    def compute(self):
        return self.sum_dice_scores / self.total



def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class ModelBase(pl.LightningModule):
    """Model superclass that handles:
            - loss
            - optimizers
            - schedulers
            - train, validation and test steps
            - logging
            - forward
            - cam processing
        Such that subclass models only need:
            - forward_features (i.e. all layers except final global pooling)

        Args:
            lr (float): learning rate. Default: 1-e3.
            weight_decay (float): AdamW weight decay. Default: 0., i.e. normal Adam.
            label_smoothing (float): Amount of label smoothing to use. Default: 0.
            schedule (str): Learning rate scheduler to use. Default None. Options: None, step, cosine
            max_iters (int): Maximal number of iterations, which is necessary for cosine lr scheduling. Default: -1 / needs to be set for cosine schedule.
            warmup_iters (int): Number of iterations to use for linear learning rate warmup. Default: 0 / No warmup.
            pooling (str): Type of global pooling to use as final layer. Default: avg. Options: avg, max, ngwp
            feature_dropout (float): dropout probability before final 1x1 convolution. Default: 0.0.
            cam_dropout (float): dropout probability before final pooling. Default: 0.0.

    """
    def __init__(self, lr=1e-3, weight_decay=0, label_smoothing=0,
                schedule=None, max_iters=-1, warmup_iters=0, num_classes=1,
                heavy_log=10, pooling="avg", feature_dropout=0.0,
                cam_dropout=0.0, final_feature_size=None,
                myo_mask_pooling=False, myo_mask_threshold=0.3, downsampling=8,
                myo_mask_dilation=0, cheat_dice=False,
                myo_mask_prob=1, myo_input=False):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.schedule = schedule
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.num_classes = num_classes
        self.save_hyperparameters()
        self.heavy_log = heavy_log
        self.pooling = pooling
        self.myo_mask_pooling = myo_mask_pooling
        self.myo_mask_threshold = myo_mask_threshold
        self.downsampling = downsampling
        self.cheat_dice = cheat_dice
        self.myo_mask_prob = myo_mask_prob
        self.myo_input = myo_input

        if self.myo_mask_pooling == True:
            self.mask_cam = True
        else:
            self.mask_cam = False

        self.myo_mask_dilation = myo_mask_dilation

        # dropout
        self.feature_dropout_layer = nn.Dropout2d(p=feature_dropout)
        self.cam_dropout_layer = nn.Dropout(p=cam_dropout)
        # 1x1 class convolution
        self.classifier = nn.Conv2d(final_feature_size, self.num_classes, 1, bias=True)

        # global pooling
        assert pooling in ["avg", "max", "ngwp"]
        if self.myo_mask_pooling == False:
            if pooling == "avg":
                self.final_pool =  nn.AdaptiveAvgPool2d((1, 1))
            elif pooling == "max":
                self.final_pool = nn.AdaptiveMaxPool2d((1, 1))
            elif pooling == "ngwp":
                self.final_pool = NormalizedGlobalWeightedPooling()
        else:
            if pooling == "avg":
                self.final_pool = MaskedAdaptiveAvgPool2d((1, 1))
            elif pooling == "max":
                self.final_pool = MaskedAdaptiveMaxPool2d((1, 1))


        # cam upsampling params
        self.up_bilinear = nn.UpsamplingBilinear2d(scale_factor=self.downsampling)
        up = nn.ConvTranspose2d(num_classes, num_classes, self.downsampling*2, stride=self.downsampling, padding=int(self.downsampling/2),
                                output_padding=0, groups=num_classes,
                                bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        self.up_conv_transpose = up


        # metrics
        def _make_metric_dict(metric, splits=["Train", "Val", "Test"], **kwargs):
            return nn.ModuleDict({split : metric(**kwargs) for split in splits })

        self.accuracy = _make_metric_dict(torchmetrics.Accuracy, threshold=0.5)
        self.confusion = _make_metric_dict(torchmetrics.ConfusionMatrix, num_classes=max(2, self.num_classes), threshold=0.5)
        self.auroc = _make_metric_dict(torchmetrics.AUROC)
        self.pr_curve = _make_metric_dict(torchmetrics.PrecisionRecallCurve)
        self.roc = _make_metric_dict(torchmetrics.ROC)
        self.auc = _make_metric_dict(torchmetrics.AUC)

        self.dice_t1_2D = _make_metric_dict(MyDice, threshold=0.1, smoothing=1, name="Dice_t.1_2D")
        self.dice_t3_2D = _make_metric_dict(MyDice, threshold=0.3, smoothing=1, name="Dice_t.3_2D")
        self.dice_t5_2D = _make_metric_dict(MyDice, threshold=0.5, smoothing=1, name="Dice_t.5_2D")
        self.dice_t1_2D_unsmoothed = _make_metric_dict(MyDice, threshold=0.1, smoothing=0, name="Dice_t.1_2D_unsmoothed")
        self.dice_t3_2D_unsmoothed = _make_metric_dict(MyDice, threshold=0.3, smoothing=0, name="Dice_t.3_2D_unsmoothed")
        self.dice_t5_2D_unsmoothed = _make_metric_dict(MyDice, threshold=0.5, smoothing=0, name="Dice_t.5_2D_unsmoothed")

        self.dices = [self.dice_t1_2D, self.dice_t3_2D, self.dice_t5_2D,
                     self.dice_t1_2D_unsmoothed, self.dice_t3_2D_unsmoothed, self.dice_t5_2D_unsmoothed]

        #self.train_accuracy = torchmetrics.Accuracy(threshold=0.5)
        #self.validation_accuracy = torchmetrics.Accuracy(threshold=0.5)
        #self.test_accuracy = torchmetrics.Accuracy(threshold=0.5)

        #confusion_matrix_size = 2 if num_classes == 1 else num_classes
        #self.train_confusion = torchmetrics.ConfusionMatrix(confusion_matrix_size, threshold=0.5)
        #self.validation_confusion = torchmetrics.ConfusionMatrix(confusion_matrix_size, threshold=0.5)
        #self.test_confusion = torchmetrics.ConfusionMatrix(confusion_matrix_size, threshold=0.5)

        #curve_num_classes = None if num_classes == 1 else num_classes
        #self.train_auroc = torchmetrics.AUROC(curve_num_classes)
        #self.validation_auroc = torchmetrics.AUROC(curve_num_classes)
        #self.test_auroc = torchmetrics.AUROC(curve_num_classes)

        #self.train_pr_curve = torchmetrics.PrecisionRecallCurve(curve_num_classes)
        #self.validation_pr_curve = torchmetrics.PrecisionRecallCurve(curve_num_classes)
        #self.test_pr_curve = torchmetrics.PrecisionRecallCurve(curve_num_classes)

        #self.train_roc = torchmetrics.ROC(curve_num_classes)
        #self.validation_roc = torchmetrics.ROC(curve_num_classes)
        #self.test_roc = torchmetrics.ROC(curve_num_classes)

        #self.train_auc = torchmetrics.AUC()
        #self.validation_auc = torchmetrics.AUC()
        #self.test_auc = torchmetrics.AUC()

        #self.train_dice = MyDice(threshold=0.5)
        #self.validation_dice = MyDice(threshold=0.5)
        #self.train_dice2 = MyDice(threshold=0.8)
        #self.validation_dice2 = MyDice(threshold=0.8)
        #self.train_dice3 = MyDice(threshold=0.2)
        #self.validation_dice3 = MyDice(threshold=0.2)

        # var for image + cam plotting
        self.example_batch = {"Train" : None, "Val" : None , "Test": None}
        #self.train_example_batch = None
        #self.val_example_batch = None

        self.output2pred = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=-1)


    def forward_features(self, x):
        raise NotImplementedError("Please implement forward_features() in subclass.")


    def forward(self, x, myo_seg=None):
        if myo_seg != None and self.myo_input == True:
            x = torch.cat((x, myo_seg), dim=1)
        # forward features (class activation map)
        x = self.forward_features(x)
        x = self.feature_dropout_layer(x)
        x = self.classifier(x)
        x = self.cam_dropout_layer(x)
        if myo_seg == None or self.myo_mask_pooling == False:
            x = self.final_pool(x)
        else:
            myo_mask = self.myo_seg_to_mask(myo_seg, dilation=self.myo_mask_dilation)
            x = self.final_pool(x, myo_mask)
        x = torch.flatten(x, start_dim=1)
        return x

    def myo_seg_to_mask(self, x, dilation=0, size="cam"):
        assert size in ["cam", "image"]
        if dilation != 0:
            x = F.max_pool2d(x, kernel_size=dilation, stride=1, padding=dilation//2)

        if size == "cam":
            x = F.interpolate(x, scale_factor=1/self.downsampling,mode='area')
        elif size == "image":
            pass
        else:
            raise NotImplementedError

        x = (x > self.myo_mask_threshold).float()

        # mask only in training mode, or when in eval mode and model is trained using mask_prob 1
        myo_mask_prob = self.myo_mask_prob if (self.training or self.myo_mask_prob == 1) else 0
        no_mask = torch.rand(x.shape[0]) >  myo_mask_prob
        #print(f"{no_mask=}")
        x[no_mask, ...] = 1.0

        return x


    def make_cam(self, x, select_values="pos_only", upsampling="no", myo_seg=None, fullsize_mask=False, cheat_gt=None):
        if cheat_gt != None:
            raise NotImplementedError
        if select_values not in ["pos_only", "all", "sigmoid", "raw"]:
            raise NotImplementedError
        if upsampling not  in ["no", "bilinear", "conv_transpose"]:
            raise NotImplementedError

        if myo_seg != None and self.myo_input == True:
            x = torch.cat((x, myo_seg), dim=1)
        x = self.forward_features(x)
        x = self.classifier(x)


        if myo_seg != None and self.myo_mask_pooling == True:
            myo_mask = self.myo_seg_to_mask(myo_seg, dilation=self.myo_mask_dilation)
            assert myo_mask.shape == x.shape, f"shapes {myo_mask.shape} and {x.shape} don't match"
            x = x * myo_mask

        eps = 1e-5
        if select_values == "pos_only":
            # remove negative and scale each class in [0,1]
            x = F.relu(x)
            x = x / (torch.amax(x, (2, 3), keepdims=True) + eps)
        elif select_values == "all":
            # scale all values in [0, 1]
            x = x - torch.amin(x, (2, 3), keepdims=True)
            x  = x / (torch.amax(x, (2, 3), keepdims=True) + eps)
        elif select_values == "sigmoid":
            x = torch.sigmoid(x)
        elif select_values == "raw":
            pass

        if upsampling == "bilinear":
            x = self.up_bilinear(x)
        elif upsampling == "conv_transpose":
            x = self.up_conv_transpose(x)
        elif upsampling == "no":
            pass

        if fullsize_mask == True:
            myo_mask = self.myo_seg_to_mask(myo_seg, dilation=self.myo_mask_dilation, size='image')
            x = x * myo_mask

        return x


    def step_dice(self, torchmetric_dice, x, fibrosis_seg_label, myo_seg, cheat=False):
        labeled =  fibrosis_seg_label.amin(dim=(1, 2, 3), keepdim=False) != -1 * (fibrosis_seg_label.sum(dim=(1, 2, 3), keepdim=False) != 0)
        if labeled.sum() == 0:
            return
        else:
            x = x[labeled]
            fibrosis_seg_label = fibrosis_seg_label[labeled]
            myo_seg = myo_seg[labeled]

            if cheat == True:
                if self.myo_mask_pooling == True:
                    myo_mask = self.myo_seg_to_mask(myo_seg, dilation=self.myo_mask_dilation, size="image")
                    cam = fibrosis_seg_label * myo_mask
                else:
                    cam = fibrosis_seg_label
            else:
                cam = self.make_cam(x, select_values="pos_only", upsampling="conv_transpose", myo_seg=myo_seg)
            torchmetric_dice(cam, fibrosis_seg_label)

        return




    def configure_optimizers(self):
        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.warmup_iters == 0 and self.schedule == None:
            return optimizer

        schedulers_list = []
        if self.warmup_iters != 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=self.warmup_iters)
            schedulers_list.append(warmup_scheduler)

        if self.schedule == 'step':
            schedulers_list.append(torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1))
        elif self.schedule == 'cosine':
            schedulers_list.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_iters - self.warmup_iters))

        if self.warmup_iters != 0 and self.schedule != None:
            milestones = [self.warmup_iters]
        else:
            milestones = []

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers_list, milestones=milestones)
        scheduler.optimizer = optimizer # dirty fix for bug in SequentialLR/ Lightning interaction

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }



    def cross_entropy_loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels, label_smoothing=self.label_smoothing)



    def bce_with_logits_loss(self, outputs, labels):
        # remove extra dim
        outputs = outputs[:,0]
        # bce requires float labels
        labels = labels.float()
        if self.label_smoothing != 0:
            with torch.no_grad():
                labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(outputs, labels)

    def soft_margin_loss(self, outputs, labels):
        labels[labels == 0] = -1
        return F.multilabel_soft_margin_loss(outputs, labels)

    def _step(self, batch, split, logging=True):
        assert split in ["Train", "Val", "Test"]

        image = batch['img']
        labels = batch['label']
        myo_seg = batch['myo_seg']
        fibrosis_seg_label = batch['fibrosis_seg_label']
        outputs = self.forward(image, myo_seg)

        if self.num_classes == 1:
            loss = self.bce_with_logits_loss(outputs, labels)
        else:
            loss = self.cross_entropy_loss(outputs, labels)

        batch_dict= {
            "loss" : loss
        }

        if logging == True:
            preds = self.output2pred(outputs).squeeze(dim=1)

            self.accuracy[split](preds, labels)

            if (self.current_epoch + 1) % self.heavy_log == 0:
                # ADD HEAVY LOGGING
                self.auroc[split](preds, labels)
                self.confusion[split](preds, labels)
                self.pr_curve[split](preds, labels)
                self.roc[split](preds, labels)

                # log dice
                cam = self.make_cam(image, myo_seg=myo_seg, select_values="pos_only", upsampling="bilinear")
                for metric in self.dices:
                    metric[split](cam, fibrosis_seg_label)

                if self.example_batch[split] == None:
                    self.example_batch[split] = batch

        return batch_dict




    def training_step(self, train_batch, batch_idx):
        batch_dict = self._step(train_batch, "Train")
        return batch_dict
        """
        x = train_batch['img']#train_batch[-2]
        labels = train_batch['label']#train_batch[-1]
        myo_seg = train_batch['myo_seg']
        fibrosis_seg_label = train_batch['fibrosis_seg_label']
        self.step_dice(self.train_dice, x, fibrosis_seg_label, myo_seg, cheat=self.cheat_dice)
        self.step_dice(self.train_dice2, x, fibrosis_seg_label, myo_seg, cheat=self.cheat_dice)
        self.step_dice(self.train_dice3, x, fibrosis_seg_label, myo_seg, cheat=self.cheat_dice)
        outputs = self.forward(x, myo_seg)
        preds = torch.squeeze(self.output2pred(outputs))


        if self.num_classes == 1:
            loss = self.bce_with_logits_loss(outputs, labels)
        else:
            loss = self.cross_entropy_loss(outputs, labels)

        self.train_accuracy(preds, labels)
        if (self.current_epoch + 1) % self.heavy_log == 0:
            self.train_auroc(preds, labels)
            self.train_confusion(preds, labels)
            self.train_pr_curve(preds, labels)
            self.train_roc(preds, labels)
            #if self.train_reference_img_batch == None:
            #    self.train_reference_img_batch = x
            #    self.train_reference_label_batch = labels
            if self.train_example_batch == None:
                self.train_example_batch = train_batch

        batch_dict= {
            "loss" : loss
        }
        return batch_dict"""



    def validation_step(self, val_batch, batch_idx):
        batch_dict = self._step(val_batch, "Val")
        return batch_dict

        """
        x = val_batch['img']#val_batch[-2]
        labels = val_batch['label']#val_batch[-1]
        myo_seg = val_batch['myo_seg']
        fibrosis_seg_label = val_batch['fibrosis_seg_label']

        self.step_dice(self.validation_dice, x, fibrosis_seg_label, myo_seg, cheat=self.cheat_dice)
        self.step_dice(self.validation_dice2, x, fibrosis_seg_label, myo_seg, cheat=self.cheat_dice)
        self.step_dice(self.validation_dice3, x, fibrosis_seg_label, myo_seg, cheat=self.cheat_dice)

        outputs = self.forward(x, myo_seg)
        print(f"{outputs.shape=}")
        preds = torch.squeeze(self.output2pred(outputs))

        if self.num_classes == 1:
            loss = self.bce_with_logits_loss(outputs, labels)
        else:
            loss = self.cross_entropy_loss(outputs, labels)

        self.validation_accuracy(preds, labels)
        if (self.current_epoch + 1) % self.heavy_log == 0:
            self.validation_auroc(preds, labels)
            self.validation_confusion(preds, labels)
            self.validation_pr_curve(preds, labels)
            self.validation_roc(preds, labels)
            #if self.val_reference_img_batch == None:
            #    self.val_reference_img_batch = x
            #    self.val_reference_label_batch = labels
            if self.val_example_batch == None:
                self.val_example_batch = val_batch

        batch_dict= {
            "loss" : loss
        }
        return batch_dict"""



    def test_step(self, test_batch, batch_idx):
        batch_dict = self._step(test_batch, "Test")
        return batch_dict

        """
        x = test_batch['img']#test_batch[-2]
        labels = test_batch['label']#test_batch[-1]
        myo_seg = test_batch['myo_seg']

        outputs = self.forward(x, myo_seg)
        preds = torch.squeeze(self.output2pred(outputs))

        if self.num_classes == 1:
            loss = self.bce_with_logits_loss(outputs, labels)
        else:
            loss = self.cross_entropy_loss(outputs, labels)

        self.test_accuracy(preds, labels)
        self.test_auroc(preds, labels)
        self.test_confusion(preds, labels)
        self.test_pr_curve(preds, labels)
        self.test_roc(preds, labels)
        batch_dict= {
            "loss" : loss
        }
        return batch_dict"""


    def log_confusion_matrix(self, split):
        class_names = ["Normal", "Fibrotic"]

        cm = self.confusion[split].compute().detach().cpu().numpy()
        self.confusion[split].reset()

        figure = plt.figure(figsize=(7, 7))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          color = "white" if cm[i, j] > threshold else "black"
          plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        self.logger.experiment.add_figure(f"Confusion/{split}", figure, self.current_epoch)



    def log_pr_curve(self, split):
        precision, recall, thresholds = self.pr_curve[split].compute()
        self.pr_curve[split].reset()

        if not isinstance(precision, list):
            precision, recall = [precision], [recall]

        figure = plt.figure(figsize=(7, 7))
        plt.title("Precision-recall curve")
        plt.tight_layout()
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        for precision_class, recall_class in zip(precision, recall):
            self.auc[split](recall_class, precision_class)
            auc_class = self.auc[split].compute()
            self.auc[split].reset()

            plt.plot(recall_class.detach().cpu(), precision_class.detach().cpu(), label=f"AUC:{auc_class:.3f}")
        plt.legend()

        self.logger.experiment.add_figure(f"PR-curve/{split}", figure, self.current_epoch)



    def log_roc(self, split):
        fpr, tpr, thresholds = self.roc[split].compute()
        self.roc[split].reset()

        if not isinstance(fpr, list):
            fpr, tpr = [fpr], [tpr]

        figure = plt.figure(figsize=(7, 7))
        plt.title("Receiver Operating Characteristic")
        plt.tight_layout()
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.plot([0, 1], [0, 1], linestyle='--')
        for fpr_class, tpr_class in zip(fpr, tpr):
            self.auc[split](fpr_class, tpr_class)
            auc_class = self.auc[split].compute()
            self.auc[split].reset()
            plt.plot(fpr_class.detach().cpu(), tpr_class.detach().cpu(), label=f"AUC:{auc_class:.3f}")
        plt.legend()

        self.logger.experiment.add_figure(f"ROC/{split}", figure, self.current_epoch)


    def log_cams(self, example_batch, split_name, upsampling="bilinear", select_values="all"):
        # reference image grid with labels
        example_img_batch = example_batch['img']
        example_label_batch = example_batch['label']
        example_myo_seg_batch = example_batch['myo_seg']
        example_fibrosis_seg_batch = example_batch['fibrosis_seg_label']
        example_img_path = example_batch['img_path']
        example_prediction_batch = self.output2pred(self.forward(example_img_batch, example_myo_seg_batch)).squeeze(dim=1).cpu().detach()
        #torch.squeeze(self.output2pred(self.forward(example_img_batch, example_myo_seg_batch))).cpu().detach()

        grid_imgs = make_grid_with_labels(example_img_batch, example_label_batch, example_prediction_batch, nrow=4, normalize=True).detach().cpu()
        grid_myo = torchvision.utils.make_grid(example_myo_seg_batch, nrow=4, padding=2).cpu().detach()
        grid_myo_masked = np.ma.masked_where(grid_myo < self.myo_mask_threshold, grid_myo)
        grid_fibrosis = torchvision.utils.make_grid(example_fibrosis_seg_batch, nrow=4, padding=2).cpu().detach()
        grid_fibrosis_masked = np.ma.masked_where(grid_fibrosis <= 0, grid_fibrosis)

        # cams bilinear all
        cam_batch = self.make_cam(example_img_batch, select_values=select_values, upsampling=upsampling, myo_seg=example_myo_seg_batch).cpu().detach()
        grid_cams = torchvision.utils.make_grid(cam_batch, nrow=4, padding=2).cpu().detach()
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        fig.colorbar(cm.ScalarMappable(cmap="coolwarm"), ax=axs.ravel().tolist())
        axs[0].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[1].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[1].imshow(grid_myo_masked[0,:,:], cmap="Blues", alpha=0.5, vmin=0, vmax=1)
        axs[1].imshow(grid_fibrosis_masked[0,:,:], cmap="Reds", alpha=0.8, vmin=0, vmax=1)
        axs[2].imshow(grid_imgs[0,:,:], cmap="gray")
        axs[2].imshow(grid_cams[0,:,:], cmap="coolwarm", alpha=0.5, vmin=0, vmax=1)


        axs[0].set_axis_off()
        axs[1].set_axis_off()
        axs[2].set_axis_off()
        self.logger.experiment.add_figure(f"CAM_{select_values=}_{upsampling=}/{split_name}", fig, self.current_epoch)
        return



    def _epoch_end(self, outputs, split):
        assert split in ["Train", "Val", "Test"]
        epoch_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f"Loss/{split}", epoch_loss, self.current_epoch)

        epoch_accuracy = self.accuracy[split].compute()
        self.accuracy[split].reset()
        self.logger.experiment.add_scalar(f"Accuracy/{split}", epoch_accuracy, self.current_epoch)

        if (self.current_epoch + 1) % self.heavy_log == 0:
            epoch_auroc = self.auroc[split].compute()
            self.auroc[split].reset()
            self.logger.experiment.add_scalar(f"AUROC/{split}", epoch_auroc, self.current_epoch)

            self.log_confusion_matrix(split)
            self.log_pr_curve(split)
            self.log_roc(split)

            self.log_cams(self.example_batch[split], split, upsampling="bilinear", select_values="pos_only")
            self.log_cams(self.example_batch[split], split, upsampling="bilinear", select_values="all")
            self.log_cams(self.example_batch[split], split, upsampling="bilinear", select_values="sigmoid")
            self.example_batch[split] = None

            # log dice
            for metric in self.dices:
                epoch_m = metric[split].compute()
                #print(f"{metric[split].total=}, {epoch_m=}")
                metric[split].reset()
                self.logger.experiment.add_scalar(f"{metric[split].name}/{split}", epoch_m, self.current_epoch)




    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "Train")

        """
        epoch_loss = torch.stack([x['loss'] for x in outputs]).mean()
        epoch_accuracy = self.train_accuracy.compute()
        self.train_accuracy.reset()
        self.logger.experiment.add_scalar("Loss/Train", epoch_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train", epoch_accuracy, self.current_epoch)

        epoch_dice = self.train_dice.compute()
        epoch_dice2 = self.train_dice2.compute()
        epoch_dice3 = self.train_dice3.compute()
        self.train_dice.reset()
        self.train_dice2.reset()
        self.train_dice3.reset()
        self.logger.experiment.add_scalar("Dice_t=0.5/Train", epoch_dice, self.current_epoch)
        self.logger.experiment.add_scalar("Dice_t=0.8/Train", epoch_dice2, self.current_epoch)
        self.logger.experiment.add_scalar("Dice_t=0.2/Train", epoch_dice3, self.current_epoch)

        if (self.current_epoch + 1) % self.heavy_log == 0:
            epoch_auroc = self.train_auroc.compute()
            self.train_auroc.reset()
            confusion_matrix = self.train_confusion.compute()
            self.train_confusion.reset()
            confusion_figure = plot_confusion_matrix(confusion_matrix, range(confusion_matrix.shape[0]))
            precision, recall, pr_thresholds = self.train_pr_curve.compute()
            self.train_pr_curve.reset()
            pr_curve_figure = plot_pr_curve(precision, recall, pr_thresholds, self.train_auc)
            self.train_auc.reset()
            fpr, tpr, roc_thresholds = self.train_roc.compute()
            self.train_roc.reset()
            roc_figure = plot_roc(fpr, tpr, roc_thresholds, self.train_auc)
            self.train_auc.reset()

            self.logger.experiment.add_scalar("AUROC/Train", epoch_auroc, self.current_epoch)
            self.logger.experiment.add_figure("Confusion matrix/Train", confusion_figure, self.current_epoch)
            self.logger.experiment.add_figure("PR-curve/Train", pr_curve_figure, self.current_epoch)
            self.logger.experiment.add_figure("ROC/Train", roc_figure, self.current_epoch)
            self.log_cams(self.train_example_batch, "Train", upsampling="conv_transpose", select_values="pos_only")
            self.log_cams(self.train_example_batch, "Train", upsampling="conv_transpose", select_values="all")
            self.log_cams(self.train_example_batch, "Train", upsampling="conv_transpose", select_values="sigmoid")
            self.train_example_batch = None"""



    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "Val")

        """
        epoch_loss = torch.stack([x['loss'] for x in outputs]).mean()
        epoch_accuracy = self.validation_accuracy.compute()
        self.validation_accuracy.reset()
        self.logger.experiment.add_scalar("Loss/Val", epoch_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Val", epoch_accuracy, self.current_epoch)

        epoch_dice = self.validation_dice.compute()
        epoch_dice2 = self.validation_dice2.compute()
        epoch_dice3 = self.validation_dice3.compute()
        self.validation_dice.reset()
        self.validation_dice2.reset()
        self.validation_dice3.reset()
        self.logger.experiment.add_scalar("Dice_t=0.5/Val", epoch_dice, self.current_epoch)
        self.logger.experiment.add_scalar("Dice_t=0.8/Val", epoch_dice2, self.current_epoch)
        self.logger.experiment.add_scalar("Dice_t=0.2/Val", epoch_dice3, self.current_epoch)

        if (self.current_epoch + 1) % self.heavy_log == 0:
            epoch_auroc = self.validation_auroc.compute()
            self.validation_auroc.reset()
            confusion_matrix = self.validation_confusion.compute()
            self.validation_confusion.reset()
            confusion_figure = plot_confusion_matrix(confusion_matrix, range(confusion_matrix.shape[0]))
            precision, recall, thresholds = self.validation_pr_curve.compute()
            self.validation_pr_curve.reset()
            pr_curve_figure = plot_pr_curve(precision, recall, thresholds, self.validation_auc)
            self.validation_auc.reset()
            fpr, tpr, roc_thresholds = self.validation_roc.compute()
            self.validation_roc.reset()
            roc_figure = plot_roc(fpr, tpr, roc_thresholds, self.validation_auc)
            self.validation_auc.reset()

            self.logger.experiment.add_scalar("AUROC/Val", epoch_auroc, self.current_epoch)
            self.logger.experiment.add_figure("Confusion matrix/Val", confusion_figure, self.current_epoch)
            self.logger.experiment.add_figure("PR-curve/Val", pr_curve_figure, self.current_epoch)
            self.logger.experiment.add_figure("ROC/Val", roc_figure, self.current_epoch)
            self.log_cams(self.val_example_batch, "Val", upsampling="conv_transpose", select_values="pos_only")
            self.log_cams(self.val_example_batch, "Val", upsampling="conv_transpose", select_values="all")
            self.log_cams(self.val_example_batch, "Val", upsampling="conv_transpose", select_values="sigmoid")"""



    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "Test")

        """
        epoch_loss = torch.stack([x['loss'] for x in outputs]).mean()
        epoch_accuracy = self.test_accuracy.compute()
        self.test_accuracy.reset()
        epoch_auroc = self.test_auroc.compute()
        self.test_auroc.reset()
        confusion_matrix = self.test_confusion.compute()
        self.test_confusion.reset()
        confusion_figure = plot_confusion_matrix(confusion_matrix, range(confusion_matrix.shape[0]))
        precision, recall, thresholds = self.test_pr_curve.compute()
        self.test_pr_curve.reset()
        pr_curve_figure = plot_pr_curve(precision, recall, thresholds, self.test_auc)
        self.test_auc.reset()
        fpr, tpr, roc_thresholds = self.test_roc.compute()
        self.test_roc.reset()
        roc_figure = plot_roc(fpr, tpr, roc_thresholds, self.test_auc)
        self.test_auc.reset()

        self.logger.experiment.add_scalar("Loss/Test", epoch_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Test", epoch_accuracy, self.current_epoch)
        self.logger.experiment.add_scalar("AUROC/Test", epoch_auroc, self.current_epoch)
        self.logger.experiment.add_figure("Confusion matrix/Test", confusion_figure, self.current_epoch)
        self.logger.experiment.add_figure("PR-curve/Test", pr_curve_figure, self.current_epoch)
        self.logger.experiment.add_figure("ROC/Test", roc_figure, self.current_epoch)"""












def plot_confusion_matrix(cm, class_names):
  """
  stolen from www.tensorflow.org/tensorboard/image_summaries

  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  cm = cm.detach().cpu().numpy()
  figure = plt.figure(figsize=(7, 7))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def plot_pr_curve(precision, recall, thresholds, auc):
    if not isinstance(precision, list):
        precision, recall = [precision], [recall]
    figure = plt.figure(figsize=(7, 7))
    plt.title("Precision-recall curve")
    plt.tight_layout()
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    for precision_class, recall_class in zip(precision, recall):
        auc(recall_class, precision_class)
        auc_class = auc.compute()
        plt.plot(recall_class.detach().cpu(), precision_class.detach().cpu(), label=f"AUC:{auc_class:.3f}")
    plt.legend()
    return figure

def plot_roc(fpr, tpr, thresholds, auc):
    if not isinstance(fpr, list):
        fpr, tpr = [fpr], [tpr]
    figure = plt.figure(figsize=(7, 7))
    plt.title("Receiver Operating Characteristic")
    plt.tight_layout()
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.plot([0, 1], [0, 1], linestyle='--')
    for fpr_class, tpr_class in zip(fpr, tpr):
        auc(fpr_class, tpr_class)
        auc_class = auc.compute()
        plt.plot(fpr_class.detach().cpu(), tpr_class.detach().cpu(), label=f"AUC:{auc_class:.3f}")
    plt.legend()
    return figure


irange = range

def make_grid_with_labels(tensor, labels, predictions, nrow=8, limit=20, padding=2,
                          normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        labels (Tensor):  labels tensor of shape (B)
        predicitons (Tensor):  predictions tensor of shape (B)
        limit ( int, optional): Limits number of images and labels to make grid of
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    # Opencv configs
    if limit is not None:
        tensor = tensor[:limit, ::]
        labels = labels[:limit]
        predictions = predictions[:limit]


    font = 1
    fontScale = 1
    color = (255, 0, 0)
    thickness = 1

    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            working_tensor = tensor[k]
            if labels is not None:
                org = (0, int(tensor[k].shape[1] * 0.1))
                working_image = cv2.UMat(
                    np.asarray(np.transpose(working_tensor.cpu().numpy(), (1, 2, 0)) * 255).astype('uint8'))
                image = cv2.putText(working_image, f"label: {str(labels[k].item())}      pred: {predictions[k].item():.2f}",
                                    org, font, fontScale, color, thickness, cv2.LINE_AA)
                working_tensor = transforms.ToTensor()(image.get())
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(working_tensor)
            k = k + 1
    return grid



class NormalizedGlobalWeightedPooling(nn.Module):
    """ nGWP as described in https://arxiv.org/pdf/2005.08104.pdf, following implementation https://github.com/visinf/1-stage-wseg/blob/master/models/stage_net.py
        Assumes input shape (B, C, H, W)"""
    def __init__(self, background_value=1.0, eps=1.0):
        super().__init__()
        self.background_value = background_value
        self.eps = eps

    def forward(self, x):
        # add background channel
        background = torch.ones_like(x[:, :1])
        x = torch.cat([background, x], dim=1)
        # softmax with background channel
        masks = F.softmax(x, dim=1)
        # scores
        y = (x * masks).sum(-1).sum(-1) / (self.eps + masks.sum(-1).sum(-1))
        # remove the background channel
        y = y[:, 1:]
        return y


class MaskedAdaptiveAvgPool2d(nn.Module):
    """Assumes input shape (B, C, H, W),
       output shape (B, C, output_size, output_size)
        Takes adaptive average only where mask is True"""
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x, mask):
        #print(f"{x=}")
        #print(f"{mask=}")

        x = x * mask
        #print(f"{x=}")
        x = F.adaptive_avg_pool2d(x, self.output_size)
        # compensate for masked contribution to avg
        #number_false = (mask == False).sum(dim=(2, 3), keepdims=True)
        #number_true = (mask == True).sum(dim=(2, 3), keepdims=True)
        #print(f"{number_false=}")
        #print(f"{number_true=}")
        #x = x * (number_false + number_true) / (number_true + 1e-6)
        #print(f"{x=}")
        """ Empty masks will have value zero -> prediction 0.5
            Probably not good (when using good myocardium predictions), since no myocardium means no fibrosis (i.e. prediction should be 0)"""
        return x



class MaskedAdaptiveMaxPool2d(nn.Module):
    """Assumes input shape (B, C, H, W),
       output shape (B, C, output_size, output_size)
        Takes adaptive max only where mask is True"""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x, mask):
        # first zero out masked values
        x = x * mask
        # set masked out values to low value (consequence: empty mask leads to prediction ~0)
        x[mask == False] = -10000
        # take normal max pooling
        x = F.adaptive_max_pool2d(x, self.output_size)
        """Empty masks will have value -10000 -> prediction ~0.0"""
        return x









if __name__ == '__main__':
    print("nothing here")
