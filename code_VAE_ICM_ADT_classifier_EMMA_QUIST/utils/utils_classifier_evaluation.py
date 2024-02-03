import json
import torch
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryROC, BinaryAUROC
from sklearn.metrics import confusion_matrix as conf_matrix

def auroc_accuracy(targets, weights, sigmoid_applied=False, n_out=2):
        """
        Evaluates the mean accuracy in prediction
 
        """
        print(targets, weights)
        if not sigmoid_applied:
            sigmoid = torch.nn.Sigmoid()
            weights = sigmoid(weights)

        weights = weights.squeeze()
        if n_out == 2:
            labels = torch.argmax(targets, dim=1)
            predictions = torch.argmax(weights, dim=1)
        else:
            labels = targets
            predictions = np.zeros_like(weights)
            predictions[weights >= 0.5] = 1

        correct = predictions == labels

        acc = torch.sum(correct.float()) /labels.view(-1).size(0)
        b_auroc = BinaryAUROC()
        score_roc_auc = b_auroc(weights.detach().cpu(), targets.detach().cpu())

        return acc, score_roc_auc

def make_confusion_matrix(args, cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sn.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

    if args.external_val:
        file_name = 'external_cm.png'
    else:
        file_name = 'cm.png'
    try:
        plt.savefig(args.eval_dir + file_name)
    except:
        plt.savefig(args.logdir + file_name)

    plt.close()

    return precision, recall, f1_score, accuracy

def make_roc(args, weights, targets, title=False):

    metric = BinaryROC()
    metric.update(weights, targets)
    # plt.figure()
    fig, ax = metric.plot(score=True)
    if args.external_val:
        file_name = 'external_roc.png'
    else:
        file_name = 'roc.png'
    try:
        fig.savefig(args.eval_dir + file_name)
    except:
        fig.savefig(args.logdir + file_name)
    plt.close()

def roc_cm(args, weights, targets, sigmoid_applied=False, n_out=2):
    '''
    method of plotting confusions matrix with first numerical counts and second percentage values
    '''
    if not sigmoid_applied:
        sigmoid = torch.nn.Sigmoid()
        weights = sigmoid(weights)

    weights = weights.squeeze()
    if n_out == 2:
        labels = torch.argmax(targets, dim=1).detach().cpu()
        predictions = torch.argmax(weights, dim=1).detach().cpu()
        weights = weights.detach().cpu()
        targets = targets.detach().cpu()
    else:
        targets = torch.tensor(targets)
        weights = torch.tensor(weights)
        labels = targets
        predictions = weights
        
    #     predictions = np.array(outputs.detach().cpu())
    #     labels = np.array(train_labels.detach().cpu())

    cf_matrix = conf_matrix(labels, predictions)
    
    group_names = ['TN','FP','FN','TP']
    categories = ['0', '1']

    precision, recall, f1_score, accuracy = make_confusion_matrix(args, cf_matrix, 
                                                                    group_names=group_names,
                                                                    categories=categories, 
                                                                   cmap='Blues')  
    if n_out == 2:
        make_roc(args, weights, targets)  

    return precision, recall, f1_score, accuracy


