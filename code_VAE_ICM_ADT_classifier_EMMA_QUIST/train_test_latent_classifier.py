import torch 
import numpy as np
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
# fit a svm on an imbalanced classification dataset
from numpy import mean

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC

from utils.utils_general import early_stopping
from utils.utils_classifier_evaluation import  roc_cm, auroc_accuracy

def obtain_latents(args, model, full_loader, test_loader):
    model.eval()
    mus_train, mus_test = [], []
    labels_train, labels_test = [], []
    with torch.no_grad():
        for images, label_batch in full_loader:
            images = Variable(images.to(args.device, dtype=torch.float))

            if args.model_type == 'vae_mlp':
                label_batch = Variable(label_batch).to(args.device)

            z, mu, logvar = model.encoder(images, sample=True)
            mu = z
            # recon_batch, mu, logvar =  output_dict['img'],  output_dict['mu'],  output_dict['log_var']  
            for i in range(len(mu)):
                mus_train.append(mu[i])
                labels_train.append(label_batch[i])
        
        for images, label_test_batch in test_loader:
            images = Variable(images.to(args.device, dtype=torch.float))

            if args.model_type == 'vae_mlp':
                label_test_batch = Variable(label_test_batch).to(args.device)

            z, mu, logvar  = model.encoder(images, sample=True)
            mu = z
            
            # recon_batch, mu, logvar =  output_dict['img'],  output_dict['mu'],  output_dict['log_var']  
            
            for i in range(len(mu)):
                mus_test.append(mu[i])
                labels_test.append(label_test_batch[i])

    return mus_train, labels_train, mus_test, labels_test

def test_deep_classifier(args, loader, model, loss_fn, n_out=1):
    '''
    function to calculate validation accuracy and loss
    '''

    running_loss = 0
    n = 1    # counter for number of minibatches
    output_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        for data in loader:
            latents, labels = data
            latents = latents.float().to(args.device)
    
            outputs = model(latents)    

            if n_out == 2:
                labels = torch.nn.functional.one_hot(labels, num_classes=2).to(args.device)
            else:
                labels = labels.reshape(-1, 1).float().to(args.device)
                outputs = outputs.reshape(-1, 1)

            
            running_loss +=  (loss_fn(outputs.type(torch.float),  labels.type(torch.float)) / labels.shape[0])

            n += 1

            output_list.append(outputs.detach().cpu())
            label_list.append(labels.detach().cpu())


        output_list = torch.concat(output_list)
        label_list = torch.concat(label_list)
    return running_loss.cpu()/n, output_list, label_list

def train_deep_classifier(args, train_loader, valid_loader, test_loader, mlp_model, writer, n_out=1):
    mlp_model.to(args.device)

    statsrec = np.zeros((4,args.max_epochs))
    if n_out == 2:
        if type(args.class_weights) != bool:
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=args.class_weights.to(args.device))
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fn = nn.BCELoss(reduction='sum')

    optimizer = optim.Adam(mlp_model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, 
                                                               threshold=0.001, threshold_mode='abs')
    counter = 0
    best_acc = np.finfo('f').min
    best_auc = np.finfo('f').min
    for epoch in range(1,args.max_epochs+1):  
        running_loss = 0.0   
        n = 0       
        mlp_model.train()
        output_list = []
        label_list = []
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.float().to(args.device)

            optimizer.zero_grad()

            outputs = mlp_model(inputs)
            if n_out == 2:
                labels = torch.nn.functional.one_hot(labels, num_classes=2).to(args.device)
                
            else:
                labels = labels.reshape(-1, 1).float().to(args.device)
                outputs = outputs.reshape(-1, 1)

            loss = loss_fn(outputs.type(torch.float),  labels.type(torch.float)) / labels.shape[0]

            loss.backward()
            optimizer.step()

            output_list.append(outputs.detach().cpu())
            label_list.append(labels.detach().cpu())

            running_loss += loss.item()

            n += 1

        train_outputs = torch.concat(output_list)
        train_labels = torch.concat(label_list)
        if n_out == 2:
            train_acc, train_auroc = auroc_accuracy(train_labels, train_outputs, sigmoid_applied=False)
        else:
            train_acc, train_auroc = auroc_accuracy(train_labels, train_outputs, sigmoid_applied=True)

        train_loss = running_loss/n

        val_loss, val_outputs, val_labels = test_deep_classifier(args, valid_loader, mlp_model, loss_fn=loss_fn, n_out=n_out)

        if n_out == 2:
            val_acc, val_auroc = auroc_accuracy(val_labels, val_outputs, sigmoid_applied=False)
        else:
            val_acc, val_auroc = auroc_accuracy(val_labels, val_outputs, sigmoid_applied=True)

        if val_auroc > best_auc:
            best_auc = val_auroc
            best_test_loss_epoch = epoch+1
            print(f"Best loss is achieved {best_auc} in the epoch: {best_test_loss_epoch}")
            torch.save(mlp_model.state_dict(), args.logdir + "best_auc_loss_model.pth")
        if val_acc > best_acc:
            best_acc = val_acc
            best_metric_epoch = epoch + 1
            print(f"Best accuracy is achieved {best_acc} in the epoch: {best_metric_epoch}")
            torch.save(mlp_model.state_dict(), args.logdir +  "best_acc_model.pth")
        statsrec[:,epoch-1] = (train_loss, train_acc, val_loss.item(), val_loss)
        if epoch % 75 == 0 or epoch == 1 or epoch == args.max_epochs - 1 or counter == 25:
            print(f"epoch: {epoch} training loss: {train_loss: .3f} training accuracy: {train_acc: .1%}  validation loss: {val_loss: .3f} validation accuracy: {val_acc: .1%}")
           
        writer.add_scalar("MLP loss training", train_loss, epoch)
        writer.add_scalar("MLP loss Validation", val_loss, epoch) 
        writer.add_scalar("Acc training", train_acc, epoch)
        writer.add_scalar("Acc validation", val_acc, epoch)
        writer.add_scalar("AUC training", train_auroc, epoch)
        writer.add_scalar("AUC validation", val_auroc, epoch)

        scheduler.step(val_loss)
        # counter = early_stopping(counter, train_loss, val_loss, min_delta=0.4)

        # if counter > 25:
        #     print("At Epoch:", epoch)
        #     break
    best_model = args.best_model
    mlp_model.load_state_dict(torch.load(args.logdir + "best_{}_model.pth".format(best_model)))
    test_loss, test_outputs, test_labels = test_deep_classifier(args, test_loader, mlp_model,  loss_fn=loss_fn, n_out=n_out)
    
    if n_out == 2:
        precision, recall, f1_score, accuracy = roc_cm(args, test_outputs, test_labels, sigmoid_applied=False, n_out=2)
    else:
        precision, recall, f1_score, accuracy = roc_cm(args, test_outputs, test_labels, sigmoid_applied=False, n_out=1)

    if n_out == 2:
        _, auroc = auroc_accuracy(test_labels, test_outputs, sigmoid_applied=False)
    else:
        _, auroc = auroc_accuracy(test_labels, test_outputs, sigmoid_applied=True)

    print('test_loss:', test_loss.item(),'test accuracy:', accuracy, '%')  
    # save network parameters, losses and accuracy
    # torch.save({"stats": statsrec}, save_results_path) #"state_dict": model.state_dict()

    print('AUC is:', auroc)

    writer.add_scalar("MLP loss test", test_loss, 1)
    writer.add_scalar("Precision test", precision, 1)
    writer.add_scalar("Recall test", recall, 1) 
    writer.add_scalar("F1 test", f1_score, 1)  
    writer.add_scalar("Acc test", accuracy, 1)
    writer.add_scalar("AUC test", auroc, 1)
    return test_loss.item(), precision, recall, f1_score, accuracy, auroc

def train_test_svm(args, train_latents, train_labels, test_latents, test_labels, model, writer, n_out=1):

    m = nn.AvgPool3d(3, stride=2)

    train_latents = m(train_latents).view(len(train_labels), -1)
    test_latents = m(test_latents).view(len(test_labels), -1)
    # SVM Classifier
    scaler = StandardScaler()
    X = train_latents
    X_test = test_latents
    # X = scaler.fit_transform(train_latents)
    # X_test = scaler.fit_transform(test_latents)
    if args.svm_search:
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(X, train_labels)

        print(
            "The best parameters are %s with a score of %0.2f"
            % (grid.best_params_, grid.best_score_)
        )
    else:
        # svc = SVC(C=0.01, kernel='rbf', gamma=9.999999999999999e-10)
        # svc = SVC(C=0.01, kernel='rbf', gamma='scale')
        # svc.fit(X, train_labels)
        # test_outputs = svc.predict(X_test)

        # define model
        model = SVC(gamma='scale')
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X, train_labels, scoring='roc_auc', cv=cv, n_jobs=-1)
        # summarize performance
        print('Mean ROC AUC: %.3f' % mean(scores))
        # fit a svm on an imbalanced classification dataset

        # define model
        model = SVC(gamma='scale')
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=args.seed)
        # evaluate model
        scores = cross_val_score(model, X, train_labels, scoring='roc_auc', cv=cv, n_jobs=-1)
        # summarize performance
        print('Mean ROC AUC: %.3f' % mean(scores))
        model.fit(X, train_labels)
        test_outputs = model.predict(X_test)


    precision, recall, f1_score, accuracy = roc_cm(args, test_outputs, test_labels, sigmoid_applied=True, n_out=1)

    _, auroc = auroc_accuracy(test_labels, test_outputs, sigmoid_applied=True, n_out=1)
    print('AUC is:', auroc)
    
    writer.add_scalar("Precision test", precision, 1)
    writer.add_scalar("Recall test", recall, 1) 
    writer.add_scalar("F1 test", f1_score, 1)  
    writer.add_scalar("Acc test", accuracy, 1)
    writer.add_scalar("AUC test", auroc, 1)

    return 0, precision, recall, f1_score, accuracy, auroc



