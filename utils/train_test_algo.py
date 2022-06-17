import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from utils.utils import *
from utils.atk import fgsm, Sap, tlm_uap
from utils.trades import trades_loss
from utils.mart import mart_loss
from utils.hbar import hbar_loss
from utils.hbar_pgd import hbar_pgd_loss
from utils.pgd import pgd_loss


# %% define train and test function
def train(model,
            device,
            loss_fn,
            optimizer, 
            data,
            label,
            class_weight,
            batch_size=64,
            df_method=None,
            step_size=0.003,
            epsilon=0.031,
            config_dict = [],
            ):
    model.train()
    train_loss, correct = 0, 0
    y_true=label.cpu().tolist()
    y_pred=[]
    size = len(data)
    num_batches = np.int(np.ceil(np.float(size) / np.float(batch_size)))
    for batch_idx in range(num_batches):
        s = batch_idx * batch_size
        e = min((batch_idx + 1) * batch_size, size)
        X = data[s:e]
        y = label[s:e]

        pred = model(X)[0]
        if df_method == 'trades':
            loss = trades_loss(model=model,
                               x_natural=X,
                               y=y,
                               optimizer=optimizer,
                               step_size=step_size,
                               epsilon=epsilon,
                               beta=config_dict[df_method],
                               class_weight=class_weight,
                               )

        elif df_method == 'mart':
            loss = mart_loss(model=model,
                               x_natural=X,
                               y=y,
                               optimizer=optimizer,
                               step_size=step_size,
                               epsilon=epsilon,
                               beta=config_dict[df_method],
                               class_weight=class_weight,
                               )
        elif df_method == 'hbar':
            loss = hbar_loss(model=model,
                            x_natural=X,
                            y=y,
                            optimizer=optimizer,
                            device=device,
                            class_weight=class_weight,
                            config_dict=config_dict[df_method],
            )

        elif df_method == 'hbar_pgd':
            loss = hbar_pgd_loss(model=model,
                            x_natural=X,
                            y=y,
                            optimizer=optimizer,
                            step_size=step_size,
                            epsilon=epsilon,
                            device=device,
                            class_weight=class_weight,
                            config_dict=config_dict[df_method],
            )
        
        elif df_method == 'pgd':
            loss = pgd_loss(model=model,
                            x_natural=X,
                            y=y,
                            optimizer=optimizer,
                            step_size=step_size,
                            epsilon=epsilon,
                            class_weight=class_weight,
            )

        elif df_method == None or df_method == 'none':
            loss_fn = nn.CrossEntropyLoss(weight=class_weight).to(device)
            loss = loss_fn(pred, y,)
        else:
            raise ValueError('Please choose correct defense method') 

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # correct += balanced_accuracy_score(y.cpu(), pred.max(1)[1].cpu())
        # print(balanced_accuracy_score(y.cpu(), pred.max(1)[1].cpu()), (pred.max(1)[1] == y).type(torch.float).sum().item())
        correct += (pred.max(1)[1] == y).type(torch.float).sum().item()
        y_pred += pred.max(1)[1].cpu().numpy().tolist()

    train_loss /= float(size)
    correct /= float(size)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    # print('train_loss:',train_loss)
        # if batch_idx % 500 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
    return correct, train_loss

def test(model, device, loss_fn, data, label, batch_size):
    model.eval()
    test_loss, correct = 0, 0
    y_true=label.cpu().tolist()
    y_pred=[]
    size = len(data)
    num_batches = np.int(np.ceil(np.float(size) / np.float(batch_size)))
    for batch_idx in range(num_batches):
        s = batch_idx * batch_size
        e = min((batch_idx + 1) * batch_size, size)
        X = data[s:e]
        y = label[s:e]
        with torch.no_grad():
            pred = model(X)[0]
            test_loss += loss_fn(pred, y).item()
            correct += (pred.max(1)[1] == y).type(torch.float).sum().item()
            y_pred += pred.max(1)[1].cpu().numpy().tolist()
    test_loss /= float(size)
    correct /= float(size)

    if len(np.unique(label.cpu().numpy())) == 2:
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        return balanced_acc, test_loss

    return correct, test_loss

def test_fgsm(model, device, data, label, batch_size, eps=0.0001):
    model.eval()
    acc_adv = 0

    TP, FP, TN, FN = 0, 0, 0, 0

    size = len(data)
    num_batches = np.int(np.ceil(np.float(size) / np.float(batch_size)))
    for batch_idx in range(num_batches):
        s = batch_idx * batch_size
        e = min((batch_idx + 1) * batch_size, size)
        X = data[s:e]
        y = label[s:e]

        # X, y = X.to(device), y.type(torch.LongTensor).to(device)
        X_adv = fgsm(model, device, X, y, eps)

        with torch.no_grad():
            pred = model(X)[0]
            init_pred = pred.max(1)[1]
            pred_ = model(X_adv)[0]
            final_pred = pred_.max(1)[1]
            idx1 = (init_pred == y)
            idx2 = (init_pred == final_pred)
            idx = idx1 & idx2
            acc_adv += idx.type(torch.float).sum().item()

            if len(np.unique(label.cpu().numpy())) == 2:
                #二分类 balanced acc
                for i in range(e-s):
                    if init_pred[i] == y[i] and init_pred[i] == final_pred[i]:
                        if y[i]:
                            TP += 1
                        else:
                            TN += 1
                    else:
                        if y[i]:
                            FN += 1
                        else:
                            FP += 1
    
    # savemat('/home/lyh2020/code/EEG_IB/output/result/fgsmadv.mat', {'adv':X_adv.detach().cpu().numpy(), 'benign':X.detach().cpu().numpy()})
        
    if len(np.unique(label.cpu().numpy())) == 2:
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        balanced_acc = (sensitivity + specificity) / 2.
        return balanced_acc

    acc_adv /= float(size)
    
    return acc_adv


def test_tlm_uap(model, device, X_train, y_train, X_test, y_test, batch_size, class_weight, acc_last=1.0, eps=0.0001):
    v, acc_adv, _ = tlm_uap(model, device, X_train, y_train, X_test, y_test, class_weight=class_weight, batch_size=batch_size, eps=eps, acc_last=acc_last)
    # savemat('/home/lyh2020/code/EEG_IB/output/result/uapadv.mat', {'adv':X_test.detach().cpu().numpy()+v, 'benign':X_test.detach().cpu().numpy()})
    return v, acc_adv

def test_sap(model, device, data, label, batch_size, eps=0.0001):
    model.eval()
    acc_adv = 0
    atk = Sap(model=model, device=device, eps=eps)
    TP, FP, TN, FN = 0, 0, 0, 0
    size = len(data)
    num_batches = np.int(np.ceil(np.float(size) / np.float(batch_size)))
    for batch_idx in range(num_batches):
        s = batch_idx * batch_size
        e = min((batch_idx + 1) * batch_size, size)
        X = data[s:e]
        y = label[s:e]

        X_adv = atk.forward(X, y)
        
        with torch.no_grad():
            pred = model(X)[0]
            init_pred = pred.max(1)[1]
            pred_ = model(X_adv)[0]
            final_pred = pred_.max(1)[1]
            idx1 = (init_pred == y)
            idx2 = (init_pred == final_pred)
            idx = idx1 & idx2

            acc_adv += idx.type(torch.float).sum().item()        
            
            if len(np.unique(label.cpu().numpy())) == 2:
                #二分类 balanced acc
                for i in range(e-s):
                    if init_pred[i] == y[i] and init_pred[i] == final_pred[i]:
                        if y[i]:
                            TP += 1
                        else:
                            TN += 1
                    else:
                        if y[i]:
                            FN += 1
                        else:
                            FP += 1
    # savemat('/home/lyh2020/code/EEG_IB/output/result/sapadv.mat', {'adv':X_adv.detach().cpu().numpy(), 'benign':X.detach().cpu().numpy()})

    if len(np.unique(label.cpu().numpy())) == 2:
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        balanced_acc = (sensitivity + specificity) / 2.
        return balanced_acc

    acc_adv /= float(size)
    return acc_adv
