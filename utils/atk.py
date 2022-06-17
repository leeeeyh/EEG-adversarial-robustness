import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os,sys
from sklearn.metrics import balanced_accuracy_score

def fgsm(model, device, images, labels, eps):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    loss = nn.CrossEntropyLoss()

    images.requires_grad = True
    outputs = model(images)[0]

    # Calculate loss
    cost = loss(outputs, labels)

    # Update adversarial images
    grad = torch.autograd.grad(cost, images,
                                retain_graph=False, create_graph=False)[0]

    adv_images = images + eps*grad.sign()

    adv_images = torch.clamp(adv_images, min=torch.min(adv_images)-eps, max=torch.max(adv_images)+eps).detach()

    return adv_images

class Sap():
    def __init__(self, model, device, eps, criterion=nn.CrossEntropyLoss(),
                    step_alpha=1, 
                    num_steps=10, 
                    sizes=[5, 7, 11, 15, 19], 
                    sigmas=[1.0, 3.0, 5.0, 7.0, 10.0],): 
        self.model = model
        self.criterion = criterion
        self.device = device
        self.eps = eps
        self.step_alpha = step_alpha
        self.num_steps = num_steps
        self.sizes = sizes
        self.sigmas = sigmas
        crafting_sizes = []
        crafting_weights = []
        for size in self.sizes:
            for sigma in self.sigmas:
                crafting_sizes.append(size)
                weight = np.arange(size) - size//2
                weight = np.exp(-weight**2.0/2.0/(sigma**2))/np.sum(np.exp(-weight**2.0/2.0/(sigma**2)))
                weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
                crafting_weights.append(weight)
        self.sizes = crafting_sizes
        self.weights = crafting_weights
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i].unsqueeze(0)

    def forward(self, inputs, targets):
        """
        :param inputs: Clean samples (Batch X Size)
        :param targets: True labels
        :param model: Model
        :param criterion: Loss function
        :param gamma:
        :return:
        """
        step_alpha = self.step_alpha
        num_steps = self.num_steps
        sizes = self.sizes
        weights = self.weights
        model = self.model
        eps = self.eps
        criterion = self.criterion
        device = self.device

        crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
        crafting_target = torch.autograd.Variable(targets.clone())
        for _ in range(num_steps):
            output = model(crafting_input)[0]
            loss = criterion(output, crafting_target)
            if crafting_input.grad is not None:
                crafting_input.grad.data.zero_()
            loss.backward()
            added = torch.sign(crafting_input.grad.data)
            step_output = crafting_input + step_alpha * added
            total_adv = step_output - inputs
            total_adv = torch.clamp(total_adv, -eps, eps)
            crafting_output = inputs + total_adv
            crafting_input = torch.autograd.Variable(crafting_output.detach().clone(), requires_grad=True)
        added = crafting_output - inputs
        added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
        for _ in range(num_steps*2):
            temp = F.conv2d(added, weights[0].to(device), padding = (0, sizes[0]//2))
            for j in range(len(sizes)-1):
                temp = temp + F.conv2d(added, weights[j+1].to(device), padding = (0, sizes[j+1]//2))
            temp = temp/float(len(sizes))
            output = model(inputs + temp)[0]
            loss = criterion(output, targets)
            loss.backward()
            added = added + step_alpha * torch.sign(added.grad.data)
            added = torch.clamp(added, -eps, eps)
            added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
        temp = F.conv2d(added, weights[0].to(device), padding = (0, sizes[0]//2))
        for j in range(len(sizes)-1):
            temp = temp + F.conv2d(added, weights[j+1].to(device), padding = (0, sizes[j+1]//2))
        temp = temp/float(len(sizes))
        crafting_output = inputs + temp.detach()
        crafting_output_clamp = crafting_output.clone()

        sys.stdout.flush()
        return  crafting_output_clamp

def tlm_uap(model, device, X_train, y_train, X_test, y_test, class_weight, batch_size=64, eps=0.001, lr=0.001, epochs=500, alpha=0, delta=0.6, acc_last=1.0):
    model_train = copy.deepcopy(model)
    model_val = copy.deepcopy(model)
    optimizer = optim.Adam(model_train.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight).to(device)

    for k,v in model_train.named_parameters():
        if k != 'universal_nosie':
            v.requires_grad=False
        else:
            v.requires_grad=True

    ASR_best = -1
    es = 0
    v_best = np.zeros((1,1,X_test.shape[-2],X_test.shape[-1]))
    _acc_adv = 1.0
    _acc_natural = 0
    
    for _ in range(epochs):
        model_train.train()
        train_loss = 0
        size_train = len(X_train)
        num_batches = np.int(np.ceil(np.float(size_train) / np.float(batch_size)))
        for batch_idx in range(num_batches):
            s = batch_idx * batch_size
            e = min((batch_idx + 1) * batch_size, size_train)
            X = X_train[s:e]
            y = y_train[s:e]

            pred, v = model_train.forward(X, eps)
        
            loss = -torch.mean(loss_fn(pred, y)) + alpha * torch.mean(torch.abs(v))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            v = torch.clamp(v, -eps, eps)
        train_loss /= float(size_train)
        
        model_val.eval()
        acc_natural = 0
        acc_adv = 0
        ASR = 0
        TP, FP, TN, FN = 0, 0, 0, 0
        y_true=y_test.cpu().tolist()
        y_pred=[]
        size_test = len(X_test)
        num_batches = np.int(np.ceil(np.float(size_test) / np.float(batch_size)))
        for batch_idx in range(num_batches):
            s = batch_idx * batch_size
            e = min((batch_idx + 1) * batch_size, size_test)
            X = X_test[s:e]
            y = y_test[s:e]

            with torch.no_grad():
                pred = model_val(X)[0]
                pred_ = model_val(X+v)[0]

                idx1 = (pred.max(1)[1] == y)
                idx2 = (pred.max(1)[1] != pred_.max(1)[1])
                idx3 = (pred.max(1)[1] == pred_.max(1)[1])

                ASR += (idx1 & idx2).type(torch.float).sum().item()

                acc_natural += idx1.type(torch.float).sum().item()
                y_pred += pred.max(1)[1].cpu().numpy().tolist()

                acc_adv += (idx1 & idx3).type(torch.float).sum().item()

                if len(np.unique(y_train.cpu().numpy())) == 2:
                    #二分类 balanced acc
                    for i in range(e-s):
                        if pred.max(1)[1][i] == y[i] and pred.max(1)[1][i] == pred_.max(1)[1][i]:
                            if y[i]:
                                TP += 1
                            else:
                                TN += 1
                        else:
                            if y[i]:
                                FN += 1
                            else:
                                FP += 1

        if len(np.unique(y_train.cpu().numpy())) == 2:
            sensitivity = TP/(TP+FN)
            specificity = TN/(FP+TN)
            balanced_acc_adv = (sensitivity + specificity) / 2.
            balanced_acc_natural = balanced_accuracy_score(y_true, y_pred)
            acc_adv = balanced_acc_adv
            acc_natural = balanced_acc_natural
        else:
            acc_adv /= float(size_test)
            acc_natural /= float(size_test)

        ASR /= float(size_test)

        delta = min(acc_natural-0.10, acc_natural-acc_last+0.40)

        if ASR >= ASR_best and acc_adv <= acc_last+1:
            ASR_best = ASR
            v_best = v.detach().cpu().numpy()
            _acc_adv = acc_adv
            _acc_natural = acc_natural
            es = 0
        else:
            es += 1
            if es > 50:
                break

        if ASR_best >= delta:
            # print('dddd',ASR_best,delta,_acc_adv,_acc_natural,acc_last)
            break
    # print('delta:',delta)
        
    return v_best, _acc_adv, _acc_natural




