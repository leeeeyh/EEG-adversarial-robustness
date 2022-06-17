import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from utils.hsic import *

def hbar_loss(model,
            x_natural,
            y,
            device,
            optimizer,
            class_weight,
            config_dict=[],
            ):
    model.train()

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    pred_natural, Z_natural = model(x_natural)
    loss_natural = F.cross_entropy(pred_natural*class_weight, y)

    # compute hsic
    h_target = y.reshape(-1,1)
    h_target = to_categorical(h_target, num_classes = len(torch.unique(y))).float().to(device)
    h_data = x_natural.reshape(-1, np.prod(x_natural.size()[1:]))
    hiddens = Z_natural

    # new variable
    hx_l_list = []
    hy_l_list = []
    lx, ly, ld = config_dict['lambda_x'], config_dict['lambda_y'], 0.
    if ld > 0:
        lx, ly = lx * (ld ** len(hiddens)), ly * (ld ** len(hiddens))

    loss_robust = 0  
    for i in range(len(hiddens)):
        
        if len(hiddens[i].size()) > 2:
            hiddens[i] = hiddens[i].reshape(-1, np.prod(hiddens[i].size()[1:]))

        hx_l, hy_l = hsic_objective(
                hiddens[i],
                h_target=h_target.float(),
                h_data=h_data,
        )

        hx_l_list.append(hx_l)
        hy_l_list.append(hy_l)
        
        if ld > 0:
            lx, ly = lx/ld, ly/ld
            #print(i, lx, ly)
        temp_hsic = lx * hx_l - ly * hy_l
        loss_robust += temp_hsic.to(device)

    loss = loss_natural + loss_robust

    return loss
