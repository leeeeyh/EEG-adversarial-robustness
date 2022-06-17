import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import pdist, squareform
from utils.hsic import *

def hbar_pgd_loss(model,
            x_natural,
            y,
            device,
            optimizer,
            class_weight,
            step_size=0.003,
            epsilon=0.031,
            perturb_steps=10,
            distance='l_inf',
            config_dict=[],
            ):
    model.eval()     
   
    # generate adversarial example
    x_adv = x_natural.detach() + 0.0001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv)[0], y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            # x_adv = torch.clamp(x_adv, 0.0, 1.0)


    model.train()
    # x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_adv = Variable(x_adv, requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    pred_natural, Z_natural = model(x_natural)
    pred_adv, Z_adv = model(x_adv)
    loss_natural = F.cross_entropy(pred_natural*class_weight, y)
    loss_adv = F.cross_entropy(pred_adv*class_weight, y)

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

        #占用cpu过多，如果用gpu运算，结果会变差，不知道为什么
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

    loss = loss_natural + loss_adv + loss_robust

    return loss
