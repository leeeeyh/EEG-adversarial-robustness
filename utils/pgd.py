import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def pgd_loss(model,
            x_natural,
            y,
            optimizer,
            class_weight,
            step_size=0.003,
            epsilon=0.031,
            perturb_steps=10,
            distance='l_inf'):

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
    else:
        # x_adv = torch.clamp(x_adv, 0.0, 1.0)
        pass

    model.train()
    # x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_adv = Variable(x_adv, requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    pred_natural, Z_natural = model(x_natural)
    pred_adv, Z_adv = model(x_adv)
    loss_natural = F.cross_entropy(pred_natural, y, weight=class_weight)
    loss_adv = F.cross_entropy(pred_adv, y, weight=class_weight)
    
    loss = loss_natural + loss_adv 

    return loss

