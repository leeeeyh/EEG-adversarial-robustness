from types import new_class
import torch
import torch.nn as nn
from utils.utils import *

#%% Deep convnet - Baseline 1
class ShallowConvNet(nn.Module):
    def square(self, x):
        return x * x

    def safe_log(self, x, eps=1e-6):
        """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
        return torch.log(torch.clamp(x, min=eps))
        
    def firstBlock(self, outF, dropoutP, kernalSize, nChan):
        return nn.Sequential(
                Conv2dWithConstraint(1,outF, kernalSize, padding = 0, max_norm = 2),
                Conv2dWithConstraint(40, 40, (nChan, 1), padding = 0, bias= False, max_norm = 2),
                nn.BatchNorm2d(outF),
                Expression(self.square),
                nn.AvgPool2d((1,75), stride = (1,15)),
                Expression(self.safe_log),
                nn.Dropout(p=dropoutP),
                )

    def lastBlock(self, inF, outF, kernalSize):
        return nn.Sequential(
                Conv2dWithConstraint(inF, outF, kernalSize, max_norm = 0.5),
                nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass = 2, dropoutP = 0.5):
        super(ShallowConvNet, self).__init__()

        self.nClass = nClass
        self.nChan = nChan
        self.nTime = nTime
        
        self.universal_nosie = torch.nn.Parameter(torch.zeros((1, 1, self.nChan, self.nTime)), requires_grad=False)

        kernalSize = (1,25)
        nFilt_FirstLayer = 40
        nFiltLaterLayer = 40

        self.allButLastLayers = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, self.nChan)

        self.fSize = self.calculateOutSize(self.allButLastLayers, self.nChan, self.nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer, nClass, (1, self.fSize[1]))

    def forward(self, x, eps=0):
        if eps == 0:
            Z = self.allButLastLayers(x)
            x = self.lastLayer(Z)
            x = torch.squeeze(x, 3)
            out = torch.squeeze(x, 2)
            return out, [Z]
        else:
            self.universal_nosie.data = torch.clamp(self.universal_nosie.data, -eps, eps)
            x = x + self.universal_nosie
            x = self.allButLastLayers(x)
            x = self.lastLayer(x)
            x = torch.squeeze(x, 3)
            out = torch.squeeze(x, 2)
            return out, self.universal_nosie.data


#%% Deep convnet - Baseline 1
class DeepConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias= False, max_norm = 2),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1,3), stride = (1,3))
            )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan):
        return nn.Sequential(
                Conv2dWithConstraint(1,outF, kernalSize, padding = 0, max_norm = 2),
                Conv2dWithConstraint(25, 25, (nChan, 1), padding = 0, bias= False, max_norm = 2),
                nn.BatchNorm2d(outF),
                nn.ELU(),
                nn.MaxPool2d((1,3), stride = (1,3))
                )

    def lastBlock(self, inF, outF, kernalSize):
        return nn.Sequential(
                Conv2dWithConstraint(inF, outF, kernalSize, max_norm = 0.5),
                nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass = 4, dropoutP = 0.5):
        super(DeepConvNet, self).__init__()

        self.nClass = nClass
        self.nChan = nChan
        self.nTime = nTime
        
        kernalSize = (1,10)
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]
        self.universal_nosie = torch.nn.Parameter(torch.zeros((1, 1, self.nChan, self.nTime)), requires_grad=False)

        firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, self.nChan)
        middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, kernalSize)
            for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])

        self.allButLastLayers = nn.Sequential(firstLayer, middleLayers)

        self.fSize = self.calculateOutSize(self.allButLastLayers, self.nChan, self.nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))

    def forward(self, x, eps=0):
        if eps == 0:
            Z = self.allButLastLayers(x)
            x = self.lastLayer(Z)
            x = torch.squeeze(x, 3)
            out = torch.squeeze(x, 2)
            return out, [Z]
        else:
            self.universal_nosie.data = torch.clamp(self.universal_nosie.data, -eps, eps)
            x = x + self.universal_nosie
            x = self.allButLastLayers(x)
            x = self.lastLayer(x)
            x = torch.squeeze(x, 3)
            out = torch.squeeze(x, 2)
            return out, self.universal_nosie.data

class DeepConvNet_ERN(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias= False, max_norm = 2),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1,2), stride = (1,2))
            )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan):
        return nn.Sequential(
                Conv2dWithConstraint(1,outF, kernalSize, padding = 0, max_norm = 2),
                Conv2dWithConstraint(25, 25, (nChan, 1), padding = 0, bias= False, max_norm = 2),
                nn.BatchNorm2d(outF),
                nn.ELU(),
                nn.MaxPool2d((1,2), stride = (1,2))
                )

    def lastBlock(self, inF, outF, kernalSize):
        return nn.Sequential(
                Conv2dWithConstraint(inF, outF, kernalSize, max_norm = 0.5),
                nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass = 4, dropoutP = 0.5):
        super(DeepConvNet_ERN, self).__init__()

        self.nClass = nClass
        self.nChan = nChan
        self.nTime = nTime
        
        kernalSize = (1,5)
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]
        self.universal_nosie = torch.nn.Parameter(torch.zeros((1, 1, self.nChan, self.nTime)), requires_grad=False)

        firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, self.nChan)
        middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, kernalSize)
            for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])

        self.allButLastLayers = nn.Sequential(firstLayer, middleLayers)

        self.fSize = self.calculateOutSize(self.allButLastLayers, self.nChan, self.nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))

    def forward(self, x, eps=0):
        if eps == 0:
            Z = self.allButLastLayers(x)
            x = self.lastLayer(Z)
            x = torch.squeeze(x, 3)
            out = torch.squeeze(x, 2)
            return out, [Z]
        else:
            self.universal_nosie.data = torch.clamp(self.universal_nosie.data, -eps, eps)
            x = x + self.universal_nosie
            x = self.allButLastLayers(x)
            x = self.lastLayer(x)
            x = torch.squeeze(x, 3)
            out = torch.squeeze(x, 2)
            return out, self.universal_nosie.data
# EEGnet
#%% EEGNet Baseline 2
class EEGNet(nn.Module):
    def __init__(self, nChan, nTime, nClass = 4,
                 dropoutP = 0.5, F1=8, D = 2,
                 C1 = 125):
        super(EEGNet, self).__init__()
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.nTime = nTime
        self.nClass = nClass
        self.nChan = nChan
        self.C1 = C1

        self.universal_nosie = torch.nn.Parameter(torch.zeros((1, 1, self.nChan, self.nTime)), requires_grad=False)

        self.firstBlocks = self.initialBlocks(dropoutP)
        self.fSize = self.calculateOutSize(self.firstBlocks, self.nChan, self.nTime)
        self.lastLayer = self.lastBlock(self.F2, self.nClass, (1, self.fSize[1]))

    def initialBlocks(self, dropoutP):
        block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding = (0, self.C1//2), stride=1, bias =False),
                nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
                Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                     padding = (0, 0), bias = False, max_norm = 1, stride=1,
                                     groups=self.F1),
                nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
                nn.ELU(),
                nn.AvgPool2d((1,4*2)),
                nn.Dropout(p = dropoutP))
        block2 = nn.Sequential(
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 16),
                                     padding = (0, 16//2) , bias = False,
                                     groups=self.F1* self.D),
                nn.Conv2d(self.F1 * self.D, self.F2, (1,1),
                          stride =1, bias = False, padding = (0, 0)),
                nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
                nn.ELU(),
                nn.AvgPool2d((1,8*2)),
                nn.Dropout(p = dropoutP)
                )
        return nn.Sequential(block1, block2)

    def lastBlock(self, inF, outF, kernalSize):
        return nn.Sequential(
                nn.Conv2d(inF, outF, kernalSize),
                nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def forward(self, x, eps=0):
        if eps == 0:
            Z = self.firstBlocks(x)
            x = self.lastLayer(Z)
            x = torch.squeeze(x, 3)
            out = torch.squeeze(x, 2)
            return out, [Z]
        else:
            self.universal_nosie.data = torch.clamp(self.universal_nosie.data, -eps, eps)
            x = x + self.universal_nosie
            x = self.firstBlocks(x)
            x = self.lastLayer(x)
            x = torch.squeeze(x, 3)
            out = torch.squeeze(x, 2)
            return out, self.universal_nosie.data

class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class StdLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim = self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim=True)

class MaxLayer(nn.Module):
    '''
    The max layer: calculates the max of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma ,ima = x.max(dim = self.dim, keepdim=True)
        return ma

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
    
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class Expression(nn.Module):
    """Compute given expression on forward pass.
    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__ +
            "(expression=%s) " % expression_str
        )