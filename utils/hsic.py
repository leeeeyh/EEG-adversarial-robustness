import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

def hsic_objective(hidden, h_target, h_data, sigma=0., k_type_y='gaussian'):

    use_cuda = False
    # if h_data.shape[0] == 64:
    #     use_cuda = True
    # else:
    #     use_cuda = False
    
    hsic_hy_val = hsic_normalized_cca( hidden, h_target, sigma=sigma, k_type_y=k_type_y, use_cuda=use_cuda)
    hsic_hx_val = hsic_normalized_cca( hidden, h_data,   sigma=sigma, use_cuda=use_cuda)
        
    # hsic_hx_val = hsic_hx_val/h_data.shape[0]
    # hsic_hy_val = hsic_hy_val/h_data.shape[0]

    # print(hsic_hx_val.item(), hsic_hy_val.item())

    return hsic_hx_val, hsic_hy_val


def hsic_normalized_cca(x, y, sigma, use_cuda=False, to_numpy=True, k_type_y='gaussian'):
    """
    cpu上的结果比cuda结果要好
    """
    m = int(x.size()[0])
    m = x.shape[0]

    Kxc = kernelmat(x, sigma=sigma, use_cuda=use_cuda)
    Kyc = kernelmat(y, sigma=sigma, k_type=k_type_y, use_cuda=use_cuda)

    epsilon = 1E-5
    K_I = torch.eye(m)

    if use_cuda:
        K_I = K_I.cuda()

    # Kxc_i = torch.inverse(Kxc + epsilon*m*K_I)
    # Kyc_i = torch.inverse(Kyc + epsilon*m*K_I)
    Kxc_i = torch.linalg.inv(Kxc + epsilon*m*K_I)
    Kyc_i = torch.linalg.inv(Kyc + epsilon*m*K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))
    return Pxy


def kernelmat(X, sigma, use_cuda=True, k_type="gaussian"):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    m = X.shape[0]
    dim = int(X.size()[1]) * 1.0
    H = (torch.eye(m) - (1./m) * torch.ones([m,m]))
    if use_cuda:
        H = H.cuda()

    if k_type == "gaussian":
        Dxx = distmat(X)
        if sigma:
            variance = 2.*sigma*sigma*X.size()[1]            
            Kx = torch.exp(-Dxx / variance)   # kernel matrices        
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X,X)
                Kx = torch.exp(-Dxx / (2.*sx*sx))

            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))

    ## Adding linear kernel
    elif k_type == "linear":
        Kx = torch.mm(X, X.T)

    if not use_cuda:
        Kx = Kx.type(torch.FloatTensor)
    
    Kxc = torch.mm(Kx,H)

    return Kxc
    
# def sigma_estimation(X, Y):
#     """ sigma from median distance
#     """
#     D = distmat(torch.cat([X,Y]))
#     D = D.detach().cpu().numpy()
#     Itri = np.tril_indices(D.shape[0], -1)
#     Tri = D[Itri]
#     med = np.median(Tri)
#     if med <= 0:
#         med=np.mean(Tri)
#     if med<1E-2:
#         med=1E-2
#     return med

def sigma_estimation(x,y):
        X_numpy = x.cpu().detach().numpy()
        k = squareform(pdist(X_numpy, 'euclidean'))       
        sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))  
        return sigma 

def distmat(X):
    """ distance matrix
    """
    r = torch.sum(X*X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X,0,1))
    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
    D = torch.abs(D)
    return D


def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp( -X / (2.*sigma*sigma))
    return torch.mean(X)

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def mmd(x, y, sigma=None, use_cuda=True, to_numpy=False):
    m = int(x.size()[0])
    H = torch.eye(m) - (1./m) * torch.ones([m,m])
    # H = Variable(H)
    Dxx = distmat(x)
    Dyy = distmat(y)

    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
        sxy = sigma
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    # Kxc = torch.mm(Kx,H)            # centered kernel matrices
    # Kyc = torch.mm(Ky,H)
    Dxy = distmat(torch.cat([x,y]))
    Dxy = Dxy[:x.size()[0], x.size()[0]:]
    Kxy = torch.exp( -Dxy / (1.*sxy*sxy))

    mmdval = torch.mean(Kx) + torch.mean(Ky) - 2*torch.mean(Kxy)

    return mmdval

def mmd_pxpy_pxy(x,y,sigma=None,use_cuda=True, to_numpy=False):
    """
    """
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
    m = int(x.size()[0])

    Dxx = distmat(x)
    Dyy = distmat(y)
    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    A = torch.mean(Kx*Ky)
    B = torch.mean(torch.mean(Kx,dim=0)*torch.mean(Ky, dim=0))
    C = torch.mean(Kx)*torch.mean(Ky)
    mmd_pxpy_pxy_val = A - 2*B + C 
    return mmd_pxpy_pxy_val

def hsic_regular(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """
    """
    Kxc = kernelmat(x, sigma, use_cuda)
    Kyc = kernelmat(y, sigma, use_cuda)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.squeeze(torch.eye(num_classes)[y])


def hsic_normalized(x, y, sigma=None, use_cuda=True, to_numpy=True):
    """
    """
    m = int(x.size()[0])
    Pxy = hsic_regular(x, y, sigma, use_cuda)
    Px = torch.sqrt(hsic_regular(x, x, sigma, use_cuda))
    Py = torch.sqrt(hsic_regular(y, y, sigma, use_cuda))
    thehsic = Pxy/(Px*Py)
    return thehsic


def gussian_kernel(x,y):
    sigma = sigma_estimation(x,y)
    m = x.shape[0]
    vec_dim = x.shape[1]
    x = x.view(m, 1, vec_dim)
    y = y.view(m, vec_dim)
    z = (x - y).float()
    
    return torch.exp((-1 / (2 * (sigma ** 2))) * (torch.norm(z, dim=2) ** 2))

def hsic(x,y,device):
    x, y = x.to(device), y.to(device)
    m = x.shape[0]
    H = torch.eye(m, m) - (1 / m) * torch.ones(m, m)
    H = H.to(device)

    K_x = gussian_kernel(x,x)
    K_y = gussian_kernel(y,y)

    matrix_x = torch.mm(K_x, H)
    matrix_y = torch.mm(K_y, H)
    return (1 / (m - 1)) * torch.trace(torch.mm(matrix_x, matrix_y))