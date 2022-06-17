from scipy.spatial.distance import pdist, squareform
import numpy as np
import torch

class DIB():
    def __init__(self):
        self.alpha = 1.01
        
    def pairwise_distances(self, x):
        instances_norm = torch.sum(x**2,-1).reshape((-1,1))
        return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()
    
    def calculate_gram_mat(self, x, sigma):
        dist= self.pairwise_distances(x)
        return torch.exp(-dist /sigma)
    
    def reyi_entropy(self, x, sigma):
        k = self.calculate_gram_mat(x, sigma)
        k = k/torch.trace(k) 
        # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
        eigv = torch.abs(torch.linalg.eigh(k, UPLO='U')[0])
        eig_pow = eigv**self.alpha
        entropy = (1/(1-self.alpha))*torch.log2(torch.sum(eig_pow))
        return entropy
    
    def joint_entropy(self, x, y, s_x, s_y):
        x = self.calculate_gram_mat(x,s_x)
        y = self.calculate_gram_mat(y,s_y)
        k = torch.mul(x,y)
        k = k/torch.trace(k)
        # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
        eigv = torch.abs(torch.linalg.eigh(k, UPLO='U')[0])
        eig_pow =  eigv**self.alpha
        entropy = (1/(1-self.alpha)) * torch.log2(torch.sum(eig_pow))
        return entropy
    
    def calculate_MI(self, x, y, s_x, s_y):
        Hx = self.reyi_entropy(x, sigma=s_x)
        Hy = self.reyi_entropy(y, sigma=s_y)
        Hxy = self.joint_entropy(x, y, s_x, s_y)
        Ixy = Hx + Hy - Hxy
        #normlize = Ixy/(torch.max(Hx,Hy))
        return Ixy

    def calculate_sigma(self, x):
        X_numpy = x.cpu().detach().numpy()
        k = squareform(pdist(X_numpy, 'euclidean'))       
        sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))  
        return sigma      
    
    def distmat(self, X):
        """ 
        distance matrix
        """
        r = torch.sum(X*X, 1)
        r = r.view([-1, 1])
        a = torch.mm(X, torch.transpose(X,0,1))
        D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
        D = torch.abs(D)
        return D

    def sigma_estimation(self, X, Y):
        """ sigma from median distance
        """
        D = self.distmat(torch.cat([X,Y]))
        D = D.detach().cpu().numpy()
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            med=np.mean(Tri)
        if med<1E-2:
            med=1E-2
        return med

    def izx_loss(self, x, z):
        #x and z should be two dimensional
        X = x.reshape((x.shape[0], -1))
        Z = z.reshape((z.shape[0], -1))
        sigma_x = self.calculate_sigma(X)      
        sigma_z = self.calculate_sigma(Z) 

        # sigma_x = self.sigma_estimation(X, X)      
        # sigma_z = self.sigma_estimation(Z, Z) 

        # sigma_x = 5.  
        # sigma_z = 5.

        I_ZX_bound = self.calculate_MI(X, Z, sigma_x**2, sigma_z**2)
        return I_ZX_bound

class VIB():
    def __init__(self): 
        self.Z = 0
        self.dimZ = 0
        self.encoder_output = 0
        pass      

    def KL_between_normals(self, q_distr, p_distr):
        mu_q, sigma_q = q_distr
        mu_p, sigma_p = p_distr
        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq =  torch.mul(mu_diff, mu_diff)
        logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
        logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1)  + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
        two_kl =  fs - k + logdet_sigma_p - logdet_sigma_q
        return two_kl * 0.5

    def encoder_result(self):
        encoder_output = self.encoder_output
        mu = encoder_output[:, :self.dimZ]
        sigma = torch.nn.functional.softplus(encoder_output[:, self.dimZ:])
        return mu, sigma 

    def izx_loss(self, Z):
        self.Z = Z.reshape((Z.shape[0],-1))
        self.dimZ = self.Z.shape[1]//2
        self.encoder_output = self.Z
        batch_size = self.Z.shape[0]
        prior_Z_distr = torch.zeros(batch_size, self.dimZ).cuda(), torch.ones(batch_size, self.dimZ).cuda()
        encoder_Z_distr = self.encoder_result()
        I_ZX_bound = torch.mean(self.KL_between_normals(encoder_Z_distr, prior_Z_distr))   
        return I_ZX_bound