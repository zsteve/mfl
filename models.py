import torch
from torch.autograd import Variable
import numpy as np
import sklearn
from sklearn import neighbors
import scipy
from scipy import special
import math

# some utilities 
def round_coupling(x, p, q):
    # Alg. 2 of https://arxiv.org/pdf/1705.09634.pdf
    tmp1 = (torch.minimum(p / x.sum(1), torch.ones_like(p))).reshape(-1, 1) * x * (torch.minimum(q / x.sum(0), torch.ones_like(q))).reshape(1, -1)
    err1 = p - tmp1.sum(1)
    err2 = q - tmp1.sum(0)
    return tmp1 + err1.view(-1, 1)*err2.view(1, -1)/torch.norm(err1, 1)

def H(p, q):
    # H(p, q) = <p, log(p/q) - 1> 
    return torch.xlogy(p, p).sum() - torch.xlogy(p, q).sum() - p.sum()

def H_logdomain(x, logK):
    return torch.xlogy(x, x).sum() - (x * logK).sum() - x.sum()

def entropy_est_knn(x, d, k = 2):
    try:
        nb = sklearn.neighbors.NearestNeighbors(n_neighbors = k).fit(x)
        dist, ind = nb.kneighbors(x)
    except:
        return float("NaN")
    eps = dist[:, -1]
    C = np.pi**(d/2) / scipy.special.gamma(d/2 + 1)
    return -(scipy.special.psi(x.shape[0]) - scipy.special.psi(k-1) + np.log(C) + d*np.log(eps).mean())

def proxdiv_balanced(p, s, u, eps, lamda):
    return p/s

def proxdiv_unbalanced(p, s, u, eps, lamda):
    return (p/s)**(lamda/(lamda+eps)) * (u/(lamda+eps)).exp()

class SinkhornLoss(torch.nn.Module):
    def updateC(self, x_spt, y_spt):
        # Update cost matrix to be C_{ij} = |x_i-y_j|/(2s^2)
        # where s is a user-specified scale factor
        self.C = torch.cdist(x_spt, y_spt, p = 2)**2 / (2 * self.scale_factor**2)
    def getK(self, C, eps, x_w, y_w, a, b):
        # Get (weighted, normalized) Gibbs kernel corresponding to the formula
        # Z^{-1} exp(-C'_{ij}/eps) * dx_i * dy_j
        # where C'_{ij} = C_{ij} + a_i + b_j is the cost matrix with log-domain stabilization factors as per Schmitzer et al. (https://arxiv.org/abs/1610.06519)
        return torch.exp(-(C + a.view(-1, 1) + b.view(1, -1))/eps - self.logZ) * x_w.view(-1, 1) * y_w.view(1, -1)
    def updateReg(self, eps, flush = False):
        # update regularization level in-place, and recompute the Gibbs kernel.
        # optionally, can reset all dual variables. 
        self.eps = eps
        if flush:
            self.u = torch.ones_like(self.x_w)
            self.v = torch.ones_like(self.y_w)
            self.a = torch.zeros_like(self.u)
            self.b = torch.zeros_like(self.v)
        # cost matrix and Gibbs kernel
        self.K = self.getK(self.C, self.eps, self.x_w, self.y_w, self.a, self.b)
    def __init__(self, x_w, x_spt, y_w, y_spt, eps, n_iter, scale_factor, lamda1 = None, lamda2 = None, warm_start = False, atol = 1e-4, rtol = 1e-4):
        super(SinkhornLoss, self).__init__()
        self.n_iter = n_iter
        self.x_w = x_w
        self.x_spt = x_spt
        self.y_w = y_w
        self.y_spt = y_spt
        self.d = x_spt.shape[1] # dimension
        self.logZ = math.log((2*math.pi*eps)**(self.d/2)) # norm factor
        self.warm_start = warm_start
        self.atol = atol
        self.rtol = rtol
        # proxdiv operators (as per Chizat et al. https://arxiv.org/abs/1607.05816) for (un)balanced transport
        self.proxdiv_F1 = proxdiv_balanced if lamda1 is None else proxdiv_unbalanced
        self.proxdiv_F2 = proxdiv_balanced if lamda2 is None else proxdiv_unbalanced
        # unbalanced penalties (None if balanced)
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.iters_used = -1
        self.scale_factor = scale_factor # scale factor, has same [length] units
        # dual potentials
        # self.eps = eps
        # self.u = torch.ones_like(self.x_w)
        # self.v = torch.ones_like(self.y_w)
        # self.a = torch.zeros_like(self.u)
        # self.b = torch.zeros_like(self.v)
        self.updateC(self.x_spt, self.y_spt)
        self.updateReg(eps, flush = True)
        
    def sinkhorn(self, C, eps, lamda1, lamda2, x_w, y_w, u, v, a, b, n_iter, thresh = 100):
        # this should be called with torch.no_grad()
        K = self.getK(C, eps, x_w, y_w, a, b)
        u_prev = u
        for i in range(n_iter):
            v.copy_(self.proxdiv_F2(y_w, K.T @ u, b, eps, lamda1))
            u.copy_(self.proxdiv_F1(x_w, K @ v, a, eps, lamda2))
            self.iters_used = i
            if lamda1 is None and lamda2 is None:
                # balanced case, check marginal constraints
                err = torch.norm(v * (K.T @ u) - y_w, float('inf'))
                if (err < self.atol) or (err/torch.norm(y_w, float('inf')) < self.rtol):
                    break
            else:
                # unbalanced case
                err = torch.norm(u - u_prev, float('inf'))
                if (err < self.atol) or (err/torch.norm(u, float('inf')) < self.rtol):
                    break
                u_prev.copy_(u)
            if torch.maximum(torch.norm(u, float('inf')), torch.norm(v, float('inf'))) > thresh:
                # absorption step
                a += -eps*torch.log(u)
                b += -eps*torch.log(v)
                u.fill_(1)
                v.fill_(1) 
                K.copy_(self.getK(C, eps, x_w, y_w, a, b))
        return (u, v, a, b, K)
    
    def get_coupling(self, K, x_w, y_w, u, v):
        return K * self.u.view(-1, 1) * self.v.view(1, -1)
    
    def coupling(self):
        return self.get_coupling(self.K, self.x_w, self.y_w, self.u, self.v)
    
    def obj_dual(self, eps, lamda1, lamda2, C, x_w, y_w, u, v, a, b):
        # return dual loss at current iterate
        # assuming we are at convergence, then this will be the same as the primal loss
        def F_indicator_star(u, p):
            # dual of q -> {q == p}
            return u.dot(p)
        def F_kl_star(u, p):
            # dual of q -> KL(q|p)
            return p.dot(u.exp()-1)
        f = -eps*u.log() + a
        g = -eps*v.log() + b
        f_cts = eps*torch.logsumexp((-C - g.view(1, -1))/eps + y_w.log().view(1, -1), 1) - eps*self.logZ
        g_cts = eps*torch.logsumexp((-C - f.view(-1, 1))/eps + x_w.log().view(-1, 1), 0) - eps*self.logZ
        f1 = F_indicator_star(f_cts, x_w) if lamda1 is None else lamda1*F_kl_star(f_cts/lamda1, x_w)
        f2 = F_indicator_star(g_cts, y_w) if lamda2 is None else lamda2*F_kl_star(g_cts/lamda2, y_w)
        # return -f1 - f2 - eps*torch.dot(torch.exp(-f_cts/eps) * self.x_w, torch.exp(-C/eps - self.logZ) @ (torch.exp(-g_cts/eps) * self.y_w))
        return -f1 - f2 - eps*torch.sum(torch.exp(-(C + f_cts.view(-1, 1) + g_cts.view(1, -1))/eps - self.logZ) * self.x_w.view(-1, 1) * self.y_w.view(1, -1))
    
    def obj_primal(self, eps, C, K, x_w, y_w, u, v):
        # return primal loss at a suboptimal point
        gamma = torch.relu(round_coupling(self.get_coupling(K, x_w, y_w, u, v), x_w, y_w))
        return eps * H_logdomain(gamma, (-C/eps) + torch.log(x_w).view(-1, 1) + torch.log(y_w).view(1, -1))
    
    def forward(self):
        # carry out Sinkhorn iterations and return dual loss
        self.updateC(self.x_spt, self.y_spt)
        if (not self.warm_start):
            self.u = torch.ones_like(self.x_w)
            self.v = torch.ones_like(self.y_w)
            self.a = torch.zeros_like(self.x_w)
            self.b = torch.zeros_like(self.y_w)
        else:
            # need to clear grad information from previous forward iters, if any [no longer necessary]
            # with torch.no_grad():
            #     self.u = self.u.clone()
            #     self.v = self.v.clone()
            #     self.a = self.a.clone()
            #     self.b = self.b.clone()
            pass
        with torch.no_grad():
            self.u, self.v, self.a, self.b, self.K = self.sinkhorn(self.C, self.eps, self.lamda1, self.lamda2, self.x_w, self.y_w, self.u, self.v, self.a, self.b, self.n_iter)
        return self.obj_dual(self.eps, self.lamda1, self.lamda2, self.C, self.x_w, self.y_w, self.u, self.v, self.a, self.b)

class PathsLoss(torch.nn.Module):
    def __init__(self, x, tau, dt, sinkhorn_iters, device, scale_factors, lamda_unbal = None, warm_start = False, branching_rate_fn = None):
        super(PathsLoss, self).__init__()
        self.device = device
        self.x = x # starting empirical marginals, dimensions (T, N, d)
        self.T = x.shape[0] # number of timepoints
        self.N = x.shape[1] # number of particles per timepoint
        self.d = x.shape[2] # dimension
        self.tau = tau # diffusivity 
        self.dt = dt # timestep
        self.lamda_unbal = lamda_unbal # unbalanced penalty
        self.branching_rate_fn = branching_rate_fn # branching rate function
        self.warm_start = warm_start 
        self.scale_factors = scale_factors
        self.sinkhorn_iters = sinkhorn_iters
        self.ot_losses = torch.nn.ModuleList([SinkhornLoss(torch.full((self.N, ), 1/self.N, device = self.device), self.x[i, :, :],
                                                           torch.full((self.N, ), 1/self.N, device = self.device), self.x[i+1, :, :], 
                                                           self.tau * dt,
                                                           self.sinkhorn_iters,
                                                           self.scale_factors[i], 
                                                           warm_start = self.warm_start, 
                                                           lamda1 = self.lamda_unbal, lamda2 = self.lamda_unbal) for i in range(self.T-1)])
    def update_tau(self, tau):
        # iteratively update the tau parameter for each Sinkhorn term, following eps = tau * dt
        self.tau = tau
        for (i, loss) in enumerate(self.ot_losses):
            loss.updateReg(tau*self.dt)
    def update_weights(self):
        # update weights if we want to incorporate branching
        normalize = lambda x : x / x.sum()
        for loss in self.ot_losses:
            loss.x_w = normalize((self.branching_rate_fn(loss.x_spt)*self.dt/2).exp())
            loss.y_w = normalize((-self.branching_rate_fn(loss.y_spt)*self.dt/2).exp())
    def forward(self):
        if self.branching_rate_fn is not None:
            # assign first weights 
            with torch.no_grad():
                self.update_weights()
        return sum([loss.forward()/self.dt for loss in self.ot_losses])
    def forward_primal(self):
        return sum([loss.obj_primal(loss.eps, loss.C, loss.K, loss.x_w, loss.y_w, loss.u, loss.v)/self.dt for loss in self.ot_losses])

class FitLoss(torch.nn.Module):
    # (x, x_w) must correspond to the model particles (not samples)
    def __init__(self, x, x_w, y, y_w, sigma, scale_factor):
        super(FitLoss, self).__init__()
        self.x = x
        self.x_w = x_w
        self.y = y
        self.y_w = y_w
        self.sigma= sigma # bandwidth
        self.scale_factor = scale_factor # user-specified scale factor, with units [length]
        self.C = None
    def forward(self):
        self.C = (-(torch.cdist(self.x, self.y, p = 2)**2)/(2*(self.scale_factor*self.sigma)**2)) + torch.log(self.x_w).view(-1, 1)
        return -torch.logsumexp(self.C, 0).dot(self.y_w)
    def update_sigma(self, sigma):
        self.sigma = sigma

class TrajLoss(torch.nn.Module):
    def __init__(self, x0, x_obs, t_idx_obs, dt, tau, sigma, M, lamda_reg, lamda_cst, sigma_cst, branching_rate_fn, sinkhorn_iters, device, lamda_unbal = None, warm_start = False, scale_factors = None, scale_factors_fit = None, scale_factor_global = None):
        super(TrajLoss, self).__init__()
        self.device = device
        self.T = x0.shape[0] # number of timepoints
        self.d = x0.shape[2] # dimensions
        self.N = torch.unique(t_idx_obs, return_counts = True)[1] # number of sample points (per timepoint)
        self.M = M  # particles per timepoint
        self.tau = tau # diffusivity 
        self.sigma = sigma # data-fitting bandwidth
        self.lamda_reg = lamda_reg # regularization parameter
        self.lamda_cst = lamda_cst # constraining parameter 
        self.lamda_unbal = lamda_unbal # unbalanced parameter
        self.sigma_cst = sigma_cst # constraining bandwidth
        self.branching_rate_fn = branching_rate_fn
        self.dt = dt
        self.x = torch.nn.Parameter(Variable(x0, requires_grad = True))
        self.x_obs = x_obs 
        self.t_idx_obs = t_idx_obs
        self.w = self.t_idx_obs.unique(return_counts = True)[1]
        self.w = self.w/self.w.sum() # compute time-point weights
        self.warm_start = warm_start
        self.lamda_unbal = lamda_unbal
        self.scale_factors = torch.ones(self.T-1) if scale_factors is None else scale_factors
        self.scale_factors_fit = torch.ones(self.T) if scale_factors_fit is None else scale_factors_fit
        self.scale_factor_global = 1.0 if scale_factor_global is None else scale_factor_global
        self.sinkhorn_iters = sinkhorn_iters
        self.loss_reg = PathsLoss(self.x, 
                                  self.tau, 
                                  self.dt, 
                                  self.sinkhorn_iters, 
                                  self.device, 
                                  self.scale_factors, 
                                  warm_start = self.warm_start, 
                                  lamda_unbal = self.lamda_unbal,
                                  branching_rate_fn = self.branching_rate_fn)
        self.loss_fit = torch.nn.ModuleList([FitLoss(self.x[i, :, :], torch.full((self.M, ), 1/self.M, device = self.device), 
                                                     self.x_obs[self.t_idx_obs == i, :], torch.full((self.N[i], ), 1/self.N[i], device = self.device),
                                                     self.sigma,
                                                     self.scale_factors_fit[i]) for i in range(self.T)])
        self.loss_cst = ConstrainLoss(self.x, torch.full((self.M * self.T, ), 1/(self.M * self.T), device = self.device), 
                                      self.x_obs, torch.full((self.N.sum(), ), 1/(self.N.sum()), device = self.device), 
                                      self.sigma_cst, 
                                      self.scale_factor_global,
                                      self.device)
    def forward(self):
        return self.lamda_reg*self.loss_reg.forward() + \
                sum([self.w[i] * loss.forward() for (i, loss) in enumerate(self.loss_fit)]) + \
                self.lamda_cst * self.loss_cst.forward()
    def forward_primal(self):
        return self.lamda_reg*self.loss_reg.forward_primal() + \
                sum([self.w[i] * loss.forward() for (i, loss) in enumerate(self.loss_fit)]) + \
                self.lamda_cst * self.loss_cst.forward()
    def update_tau(self, tau):
        self.tau = tau
        self.loss_reg.update_tau(self.tau)
    def update_sigma(self, sigma):
        self.sigma = sigma
        for loss in self.loss_fit:
            loss.update_sigma(sigma)

class ConstrainLoss(torch.nn.Module):
    def __init__(self, x, x_w, x_obs, x_obs_w, sigma, scale_factor, device):
        super(ConstrainLoss, self).__init__()
        self.device = device
        self.x = x
        self.x_w = x_w
        self.x_obs = x_obs
        self.x_obs_w = x_obs_w
        self.d = x_obs.shape[-1]
        self.sigma = sigma
        self.C = None
        self.scale_factor = scale_factor
    def forward(self):
        self.C = -(torch.cdist(self.x.view(-1, self.d), self.x_obs.view(-1, self.d), p = 2)**2)/(2*(self.scale_factor*self.sigma)**2) + torch.log(self.x_obs_w).view(1, -1)
        return -torch.logsumexp(self.C, 1).dot(self.x_w)

class LangevinGD(torch.optim.Optimizer):
    # Langevin dynamics with diffusivity sigma2 and step-size eta, following 
    # dX_t = -dV[p_t](X_t) dt + sigma dB_t
    def __init__(self, params, eta, sigma2, N, tloss):
        defaults = dict(eta=eta, sigma2=sigma2, N=N)
        super(LangevinGD, self).__init__(params, defaults)
        self.tloss = tloss
    def __setstate__(self, state):
        super(LangevinGD, self).__setstate__(state)
    def update_sigma2(self, sigma2):
        self.param_groups[0]['sigma2'] = sigma2
    def update_eta(self, eta):
        self.param_groups[0]['eta'] = eta
    def step(self):
        assert len(self.param_groups) == 1
        for group in self.param_groups:
            for p in group["params"]:
                # need to save x if we want to update u 
                if p.grad is None:
                    continue
                eta = group["eta"] * self.tloss.scale_factor_global
                sigma2 = group["sigma2"]
                N = group["N"]
                grad = p.grad.data
                p.data.add_(torch.randn_like(grad), alpha = (sigma2*eta)**0.5)
                p.data.add_(-N*grad, alpha = eta)

def optimize(model, n_iter, tau_final, eta_final, sigma_final, temp_init, temp_ratio, dim, print_interval = 50, **kwargs):
    obj = []
    obj_primal = []
    temp_curr = temp_init
    tau_t = tau_final*temp_init  # diffusivity
    eta_t = eta_final*temp_init  # step size
    sigma_t = sigma_final*temp_init**0.5 # data-fitting bandwidth
    optim = LangevinGD(model.parameters(), eta = eta_t, sigma2 = 2*tau_t*model.lamda_reg, **kwargs) 
    # save all iterates for animations
    x_save = torch.zeros((n_iter, ) + tuple(model.x.shape))
    for i in range(n_iter):
        with torch.no_grad():
            x_save[i, :, :, :] = model.x.data.clone()
        ## set noise level
        model.update_tau(tau_t)
        model.update_sigma(sigma_t)
        optim.update_sigma2(2*tau_t*model.lamda_reg)
        optim.update_eta(eta_t)
        ##  optimize whole model
        loss = model()
        if torch.isnan(loss):
            break
        with torch.no_grad():
            # compute the primal objective before doing the step, since positions will be updated.
            x = model.x.cpu().numpy()
            loss_primal = model.forward_primal() + tau_t*model.lamda_reg*sum([entropy_est_knn(x[i, :, :], d = dim, k = 2) for i in range(x.shape[0])])
        ## langevin step
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        with torch.no_grad():
            obj_primal.append(loss_primal.item())
            obj.append(loss.item())
        
        if i % print_interval == 0:
            avg_iters = np.array([l.iters_used for l in model.loss_reg.ot_losses]).mean()
            print("Iteration %d, Loss = %0.3f, Primal loss = %0.3f, Avg. iters = %0.3f, eta = %0.3f, temp = %0.3f" % (i, loss, loss_primal, avg_iters, eta_t, max(1, temp_curr)))
            
        # update noise level
        temp_curr *= temp_ratio
        tau_t = tau_final*max(1, temp_curr)
        sigma_t = sigma_final*max(1, temp_curr**0.5)
        eta_t = eta_final*max(1, temp_curr)
        
    return obj, obj_primal, x_save
