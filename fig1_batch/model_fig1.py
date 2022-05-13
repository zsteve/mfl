import torch
from torch.autograd import grad, Variable
import autograd
import autograd.numpy as np
import copy
import scipy as sp
from scipy import stats
from sklearn import metrics
import gwot
from gwot import models, sim, ts, util
import gwot.bridgesampling as bs
import random

# setup simulation parameters
dim = 10 # dimension of simulation
sim_steps = 1_000 # number of steps to use for Euler-Maruyama method
T = 10 # number of timepoints
D = 0.25 # diffusivity
t_final = 1.25 # simulation run on [0, t_final]

# setup potential function
def Psi(x, t, dim = dim):
    x0 = np.array([1.5, ] + [0, ]*(dim - 1))
    x1 = -np.array([1.5,] + [0, ]*(dim - 1))
    # return 1.25*np.sum((x - x0)*(x - x0), axis = -1) * np.sum((x - x1)*(x - x1), axis = -1) + 10*np.sum(x[:, 2:]*x[:, 2:], axis = -1)
    return 0.5*np.sum(((x - x0)*(x - x0))[:, 0:1], axis = -1) * np.sum(((x - x1)*(x - x1))[:, 0:1], axis = -1) + 10*np.sum((x[:, 1:2] + t)*(x[:, 1:2] + t), axis = -1) + 10*np.sum(x[:, 2:]*x[:, 2:], axis = -1)
# get gradient 
dPsi = autograd.elementwise_grad(Psi)

# function for particle initialisation
ic_func = lambda N, d: np.random.randn(N, d)*0.1

branching_rate = lambda x: 0*x[:, 0]