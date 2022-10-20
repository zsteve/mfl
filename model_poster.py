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
dim = 1 # dimension of simulation
sim_steps = 1_000 # number of steps to use for Euler-Maruyama method
T = 10 # number of timepoints
D = 4.0 # diffusivity
t_final = 1.0 # simulation run on [0, t_final]

# setup potential function
def Psi(x, t, dim = dim):
    return 10*(np.cos(x) - np.abs(x))

# get gradient 
dPsi = autograd.elementwise_grad(Psi)

# function for particle initialisation
ic_func = lambda N, d: np.random.randn(N, d)

branching_rate = lambda x: 0*x[:, 0]