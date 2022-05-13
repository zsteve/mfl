import argparse 
parser = argparse.ArgumentParser(description = "")
parser.add_argument("--N", type = int, default = 50)
parser.add_argument("--gwot", action = "store_true")
parser.add_argument("--N0", type = int, default = 64)
parser.add_argument("--M", type = int, default = 100)
parser.add_argument("--sample_points_gwot", type = int, default = 100)
parser.add_argument("--lamda", type = float, default = 0.025)
parser.add_argument("--lamda_gwot", type = float, default = 0.0025)
parser.add_argument("--sigma", type = float, default = 0.5)
parser.add_argument("--eta", type = float, default = 0.1)
parser.add_argument("--n_iter", type = int, default = 2500)
parser.add_argument("--srand", type = int, default = 0)
parser.add_argument("--outfile", type = str, default = "out.npy")
parser.add_argument("--threads", type = int, default = 8)

args = parser.parse_args()

import torch
from torch.autograd import grad, Variable
import autograd
import autograd.numpy as np
import scipy as sp
from scipy import stats
from sklearn import metrics
import sys
import ot
import gwot
from gwot import models, sim, ts, util
import gwot.bridgesampling as bs

sys.path.append("..")
import importlib
import models
import random
import model_fig1 as model_sim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_num_threads(args.threads)

torch.manual_seed(args.srand)
np.random.seed(args.srand)

sim = gwot.sim.Simulation(V = model_sim.Psi, dV = model_sim.dPsi, birth_death = False, 
                          N = np.array([args.N0, ] + [args.N, ]*(model_sim.T-2) + [args.N0, ]),
                          T = model_sim.T, 
                          d = model_sim.dim, 
                          D = model_sim.D, 
                          t_final = model_sim.t_final, 
                          ic_func = model_sim.ic_func, 
                          pool = None)
sim.sample(steps_scale = int(model_sim.sim_steps/sim.T));

if args.gwot == False:
    model = models.TrajLoss(torch.randn(model_sim.T, args.M, model_sim.dim)*0.1,
                            torch.tensor(sim.x, device = device), 
                            torch.tensor(sim.t_idx, device = device), 
                            dt = model_sim.t_final/model_sim.T, tau = model_sim.D, sigma = None, M = args.M,
                            lamda_reg = args.lamda, lamda_cst = 0, sigma_cst = float("Inf"),
                            branching_rate_fn = model_sim.branching_rate,
                            sinkhorn_iters = 250, device = device, warm_start = True)
    output = models.optimize(model, n_iter = args.n_iter, eta_final = args.eta, tau_final = model_sim.D, sigma_final = args.sigma, temp_init = 1.0, temp_ratio = 1.0, N = args.M, dim = model_sim.dim, tloss = model, print_interval = 25)
    with torch.no_grad():
        np.save(args.outfile, {"args" : args, "x" : model.x, "sim_x" : sim.x, "sim_t_idx" : sim.t_idx})
else:
    model_gwot = gwot.models.OTModel(sim, lamda_reg = args.lamda_gwot,
              eps_df = 0.01*torch.ones(sim.T), 
              growth_constraint="exact", 
              pi_0 = "uniform",
              use_keops = False,
              device = device)
    model_gwot.solve_lbfgs(steps = 25, 
                    max_iter = 50, 
                    lr = 1,
                    history_size = 50, 
                    line_search_fn = 'strong_wolfe', 
                    factor = 2, 
                    tol = 1e-5, 
                    retry_max = 0);
    samples_gwot = np.stack([model_gwot.ts.x[np.random.choice(model_gwot.ts.x.shape[0], size = args.sample_points_gwot, p = torch.nn.functional.normalize(model_gwot.get_R(t).detach(), p = 1.0, dim = 0)), :] for t in range(model_sim.T)])
    with torch.no_grad():
        np.save(args.outfile, {"args" : args, "samples_gwot" : samples_gwot, "sim_x" : sim.x, "sim_t_idx" : sim.t_idx})
