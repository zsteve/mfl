import argparse 
parser = argparse.ArgumentParser(description = "")
parser.add_argument("--t_initial", type = float, default = 2.5)
parser.add_argument("--t_final", type = float, default = 6.5)
parser.add_argument("--eps_eff", type = float, default = 0.1)
parser.add_argument("--N", type = int, default = 10)
parser.add_argument("--gwot", action = "store_true")
parser.add_argument("--N0", type = int, default = 100)
parser.add_argument("--M", type = int, default = 500)
parser.add_argument("--sample_points_gwot", type = int, default = 500)
parser.add_argument("--lamda", type = float, default = 0.025)
parser.add_argument("--lamda_gwot", type = float, default = 0.0025)
parser.add_argument("--sigma", type = float, default = 0.5)
parser.add_argument("--eta", type = float, default = 0.1)
parser.add_argument("--n_iter", type = int, default = 2500)
parser.add_argument("--srand", type = int, default = 0)
parser.add_argument("--outfile", type = str, default = "out.npy")
parser.add_argument("--threads", type = int, default = 8)
parser.add_argument("--adata_path", type = str, default = "data_repr.h5ad")

args = parser.parse_args()

# fix number of threads
import os
num_threads = "%d" % args.threads
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
import sys
sys.path.append("..")

import torch
import anndata
import numpy as np
import pegasus as pg
import matplotlib.pyplot as plt
import gwot
from gwot import models, util, ts
import gwot.reprog_dataset_utils as reprog_utils
import ot
import dcor 
import sklearn
from sklearn import decomposition, preprocessing
import models

torch.manual_seed(args.srand)
np.random.seed(args.srand)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)

adata = anndata.read_h5ad(args.adata_path)
adata_ts = adata[~np.isnan(adata.obs.day), :] 
t_map = np.array(adata_ts.obs.day.unique())
t_map = t_map[(t_map >= args.t_initial) & (t_map < args.t_final)]
numcells = np.array([args.N0, ] + [args.N, ]*(len(t_map)-2) + [args.N0, ])

adata_subsamp = [reprog_utils.subsamp(adata_ts, t, numcells[i]) for (i, t) in enumerate(t_map)]
adata_s = adata_subsamp[0].concatenate(adata_subsamp[1:])
adata_s.obsm["X_pca_orig"] = adata_s.obsm["X_pca"]
days = adata_s.obs.day.unique()
dt = np.array([t_map[i+1] - t_map[i] for i in range(0, len(t_map)-1)])
del adata_s.obsm["X_pca"] # delete first X_pca because scanpy might not recompute

norm_op = preprocessing.StandardScaler(with_std = False)
pca_op = decomposition.PCA(n_components = adata_s.obsm["X_pca_orig"].shape[1])
adata_s.obsm["X_pca"] = pca_op.fit_transform(norm_op.fit_transform(adata_s.X.todense()))
pg.neighbors(adata_s)
pg.diffmap(adata_s)
adata_s.obsm['X_fle'] = np.array(adata_s.obsm['X_fle'])

# transform ground truth into the same coordinates
adata_gt = adata_ts[(adata_ts.obs.day >= args.t_initial) & (adata_ts.obs.day < args.t_final)];
X_gt = pca_op.transform(norm_op.transform(adata_gt.X.todense()))

idx = [np.where(adata_s.obs.day == t)[0] for t in t_map]
t_idx = np.zeros(adata_s.shape[0], dtype = np.int64)
for i in range(0, len(idx)):
    t_idx[idx[i]] = i

if args.gwot is False:
    tsdata = gwot.ts.TimeSeries(x = np.array(adata_s.obsm["X_pca"], dtype = np.float64), 
                    dt = dt/dt.sum(), 
                    t_idx = t_idx, 
                    D = args.eps_eff/(dt[0]/dt.sum()))
    R = torch.tensor(np.log(adata_s.obs.cell_growth_rate))/tsdata.dt[1]
    # compute scale factors
    scale_factors_tr = np.array([reprog_utils.get_C_mean(adata_s, t_map[i], t_next = t_map[i+1], mode = "tr")/2 for i in range(0, len(t_map[:-1]))])**0.5
    scale_factors_fit  = np.array([reprog_utils.get_C_mean(adata_s, t, mode = "self")/2 for t in t_map])**0.5
    adata_temp = adata_s.copy()
    adata_temp.obs.day = 0
    scale_factor_global = (reprog_utils.get_C_mean(adata_temp, 0, mode = "self")/2)**0.5
    # initial condition
    X_obs = torch.tensor(tsdata.x, device = device) 
    t_idx_obs = torch.tensor(tsdata.t_idx, device = device)
    dim = X_obs.shape[1]
    X0 = torch.stack([X_obs[tsdata.t_idx == i, :][np.random.choice((tsdata.t_idx == i).sum(), size = args.M), :] for i in range(tsdata.T)])
    # growth rate interpolation
    def R_func(x, h = 0.25*scale_factor_global):
        w = torch.softmax(-torch.cdist(x, X_obs, p = 2)**2/h**2, 1)
        return (w * R.reshape(1, -1)).sum(1)
    # fit model
    model = models.TrajLoss(X0, X_obs, t_idx_obs, dt = dt[0]/dt.sum(), tau = tsdata.D, 
                sigma = None, M = args.M, lamda_reg = args.lamda, lamda_cst = 0.0, sigma_cst = float("Inf"),
                branching_rate_fn = R_func,
                sinkhorn_iters = 250, device = device, warm_start = True, 
                lamda_unbal = None, 
                scale_factors = scale_factors_tr, scale_factors_fit = scale_factors_fit, scale_factor_global = scale_factor_global)
    output = models.optimize(model, n_iter = args.n_iter, eta_final = args.eta, tau_final = tsdata.D, sigma_final = args.sigma, temp_init = 1.0, temp_ratio = 1.0, N = args.M, dim = dim, tloss = model, print_interval = 25);
    with torch.no_grad():
        np.save(args.outfile, {"args" : args, "model_x" : model.x, "X_gt" : X_gt, "day_gt" : adata_gt.obs.day, "tsdata" : tsdata})
else:
    scale_factors_tr = np.array([reprog_utils.get_C_mean(adata_s, t_map[i], t_next = t_map[i+1], mode = "tr") for i in range(0, len(t_map[:-1]))])
    scale_factors_fit = np.array([reprog_utils.get_C_mean(adata_s, t, mode = "self") for t in t_map])
    # 
    idx = [np.where(adata_s.obs.day == t)[0] for t in t_map]
    t_idx = np.zeros(adata_s.shape[0], dtype = np.int64)
    for i in range(0, len(idx)):
        t_idx[idx[i]] = i
    tsdata = gwot.ts.TimeSeries(x = np.array(adata_s.obsm["X_pca"], dtype = np.float64), 
                    dt = dt/dt.sum(), 
                    t_idx = t_idx, 
                    D = args.eps_eff/(2*dt[0]/dt.sum()))
    # fit no-growth model
    model_ng = gwot.models.OTModel(tsdata, lamda_reg = args.lamda_gwot,
            eps_df = 0.025*torch.ones(tsdata.T, device = device), 
            kappa = torch.from_numpy(25/tsdata.dt).to(device), 
            c_scale = torch.from_numpy(scale_factors_tr).to(device),
            c_scale_df = torch.from_numpy(scale_factors_fit).to(device),
            growth_constraint="KL",
            pi_0 = "uniform", 
            device = device,
            use_keops = False)
    model_ng.solve_lbfgs(steps = 10, max_iter = 50, lr = 1, history_size = 50, line_search_fn = 'strong_wolfe', factor = 2, tol = 1e-5, retry_max = 0)
    with torch.no_grad():
        R_ng = model_ng.get_R()
    r_ng = (R_ng.T/R_ng.sum(dim = 1)).T
    growth_rate = torch.from_numpy(np.array(adata_s.obs.cell_growth_rate)).to(device)
    g = torch.stack([(torch.from_numpy(np.array(adata_s.obs.cell_growth_rate)).to(device))**(t_map[i+1]-t_map[i]) for i in range(0, tsdata.T-1)]).to(device)
    r = torch.stack([(r_ng[i, :]*(growth_rate**(t_map[i+1]-t_map[i]))).sum() for i in range(0, model_ng.ts.T-1)])
    m = torch.cumprod(torch.cat([torch.tensor([1., ]).to(device), r]), dim = 0)

    model_g = gwot.models.OTModel(tsdata, lamda_reg = args.lamda_gwot,
            eps_df = 0.025*torch.ones(tsdata.T).to(device), 
            m = m.to(device),
            g = g.to(device),
            kappa = torch.from_numpy(25/tsdata.dt).to(device), 
            c_scale = torch.from_numpy(scale_factors_tr).to(device),
            c_scale_df = torch.from_numpy(scale_factors_fit).to(device),
            growth_constraint="KL",
            pi_0 = "uniform", 
            device = device,
            use_keops = False)
    model_g.solve_lbfgs(steps = 25, max_iter = 50, lr = 0.1, history_size = 50, line_search_fn = 'strong_wolfe', factor = 2, tol = 1e-5, retry_max = 0)
    X_gwot = torch.tensor(np.stack([model_g.ts.x[np.random.choice(model_g.ts.x.shape[0], size = args.sample_points_gwot, p = torch.nn.functional.normalize(model_g.get_R(t).detach(), p = 1.0, dim = 0)), :] for t in range(tsdata.T)]))
    with torch.no_grad():
        np.save(args.outfile, {"args" : args, "samples_gwot" : X_gwot, "X_gt" : X_gt, "day_gt" : adata_gt.obs.day, "tsdata" : tsdata})
