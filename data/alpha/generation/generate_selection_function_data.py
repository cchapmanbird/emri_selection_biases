import numpy as np
from functools import partial 
from utils import draw_samples_from_population, construct_cosmo_splines, run_on_dataset, load_mlp
import pandas as pd
import torch
from scipy.stats import ncx2
import time
torch.manual_seed(1729)
np.random.seed(1729)
try:
    import cupy as xp
    xp.cuda.runtime.setDevice(1)
    device = 'cuda:1'
    print('Running on device ', device)
    xp.random.seed(1729)
except ModuleNotFoundError or ImportError:
    xp = np
    device = 'cpu'

def model_wrapper(model, device, param_inds, dL_ind, samples):
    xdata = samples[:,param_inds]
    out = run_on_dataset(model,xdata,distances=samples[:,dL_ind].flatten(),
                               device=device,y_transform_fn=None,runtime=False, eval_model=False)
    try:
        out = out.get()
    except AttributeError:
        pass
    return out

def nr_selection_function_one(model, snr_threshold, samples):
    samples_in_shapes = samples.shape
    out = model(samples.reshape(-1, samples.shape[-1]))
    out[out < 0] = 0
    probs = 1-ncx2(1,out**2).cdf(snr_threshold**2)
    return np.mean(probs.reshape(samples_in_shapes[:-1]), axis=-1)

torch.set_num_threads(1)

zmax = 1

T = 10#2.  # set to either 2 or 10 years
data_points = int(1e5)
z_spl, dL_spl, dVdz_spl = construct_cosmo_splines(zmax)
lMs = np.random.uniform(-4, -1, data_points)
mmins = np.random.uniform(5e4, 5e5, data_points)
mmaxs = np.random.uniform(5e6, 5e7, data_points)
lmus = np.random.uniform(-4, 1, data_points)
mumins = np.random.uniform(1, 5, data_points)
mumaxs = np.random.uniform(80, 100, data_points)
muas = np.random.uniform(0.05, 0.95, data_points)
sigmaas = np.random.uniform(1e-3, 2, data_points)

hypers = np.vstack((lMs,mmins,mmaxs,lmus,mumins,mumaxs,muas,sigmaas)).T

name = 'T10_LR1e-4_10L_5B_128N'#'T2_LR1e-4_10L_5B_128N'
net = load_mlp(name, device, get_state_dict=True,outdir="./models/snr").to(device)
net.eval()
net.xscalevals = torch.as_tensor(net.xscalevals, device=device).float()
net.yscalevals = torch.as_tensor(net.yscalevals, device=device).float()

wrap = partial(model_wrapper, net, device, xp.asarray([0,1,2,3,4,6,7,8,9]), 5)

part_selfun = partial(nr_selection_function_one, wrap, 20)

out_arr = np.zeros((data_points, 9))
out_arr[:,0] = lMs
out_arr[:,1] = mmins
out_arr[:,2] = mmaxs
out_arr[:,3] = lmus
out_arr[:,4] = mumins
out_arr[:,5] = mumaxs
out_arr[:,6] = muas
out_arr[:,7] = sigmaas

sampfun = partial(draw_samples_from_population, z_spl, dL_spl, T, int(5e2), int(1e5), 'cupy')

batch_size = 10
nbatches = data_points // batch_size
samps = sampfun(*hypers[0])  # cache sizes for the lols
outs = np.zeros(data_points)
st = time.perf_counter()

for bn in range(nbatches):
    print('Batch ',bn+1)
    in_samps = xp.zeros((batch_size,samps.shape[0],samps.shape[1]))
    for k,row in enumerate(hypers[bn*batch_size:(bn+1)*batch_size]):
        in_samps[k,:] = sampfun(*row)
    print('Drew samples, processing...')
    outs[bn*batch_size:(bn+1)*batch_size] = (part_selfun(in_samps))
et = time.perf_counter()

out_arr[:,8] = outs

out_df = pd.DataFrame(out_arr, columns=['lambda_M','mmin','mmax','lambda_mu','mumin','mumax','mu_a','sigma_a','selection_fraction'])
out_df.to_csv(f'../T{T}_net.csv', index=False)
print('Completed!')
print('Total time:',et-st)
print('Per event:',(et-st)/data_points)