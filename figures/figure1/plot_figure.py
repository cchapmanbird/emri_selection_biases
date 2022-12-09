import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cupy as xp
from functools import partial
from utils import model_wrapper, nn_model_wrapper, rescale_and_map, cumulative_dist, load_mlp
from matplotlib import rc
import seaborn as sns

pal = sns.color_palette(palette="colorblind").as_hex()

plt.style.use('default')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
torch.set_num_threads(1)

snr_grid = pd.read_csv('../../data/snr/T2_gridded.csv')
snr_unif = pd.read_csv('../../data/snr/T2_test.csv')
snr_unif = snr_unif.drop(np.where(np.logical_or(snr_unif['Y0'] < 0,snr_unif['SNR'] == 0))[0])

testdata = xp.asarray(snr_unif.to_numpy())
testdata[:,1] = 10**(testdata[:,1] + testdata[:,0])
test_snrs = testdata[:,-1].get()
test_params = testdata[:,:-1]
griddata = xp.asarray(snr_grid.to_numpy())
grid_snrs = griddata[:,-1]
grid_params = griddata[:,:-1]

grid_dims = xp.asarray([5,5,3,3,3,5,7,5,10])
reshaped_grid = grid_snrs.reshape(grid_dims.get())
lims = xp.zeros((len(grid_dims),2))
lims[:,0] = xp.asarray(grid_params.min(axis=0))
lims[:,1] = xp.asarray(grid_params.max(axis=0))

lin_rescaler_partial = partial(rescale_and_map,reshaped_grid, lims, grid_dims, 1)
lin_wrap = partial(model_wrapper, lin_rescaler_partial, xp.asarray([0,1,2,3,4,5,6,7,8]))

nearest_rescaler_partial = partial(rescale_and_map,reshaped_grid, lims, grid_dims, 0)
nearest_wrap = partial(model_wrapper, nearest_rescaler_partial, xp.asarray([0,1,2,3,4,5,6,7,8]))

spline_rescaler_partial = partial(rescale_and_map,reshaped_grid, lims, grid_dims, 3)
spline_wrap = partial(model_wrapper, spline_rescaler_partial, xp.asarray([0,1,2,3,4,5,6,7,8]))

device = 'cuda:0'
name = 'T2_LR1e-4_10L_5B_128N'
net = load_mlp(name, device, get_state_dict=True,outdir="../../models/snr").to(device)
net.eval()
net.xscalevals = torch.as_tensor(net.xscalevals, device=device).float()
net.yscalevals = torch.as_tensor(net.yscalevals, device=device).float()
wrap = partial(nn_model_wrapper, net, device, xp.asarray([0,1,2,3,4,5,6,7,8]))

nearest_snrs = nearest_wrap(test_params.copy())
lin_snrs = lin_wrap(test_params.copy())
spline_snrs = spline_wrap(test_params.copy())
net_snrs = wrap(test_params.copy()).cpu().numpy()
net_snrs[net_snrs < 0] = 0

near_diffs = nearest_snrs - test_snrs
lin_diffs = lin_snrs - test_snrs
spline_diffs = spline_snrs - test_snrs
net_diffs = net_snrs - test_snrs

near_abs_cdf = cumulative_dist(abs(near_diffs))
lin_abs_cdf = cumulative_dist(abs(lin_diffs))
spl_abs_cdf = cumulative_dist(abs(spline_diffs))
net_abs_cdf = cumulative_dist(abs(net_diffs))

near_mismatches = nearest_snrs/test_snrs
lin_mismatches = lin_snrs/test_snrs
spline_mismatches = spline_snrs/test_snrs
net_mismatches = net_snrs/test_snrs

near_rel_cdf = cumulative_dist(abs(1-near_mismatches[near_mismatches != 0]))
lin_rel_cdf = cumulative_dist(abs(1-lin_mismatches[lin_mismatches != 0]))
spl_rel_cdf = cumulative_dist(abs(1-spline_mismatches[spline_mismatches != 0]))
net_rel_cdf = cumulative_dist(abs(1-net_mismatches[net_mismatches != 0]))


# plt.plot(*lin_abs_cdf, label='Linear')
# plt.plot(*near_abs_cdf, label='Nearest Neighb.')
# plt.plot(*spl_abs_cdf, label='Cubic Spline')
# plt.plot(*net_abs_cdf, label='Network')
# plt.xlim(1e-4,500)
# plt.ylim(0,1)
# plt.xscale('log')
# plt.xlabel(r'$|\rho_\textrm{true} - \rho_\textrm{pred}|$', fontsize=20)
# plt.ylabel('CDF', fontsize=20)
# plt.tick_params(which='both',labelsize=17)
# plt.legend(fontsize=15)
# plt.tight_layout()
# plt.savefig('absdiff_CDF.pdf')
# plt.savefig('absdiff_CDF.png',dpi=500)
# plt.close()

# plt.plot(*lin_rel_cdf, label='Linear')
# plt.plot(*near_rel_cdf, label='Nearest Neighb.')
# plt.plot(*spl_rel_cdf, label='Cubic Spline')
# plt.plot(*net_rel_cdf, label='Network')
# plt.xlim(1e-6,20)
# plt.ylim(0,1)
# plt.xscale('log')
# plt.xlabel(r'$|1 - \rho_\textrm{true}/\rho_\textrm{pred}|$', fontsize=20)
# plt.ylabel('CDF', fontsize=20)
# plt.tick_params(which='both',labelsize=17)
# plt.legend(fontsize=15)
# plt.tight_layout()
# plt.savefig('reldiff_CDF.pdf')
# plt.savefig('reldiff_CDF.png',dpi=500)

# lws = [2,3,4,1]
# lss = ['--','-.',':','-']

lws = [4,3,2,1]
lss = [':','--','-.','-']

fig, ax = plt.subplots(ncols=2,figsize=(11,5))

ax[0].plot(*near_abs_cdf, label='Nearest Neighb.',color=pal[0],lw=lws[0], ls=lss[0])
ax[0].plot(*lin_abs_cdf, label='Linear',color=pal[1],lw=lws[1], ls=lss[1])
ax[0].plot(*spl_abs_cdf, label='Cubic Spline',color=pal[2],lw=lws[2], ls=lss[2])
ax[0].plot(*net_abs_cdf, label='Network',color=pal[3],lw=lws[3], ls=lss[3])

ax[0].set_xlim(5e-4,500)
ax[0].set_ylim(0,1)
ax[0].set_xscale('log')
ax[0].set_xlabel(r'$|\rho_\mathrm{true} - \rho_\mathrm{pred}|$', fontsize=20)
ax[0].set_ylabel('CDF', fontsize=20)
ax[0].tick_params(which='both',labelsize=17)
ax[0].legend(loc='upper left', fontsize=15, frameon=False)
# ax[0].text(0.95, 0.05, '(a)', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes, fontsize=17)

ax[1].plot(*near_rel_cdf, label='Nearest Neighb.', color=pal[0],lw=lws[0], ls=lss[0])
ax[1].plot(*lin_rel_cdf, label='Linear',color=pal[1],lw=lws[1], ls=lss[1])
ax[1].plot(*spl_rel_cdf, label='Cubic Spline',color=pal[2],lw=lws[2], ls=lss[2])
ax[1].plot(*net_rel_cdf, label='Network',color=pal[3],lw=lws[3], ls=lss[3])
ax[1].set_xlim(1e-5,20)
ax[1].set_ylim(0,1)
ax[1].set_xscale('log')
ax[1].set_xlabel(r'$|1 - \rho_\mathrm{true}/\rho_\mathrm{pred}|$', fontsize=20)
ax[1].tick_params(axis='both', which='both',labelsize=17, labelleft=False)
# ax[1].text(0.95, 0.05, '(b)', horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes, fontsize=17)

fig.tight_layout()

plt.savefig('CDFs.pdf')
plt.savefig('CDFs.png',dpi=500)