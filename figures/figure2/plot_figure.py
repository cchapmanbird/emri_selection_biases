import corner
import json 
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.lines as mlines

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'size':18})
matplotlib.rc('text', usetex=True)

pal = sns.color_palette(palette="colorblind").as_hex()

default_kwargs = dict(
    bins=32,
    smooth=0.9,
    truth_color='k',
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=False,
    show_titles=False,
    label_kwargs=dict(fontsize=24),
    labelpad=0.1)

nope = ["m_min","m_max","mu_min","mu_max","logL","logP","it"]
with open('../../data/population_posteriors/T10/0/SF/result.json','r') as f:
    sel_samples = json.load(f)["posterior_samples"]
for nop in nope:
    del sel_samples[nop]

with open('../../data/population_posteriors/T10/0/NO_SF/result.json','r') as f:
    nosel_samples = json.load(f)["posterior_samples"]
for nop in nope:
    del nosel_samples[nop]
nosel_samples["rate"] = np.random.poisson(lam=106, size=len(nosel_samples["lambda_M"])) #222 #237

with open('../../data/population_posteriors/T10/0/truths.json') as f:
    truths = json.load(f)

subset_inds = np.array([0,3,6,7,8])
truths_sub = dict()
for m,key in enumerate(truths.keys()):
    if m in subset_inds:
        truths_sub[key] = truths[key]
labels = [
    r"$\lambda_M$",r"$\lambda_\mu$", r"$\mu_a$",r"$\sigma_a$",r"$\mathcal{R}$",
]

fig = corner.corner(
    nosel_samples, color=pal[1], truths=truths_sub, labels=labels, hist_kwargs=dict(density=True,color=pal[1]), **default_kwargs
)

fig = corner.corner(
     sel_samples, fig=fig, color=pal[0], truths=truths_sub, labels=labels, hist_kwargs=dict(density=True,color=pal[0]), **default_kwargs
)

blue = mlines.Line2D([],[], color=pal[0], label='Selection effects corrected for')
orange = mlines.Line2D([],[], color=pal[1], label='Selection effects ignored')

fig.legend(handles=[blue, orange], loc=(0.37,0.85),fontsize=32, frameon=False)
fig.savefig('joint_corner_subset.pdf', bbox_inches='tight')
fig.savefig('joint_corner_subset.png',dpi=500, bbox_inches='tight')
