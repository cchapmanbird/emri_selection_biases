import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import scipy.stats
from collections import namedtuple
import pandas as pd
import seaborn as sns

def make_pp_plot(CI_file_prefix, filename=None, confidence_interval=[0.68, 0.95, 0.997],
                 lines=None, legend_fontsize='x-small', title=True,
                 confidence_interval_alpha=0.1, fig=None, ax=None,legend=True,
                 colors = None, lss = None, lws = None,
                 **kwargs):
    credible_levels = pd.read_csv(f'../../data/credible_intervals/{CI_file_prefix}_credible_intervals.csv')
    # if lines is None:
    #     colors = ["C{}".format(i) for i in range(8)]
    #     linestyles = ["-", "--", ":"]
    #     lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    # if len(lines) < len(credible_levels.keys()):
    #     raise ValueError("Larger number of parameters than unique linestyles")

    x_values = np.linspace(0, 1, 1001)

    N = len(credible_levels)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4,4))

    if isinstance(confidence_interval, float):
        confidence_interval = [confidence_interval]
    if isinstance(confidence_interval_alpha, float):
        confidence_interval_alpha = [confidence_interval_alpha] * len(confidence_interval)
    elif len(confidence_interval_alpha) != len(confidence_interval):
        raise ValueError(
            "confidence_interval_alpha must have the same length as confidence_interval")

    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1. - ci) / 2.
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')

    pvalues = []
    for ii, key in enumerate(credible_levels):
        pp = np.array([sum(credible_levels[key].values < xx) /
                       len(credible_levels) for xx in x_values])
        pvalue = scipy.stats.kstest(credible_levels[key], 'uniform').pvalue
        pvalues.append(pvalue)
        name = key
        label = "{}".format(name)
        ax.plot(x_values, pp, ls = lss[ii], c=colors[ii], lw=lws[ii], label=label, **kwargs)
    Pvals = namedtuple('pvals', ['combined_pvalue', 'pvalues', 'names'])
    pvals = Pvals(combined_pvalue=scipy.stats.combine_pvalues(pvalues)[1],
                  pvalues=pvalues,
                  names=list(credible_levels.keys()))
    print(f"{scipy.stats.combine_pvalues(pvalues)[1]:.3e}")
    # ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    if legend:
        ax.legend(handlelength=2, labelspacing=0.25, fontsize=legend_fontsize, loc='upper left', frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=500)
        plt.close()

    return fig, pvals

if __name__ == "__main__":
    from matplotlib import rc

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    texlabels = [r"$\lambda_M$",r"$M_\mathrm{min}$",r"$M_\mathrm{max}$", r"$\lambda_\mu$",r"$\mu_\mathrm{min}$",r"$\mu_\mathrm{max}$",r"$\mu_a$",r"$\sigma_a$",r"$R$"]
    labels = ["lambda_M","m_min","m_max","lambda_mu","mu_min","mu_max","mu_a","sigma_a","rate"]
    
    pal = sns.color_palette(palette="colorblind").as_hex()
    colors = [pal[i] for i in range(5)] + [pal[i] for i in range(5)]
    lines = ['-' for i in range(5)] + ['--' for i in range(5)]
    lws = [1, 1.25, 1.5, 1.75, 2.] + [1, 1.25, 1.5, 1.75, 2.]
    fig, ax = plt.subplots(nrows=3, figsize=(4,10))
    make_pp_plot('NO_SF', fig=fig,ax=ax[0],legend=False, colors=colors, lss=lines, lws=lws)
    make_pp_plot('LINEAR_SF', fig=fig, ax=ax[1], legend=False, colors=colors, lss=lines, lws=lws)
    make_pp_plot('SF', fig=fig,ax=ax[2],legend=True, colors=colors, lss=lines, lws=lws, legend_fontsize='medium')
    ax[0].xaxis.set_ticklabels([])
    ax[1].xaxis.set_ticklabels([])
    ax[2].set_xlabel("C.I.")
    plt.subplots_adjust(hspace=0.05)
    for axis, letter, xdist,c in zip(ax, ['Uncorrected','Linear','Network'], [0.70,0.81,0.77], [pal[1],pal[2],pal[3]]):
        axis.text(xdist, 0.07, letter, transform=axis.transAxes, bbox={"fc":c+'50',"ec":(0,0,0,0)})
    fig.savefig('./joint_pp.png',dpi=500, bbox_inches='tight')
    fig.savefig('./joint_pp.pdf', bbox_inches='tight')