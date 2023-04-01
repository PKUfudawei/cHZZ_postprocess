import uproot
import awkward as ak
import numpy as np
import math
from coffea.nanoevents.methods import vector
import os

import boost_histogram as bh
from matplotlib import pyplot as plt
import matplotlib as mpl
from cycler import cycler
import mplhep as hep
use_helvet = True  ## true: use helvetica for plots, make sure the system have the font installed
if use_helvet:
    CMShelvet = hep.style.CMS
    CMShelvet['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.style.use(CMShelvet)
else:
    plt.style.use(hep.style.CMS)

def _get_variable_names(expr, exclude=['awkward', 'ak', 'np', 'numpy', 'math']):
    import ast
    root = ast.parse(expr)
    return sorted({node.id for node in ast.walk(root) if isinstance(
        node, ast.Name) and not node.id.startswith('_')} - set(exclude))


def _eval_expr(expr, df, dfext, sam):
    tmp = {'df': df[sam], 'd': dfext[sam], 'dfext': dfext[sam]}
    tmp.update({'math': math, 'np': np, 'numpy': np, 'ak': ak, 'awkward': ak})
    return eval(expr, tmp)

def make_plot_flv(title, df, dfext, nbin=50, xmin=70, xmax=170, kc=10, content_arr='df.ZZMass', flv_arr='', weight_arr='df.weight', xlabel=r'$m_{4\ell}$ [GeV]', ylabel='Events / 2 GeV', 
        ylog=False, custom_command=None, store_plot=False):

    f, ax = plt.subplots(figsize=(10, 10))
    hep.cms.label(data=True, llabel='Preliminary', year=2018, ax=ax, fontname='sans-serif') # llabel='Work in progress'

    plot_bkg_info = {
        'ggH125l': ('ggH (l)', 'lightgrey'),
        'ggH125b': ('ggH (b)', 'violet'),
        'ggH125c': ('ggH (c)', 'royalblue'),
        'HC_4FSFxFxl': ('H+c $(\kappa_c=10)$ (l)', 'gainsboro'),
        'HC_4FSFxFxb': ('H+c $(\kappa_c=10)$ (b)', 'plum'),
        'HC_4FSFxFxc': ('H+c $(\kappa_c=10)$ (c)', 'cornflowerblue'),
    }
    histkey = plot_bkg_info.keys()
    # get weighted boost histogram
    hist = {}
    flv_map = {'l': 0, 'b': 5, 'c': 4}
    for sam in ['ggH125', 'HC_4FSFxFx']:
        for flv in 'lbc':
            if isinstance(nbin, int):
                axis = bh.axis.Regular(nbin, xmin, xmax)
            else:
                axis = bh.axis.Variable(nbin)
            hist[sam+flv] = bh.Histogram(axis, storage=bh.storage.Weight())
            hist[sam+flv].fill(
                _eval_expr(f'{content_arr}[{flv_arr} == {flv_map[flv]}]', df, dfext, sam),
                weight=_eval_expr(f'{weight_arr}[{flv_arr} == {flv_map[flv]}]', df, dfext, sam)
            )
            if sam == 'HC_4FSFxFx':
                hist[sam+flv].view().value = hist[sam+flv].view().value * 100
                hist[sam+flv].view().variance = hist[sam+flv].view().variance * 100**2

    hist_add = sum([hist[k] for k in hist])
    print(hist['ggH125c'].view().value / hist['ggH125l'].view().value, hist['ggH125b'].view().value / hist['ggH125l'].view().value)

    # make stacked plot
    hep.histplot(
        [hist[k].view().value for k in histkey], 
        bins=hist_add.axes[0].edges,
        yerr=[np.sqrt(hist[k].view().variance) for k in histkey],
        label=[plot_bkg_info[k][0] + ' (%.2f)'%sum(hist[k].view().value) for k in histkey], color=[plot_bkg_info[k][1] for k in histkey], 
        histtype='fill', edgecolor='k', linewidth=1, stack=True,
    )

    # plot signal stat uncertainties
    bkgtot, bkgtot_err = hist_add.view(flow=False).value, np.sqrt(hist_add.view(flow=False).variance)
    ax.fill_between(hist_add.axes[0].edges, (bkgtot-bkgtot_err).tolist()+[0], (bkgtot+bkgtot_err).tolist()+[0], label='BKG stat. unce.', step='post', hatch='///', edgecolor='darkblue', facecolor='none', linewidth=0) ## draw bkg unce.

    ax.legend(prop={'size': 20}, ncol=2, labelspacing=0.3, borderpad=0.3)
    ax.set_xlabel(xlabel, ha='right', x=1.0); ax.set_ylabel(ylabel, ha='right', y=1.0);
    ax.set_xlim(xmin, xmax); ax.set_ylim(0, ax.get_ylim()[1]*1.2)
    if ylog:
        ax.set_yscale('log')
    if custom_command is not None:
        exec(custom_command)
    
    if store_plot:
        store_dir = '../plots/histo_1112/'
        plt.savefig(store_dir + f'/ggh_{title}.jpg')
        plt.savefig(store_dir + f'/ggh_{title}.pdf')
