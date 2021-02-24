'''
Plotting routines

You can adjust the style to your liking by changing 'param'.
'''

from logging import warning
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import pandas as pd
import numpy as np
import warnings

from scipy.stats import ks_2samp
from scipy.interpolate import interp1d
import sapsan.utils.hiddenlayer as hl


params = {'axes.labelsize': 20, 'legend.fontsize': 15, 'xtick.labelsize': 17,'ytick.labelsize': 17,
          'axes.titlesize':24, 'axes.linewidth': 1, 'lines.linewidth': 1.5,
          'xtick.major.width': 1,'ytick.major.width': 1,'xtick.minor.width': 1,'ytick.minor.width': 1,
          'xtick.major.size': 4,'ytick.major.size': 4,'xtick.minor.size': 3,'ytick.minor.size': 3,
          'axes.formatter.limits' : [-7, 7], 'text.usetex': False}

def pdf_plot(series: List[np.ndarray], bins: int = 100, names: Optional[List[str]] = None):
    """ PDF plot

    @param series: series of numpy arrays to build a pdf plot from
    @param bins: number of bins
    @param names: name of series in case of multiseries plot
    @return: pyplot object
    """
    mpl.rcParams.update(params)
    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    if not names:
        names = ["Data {}".format(i) for i in range(len(series))]

    for idx, data in enumerate(series):
        ax.hist(data.flatten(), bins=bins, lw=3, density=True, histtype='step', label=names[idx])

    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2,2)) 
    plt.legend(loc=1)
    plt.yscale("log")
    plt.xlabel("Values")
    plt.ylabel("PDF")
    plt.tight_layout()

    return plt


def cdf_plot(series: List[np.ndarray], names: Optional[List[str]] = None):
    """ CDF plot

    @param series: series of numpy arrays to build a cdf plot
    @param names: name of series in case of multiseries plot
    @return: pyplot object
    """
    mpl.rcParams.update(params)
    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    if not names:
        names = ["Data {}".format(i) for i in range(len(series))]

    func = []
    val = np.zeros((len(series),np.prod(np.shape(series[0]))))
    for idx, data in enumerate(series):
        val[idx] = np.sort(data.flatten())

        #cdf calculation via linear interpolation
        length = len(val[idx])
        yvals = np.linspace(0,length-1, length)/length
        ax.plot(val[idx], yvals, label=names[idx])
        func.append(interp1d(val[idx], yvals))  

        if idx==1:
            ks_stat, pvalue = ks_2samp(val[0], val[1])
            minima = max([min(val[0]), min(val[1])])
            maxima = min([max(val[0]), max(val[1])])

            xtest = np.linspace(minima, maxima, length*10)
            D = abs(func[0](xtest)-func[1](xtest))
            Dmax = max(D)
            Dpos = xtest[np.argmax(D)]
            ax.axvline(x=Dpos, linewidth=1, color='tab:red', linestyle='--')

            txt = ('pvalue = %.3e\n'%pvalue+
                     r'$\rm ks_{stat}$'+' = %.3e\n'%ks_stat+
                     #r'$\rm ks_{line}$'+' = %.3e\n'%Dmax+
                     r'$\rm line_{pos}$'+' = %.3e'%Dpos)

            plt.figtext(0.25, 0.55, txt, fontsize=14)        

    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2,2)) 
    plt.legend()
    plt.xlabel('Values')
    plt.ylabel('CDF')
    plt.tight_layout()
    
    return plt


def slice_plot(series: List[np.ndarray], names: Optional[List[str]] = None, cmap = 'plasma'):
    mpl.rcParams.update(params)
    if not names:
        names = ["Data {}".format(i) for i in range(len(series))]
    
    #colormap range is based on the target slice
    vmin = np.amin(series[-1])
    vmax = np.amax(series[-1])
    
    fig = plt.figure(figsize = (16, 6))
    for idx, data in enumerate(series):
        fig.add_subplot(121+idx)
        im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax = vmax)
        plt.colorbar(im).ax.tick_params(labelsize=14)
        plt.title(names[idx])
    plt.tight_layout()
    
    return plt


def log_plot(show_history = True, log_path = 'logs/log.txt'):
    
    plot_data = {'epoch':[], 'train_loss':[]}

    with open(log_path) as file:
        lines = list(file)
        total_lines = len(lines)
        ind = 0

        current_epoch = int(lines[-2].split('/')[0])
        while current_epoch!=1:
            current_epoch = int(lines[total_lines-ind-2].split('/')[0])
            train_loss = float(lines[total_lines-ind-2].split('loss=')[-1])
            valid_loss = float(lines[total_lines-ind-1].split('loss=')[-1])

            metrics = {'train_loss':train_loss, 'valid_loss':valid_loss}
            plot_data['epoch'] = np.append(plot_data['epoch'], current_epoch)
            plot_data['train_loss'] = np.append(plot_data['train_loss'], metrics['train_loss'])                
            ind += 4
        
        df = pd.DataFrame(plot_data)

        if len(plot_data['epoch']) == 1:
            plotting_routine = px.scatter
        else:
            plotting_routine = px.line
        fig = plotting_routine(df, x="epoch", y="train_loss", log_y=True,
                      title='Training Progress', width=700, height=400)
        fig.update_layout(yaxis=dict(exponentformat='e'))
        fig.layout.hovermode = 'x' 
        
        if show_history: fig.show()

        return fig

    
def model_graph(model, shape: np.array, transforms = None):
    import torch
    
    if len(np.shape(shape)) != 1: raise ValueError("Error: please provide the 'shape', "
                                                   "not the input data array itself.")    
    
    if transforms == None:
        transforms = [
                        hl.transforms.Fold("Conv > MaxPool > Relu", "ConvPoolRelu"),
                        hl.transforms.Fold("Conv > MaxPool", "ConvPool"),    
                        hl.transforms.Prune("Shape"),
                        hl.transforms.Prune("Constant"),
                        hl.transforms.Prune("Gather"),
                        hl.transforms.Prune("Unsqueeze"),
                        hl.transforms.Prune("Concat"),
                        hl.transforms.Rename("Cast", to="Input"),
                        hl.transforms.FoldDuplicates()
                     ]

    shape = np.array(shape)
    if shape[1] != 1:
        shape[1] = 1
        warnings.warn("shape was changed to %s to draw a model graph."%str(shape))
    
    if len(shape) == 5: unit_input = torch.zeros([shape[0], 1, shape[2], shape[3], shape[4]])
    elif len(shape) == 4: unit_input = torch.zeros([shape[0], 1, shape[2], shape[3]])
    else: raise ValueError('Input shape can be either of 2D or 3D data')
        
    graph = hl.build_graph(model, unit_input, transforms = transforms)
    graph.theme = hl.graph.THEMES["blue"].copy()
    
    return graph
    
    
class PlotUtils(object):
    @classmethod
    def plot_histograms(cls):
        pass

    @classmethod
    def plot_pdf(cls, data):
        return pdf_plot(data)
    
    @classmethod
    def plot_cdf(cls, data):
        return cdf_plot(data)

    @classmethod
    def plot_slice(cls, data):
        return slice_plot(data)
    
    @classmethod
    def plot_log(cls, data):
        return log_plot(data)
