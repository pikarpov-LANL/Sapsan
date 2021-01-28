from logging import warning
from typing import List, Optional

import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np

from scipy.stats import ks_2samp
from scipy.interpolate import interp1d

def pdf_plot(series: List[np.ndarray], bins: int = 100, names: Optional[List[str]] = None):
    """ PDF plot

    @param series: series of numpy arrays to build a pdf plot from
    @param bins: number of bins
    @param names: name of series in case of multiseries plot
    @return: pyplot object
    """
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
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if not names:
        names = ["Data {}".format(i) for i in range(len(series))]

    func = []
    val = np.zeros((len(series),np.prod(np.shape(series[0]))))
    for idx, data in enumerate(series):
        print(idx)
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

    
def slice_of_cube(data: np.ndarray,
                  feature: Optional[int] = None,
                  n_slice: Optional[int] = None,
                  name: Optional[str] = None):
    """ Slice of 3d cube

    @param data: numpy array
    @param feature: feature of cube to use in case of multifeature plot
    @param n_slice: slice to use
    @param name: name of plot
    @return: pyplot object
    """
    if len(data.shape) not in [3, 4]:
        return None

    if len(data.shape) == 4:
        if feature is None:
            warning("Feature was not provided. First one will be used")
            feature = 0
        data = data[feature, :, :, :]

    if n_slice is None:
        warning("Slice is not selected first one will be used")
        n_slice = 0

    slice_to_plot = data[n_slice]

    return slice_to_plot


def log_plot(show_history = True):
    log_path = 'logs/log.txt'
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
    def plot_slices(cls):
        pass
    
    @classmethod
    def plot_log(cls, data):
        return log_plot(data)
