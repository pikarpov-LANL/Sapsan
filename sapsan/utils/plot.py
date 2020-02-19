from logging import warning
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import ks_2samp
from scipy.interpolate import interp1d

def pdf_plot(series: List[np.ndarray], bins: int = 100, names: Optional[List[str]] = None):
    """ PDF plot

    @param series: series of numpy arrays to build pds plot from
    @param bins: number of bins
    @param names: name of series in case of multiseries plot
    @return: pyplot object
    """
    plt.figure()
    if not names:
        names = ["Data {}".format(i) for i in range(len(series))]

    for idx, data in enumerate(series):
        plt.hist(data.flatten(), bins=bins, lw=3, normed=True, histtype='step', label=names[idx])

    plt.yscale("log")
    plt.xlabel("Values")
    plt.ylabel("PDF")

    return plt


def cdf_plot(series: List[np.ndarray], names: Optional[List[str]] = None):
    """ CDF plot

    @param series: series of numpy arrays to build pds plot from
    @param names: name of series in case of multiseries plot
    @return: pyplot object
    """
    plt.figure()
    if not names:
        names = ["Data {}".format(i) for i in range(len(series))]

    func = []
    print('shapes', np.shape(series))
    val = np.zeros((np.shape(series)[0],np.shape(series)[1]*np.shape(series)[2]))
    for idx, data in enumerate(series):
        val[idx] = np.sort(data.flatten())

        #cdf calculation via linear interpolation
        length = len(val[idx])
        yvals = np.linspace(0,length-1, length)/length
        plt.plot(val[idx], yvals, label=names[idx])
        func.append(interp1d(val[idx], yvals))  

        if idx==1:
            ks_stat, pvalue = ks_2samp(val[0], val[1])
            minima = max([min(val[0]), min(val[1])])
            maxima = min([max(val[0]), max(val[1])])

            xtest = np.linspace(minima, maxima, length*10)
            D = abs(func[0](xtest)-func[1](xtest))
            Dmax = max(D)
            Dpos = xtest[np.argmax(D)]
            plt.axvline(x=Dpos, linewidth=1, color='tab:red', linestyle='--')

            txt = ('pvalue = %.3e\n'%pvalue+
                     r'$\rm ks_{stat}$'+' = %.3e\n'%ks_stat+
                     r'$\rm ks_{line}$'+' = %.3e\n'%Dmax+
                     r'$\rm line_{pos}$'+' = %.3e'%Dpos)

            plt.figtext(0.17, 0.6, txt, fontsize=14)        

    plt.legend()
    plt.xlabel('Values')
    plt.ylabel('CDF')        

    
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


class PlotUtils(object):
    @classmethod
    def plot_histograms(cls):
        pass

    @classmethod
    def plot_pdf(cls, data):
        return pdf_plot(data)

    @classmethod
    def plot_slices(cls):
        pass
