from logging import warning
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def pdf_plot(series: List[np.ndarray], bins: int = 100, names: Optional[List[str]] = None):
    """ PDF plot

    @param series: series of numpy arrays to build pds plot from
    @param bins: number of bins
    @param names: name of series in case of multiseries plot
    @return: pyplot object
    """
    if not names:
        names = ["Data {}".format(i) for i in range(len(series))]

    for idx, data in enumerate(series):
        plt.hist(data.flatten(), bins=bins, lw=3, normed=True, histtype='step', label=names[idx])

    plt.yscale("log")
    plt.xlabel("Values")
    plt.ylabel("PDF")

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
