from logging import warning
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def pdf_plot(data: np.ndarray, bins: int = 100, names: Optional[List[str]] = None):
    """ PDF plot

    @param data: numpy array to plot
    @param bins: number of bins
    @param names: name of series in case of multiseries plot
    @return: pyplot object
    """
    plt.figure('pdf', figsize=(6, 6))

    if len(data.shape) == 2:
        if names and len(names) != data.shape[0]:
            warning("Number of names should be equal to number of data series")
            names = ["Series {}".format(idx) for idx in range(data.shape[0])]

        for idx, row in enumerate(data):
            plt.hist(row, bins=bins, lw=3, normed=True, histtype='step', label=names[idx])
    else:
        plt.hist(data.flatten(), bins=bins, lw=3, normed=True, histtype='step')

    plt.yscale("log")
    plt.xlabel("Values")
    plt.ylabel("PDF")

    return plt


def slice_plot(data: np.ndarray,
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
    plt.figure('slice', figsize=(16, 6))

    if len(data.shape) not in [3, 4]:
        return None

    if len(data.shape) == 4:
        if feature is None:
            warning("Feature was not provided. First one will be used")
            feature = 0
        data = data[:, :, :, feature]

    if n_slice is None:
        warning("Slice is not selected first one will be used")
        n_slice = 0

    slice_to_plot = data[n_slice]

    plt.title("Slice" if name is None else name)
    plt.imshow(slice_to_plot)

    return plt


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
