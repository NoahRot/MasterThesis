"""
Load-displacement data class and several functions used for ploting as well as treated the 
experimental data

This module contains one class
- LoadDisplacement
    Contains load-displacement data as function of time
This module contains one class five functions
- plot_LD
    Plot the load-displacement curve
- plot_comparison_LD
    Plot several load-displacement curves to compare them
- plot_load
    Plot load as function of time
- plot_disp
    Plot displacement as function of time
- experimental_LD_treatment
    Treatement of the load-displacement experimental data

Author
------
ROTUNNO Noah

Date
----
2026
"""

import numpy as np
from typing import Union
import matplotlib.pyplot as plt

class LoadDisplacement(object):
    """
    A class representing a load-displacement curve

    Parameters
    ----------
    t : ndarray
        Array of time values [s]
    RF2 : ndarray
        Array of load values (reaction force) [N]
    U2 : ndarray
        Array of displacement values [mm]

    Note
    ----
    The time is give in seconds [s], the displacement in milimeters [mm] 
    and the load in Newtons [N].
    """
    
    def __init__(self, t : np.ndarray, RF2 : np.ndarray, U2 : np.ndarray):
        self.t = t 
        self.load = RF2 
        self.disp = U2

def plot_LD(ld : LoadDisplacement) -> Union[plt.Figure, plt.Axes]:
    """
    Plot the load-displacement curve

    Parameters
    ----------
    ld : LoadDisplacement
        The load-displacement data

    Returns
    -------
    plt.Figure
        Figure created
    plt.Axes
        Axe created
    """

    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(ld.disp, ld.load)

    ax.set_xlabel("$\Delta$ [mm]")
    ax.set_ylabel("$L$ [N]")

    return fig, ax

def plot_comparison_LD(ld_list : list[LoadDisplacement], legend : Union[list[str], None] = None) -> Union[plt.Figure, plt.Axes]:
    """
    Plot several load-displacement curves to compare them

    Parameters
    ----------
    ld_list : list[LoadDisplacement]
        A list containing the load displacement data of each curve
    legend : list[str] or None, default=None
        A list containing the legend of each curve. Can be None, in this case, no legend is  displayed

    Returns
    -------
    plt.Figure
        Figure created
    plt.Axes
        Axe created

    Raises
    ------
    ValueError
        If the length of the legend (if not None) and list of LD curves are different
    """

    # Check length of legend and ld curves
    if legend is not None and len(ld_list) != len(legend):
        print("ERROR: not the same number of load-displacement than legend entries")
        raise ValueError("Not the same number of load-displacement than legend entries")

    fig = plt.figure()
    ax = fig.subplots()
    
    for i in range(0, len(ld_list)):
        if legend is not None:
            ax.plot(ld_list[i].disp, ld_list[i].load, label=legend[i])
        else:
            ax.plot(ld_list[i].disp, ld_list[i].load)

    ax.set_xlabel("$\Delta$ [mm]")
    ax.set_ylabel("$L$ [N]")
    ax.legend()

    return fig, ax

def plot_load(ld : LoadDisplacement) -> Union[plt.Figure, plt.Axes]:
    """
    Plot load as function of time

    Parameters
    ----------
    ld : LoadDisplacement
        The load-displacement data

    Returns
    -------
    plt.Figure
        Figure created
    plt.Axes
        Axe created
    """
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(ld.t, ld.load)

    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("$L$ [N]")

    return fig, ax

def plot_disp(ld : LoadDisplacement) -> Union[plt.Figure, plt.Axes]:
    """
    Plot displacement as function of time

    Parameters
    ----------
    ld : LoadDisplacement
        The load-displacement data

    Returns
    -------
    plt.Figure
        Figure created
    plt.Axes
        Axe created
    """
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(ld.t, ld.disp)

    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("$\Delta$ [mm]")

    return fig, ax

def experimental_LD_treatment(ld : LoadDisplacement, nbr_point_threshold : int = 5, debug_plot : bool = False) -> LoadDisplacement:
    """
    Treatement of the experimental data. Clamp the data to the data in ROI. Offset correctly the time and displacement.
    Create a sort array to correctly sort the data according to the dispalcement for the computation of the area under LD curve.

    Parameters
    ----------
    ld : LoadDisplacement
        The load-displacement experimental data
    nbr_point_threshold : int, default=5
        Number of points used to compute the mean and std to find the beginning of the experiment
    debug_plot : bool, default=False
        If true, plot a figures containing graphs that show the different steps of the treatment

    Returns
    -------
    LoadDisplacement
        Treated load-displacement data

    Warnings
    --------
    If nbr_point_threshold has been chosen too high, it can take values on the ramp of the displacement.
    """
    
    # Compute the std of the "nbr_point_threshold" first points
    mean_begining = np.mean(ld.disp[:nbr_point_threshold])
    std_begining = np.std(ld.disp[:nbr_point_threshold])
    mean_end = np.mean(ld.disp[-nbr_point_threshold:])
    std_end = np.std(ld.disp[-nbr_point_threshold:])

    # Get the index at which the slope begin (need 10 values in a row above mean+5*std)
    begin_index = 0
    threshold = mean_begining + 5*std_begining
    while not np.all(ld.disp[begin_index:begin_index + 10] > threshold):
        begin_index += 1

        if begin_index - 10 >= len(ld.disp):
            print("ERROR: Can not find the beginning of the U2 slope")
            raise ValueError("Can not find the beginning of the U2 slope")

    # Get the index at which the slope stop (need 10 values in a row below mean-5*std)
    end_index = len(ld.disp) - 1
    threshold = mean_end - 5*std_end
    while not np.all(ld.disp[end_index-10:end_index] < threshold):
        end_index -= 1

        if end_index < 10:
            print("ERROR: Can not find the ending of the U2 slope")
            raise ValueError("Can not find the ending of the U2 slope")

    # Check the end index with the force
    for i in range(begin_index, end_index):
        if ld.load[i+1] < 0.8*ld.load[i] and ld.load[i] > np.max(ld.load)*0.7:
            end_index = i
            break

    # Plot before treatment
    if debug_plot:
        fig = plt.figure()
        ax = fig.subplots(2,3)
        ax[0,0].plot(ld.t, ld.disp)
        ax[0,0].hlines(mean_begining,                  np.min(ld.t), np.max(ld.t), colors = "black", linestyles="-")
        ax[0,0].hlines(mean_begining - 5*std_begining, np.min(ld.t), np.max(ld.t), colors = "black", linestyles="--")
        ax[0,0].hlines(mean_begining + 5*std_begining, np.min(ld.t), np.max(ld.t), colors = "black", linestyles="--")
        ax[0,0].hlines(mean_end,                  np.min(ld.t), np.max(ld.t), colors = "black", linestyles="-")
        ax[0,0].hlines(mean_end - 5*std_end, np.min(ld.t), np.max(ld.t), colors = "black", linestyles="--")
        ax[0,0].hlines(mean_end + 5*std_end, np.min(ld.t), np.max(ld.t), colors = "black", linestyles="--")
        ax[0,0].vlines(ld.t[begin_index], np.min(ld.disp), np.max(ld.disp), colors = "red", linestyles="--")
        ax[0,0].vlines(ld.t[end_index], np.min(ld.disp), np.max(ld.disp), colors = "red", linestyles="--")
        ax[0,0].set_xlabel("$t$ [s]")
        ax[0,0].set_ylabel("$\Delta$ [mm]")

        ax[0,1].plot(ld.t, ld.load)
        ax[0,1].vlines(ld.t[begin_index], np.min(ld.load), np.max(ld.load), colors = "red", linestyles="--")
        ax[0,1].vlines(ld.t[end_index], np.min(ld.load), np.max(ld.load), colors = "red", linestyles="--")
        ax[0,1].set_xlabel("$t$ [s]")
        ax[0,1].set_ylabel("$L$ [kN]")

        ax[0,2].plot(ld.disp, ld.load)
        ax[0,2].set_xlabel("$\Delta$ [mm]")
        ax[0,2].set_ylabel("$L$ [kN]")

    # Treat the data
    ld.t = ld.t[begin_index:end_index] - ld.t[begin_index]
    ld.disp = ld.disp[begin_index:end_index]
    ld.load = ld.load[begin_index:end_index]

    # Place first displacement at 0
    ld.disp -= np.min(ld.disp)

    # Debug plot after treatment
    if debug_plot:
        ax[1,0].plot(ld.t, ld.disp, color="green")
        ax[1,0].set_xlabel("$t$ [s]")
        ax[1,0].set_ylabel("$\Delta$ [mm]")

        ax[1,1].plot(ld.t, ld.load, color="green")
        ax[1,1].set_xlabel("$t$ [s]")
        ax[1,1].set_ylabel("$L$ [kN]")

        ax[1,2].plot(ld.disp, ld.load, color="green")
        ax[1,2].set_xlabel("$\Delta$ [mm]")
        ax[1,2].set_ylabel("$L$ [kN]")

    return ld