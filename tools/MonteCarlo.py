import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(sample : np.ndarray, var_name : str, bins : int = 30, 
                   show_gauss : bool = False, 
                   show_std : bool = False, 
                   show_mean : bool = False, 
                   show_median : bool = False, 
                   show_percent : bool =  False, percentile : float = 5):
    """
    Plot the histogram of a sampled data

    Parameters
    ----------
    sample : np.ndarray
        Sampled data
    var_name : str
        Name of the variable to plot (for axis label)
    bins : int, default 30
        Number of bins for the histogram
    show_gauss : bool, default False
        Whether to show the Gaussian distribution curve corresponding to the sample data
    show_std : bool, default False
        Whether to show the standard deviation of the sample data on the histogram
    show_mean : bool, default False
        Whether to show the mean of the sample data on the histogram
    show_median : bool, default False
        Whether to show the median of the sample data on the histogram
    show_percent : bool, default False
        Whether to show the confidence interval corresponding to the given percentile on the histogram
    percentile : float, default 5
        Percentile to compute the confidence interval (e.g. 5 for 90% confidence interval)

    Returns
    -------
    Figure
        Matplotlib figure object
    Axes
        Matplotlib axes object
    """
    # Compute statistics
    std = np.std(sample)
    mean = np.mean(sample)
    low_per = np.percentile(sample, percentile)
    heigh_per = np.percentile(sample, 100-percentile)
    median = np.median(sample)

    # Prepare gaussian
    x = np.linspace(np.min(sample), np.max(sample), 1000)
    y = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))

    # Plot
    fig, ax = plt.subplots()
    ax.hist(sample, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    if show_mean:
        ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')
    if show_std:
        ax.axvline(mean + std, color='black', linestyle='--', label=f'Std: {std:.3f}')
        ax.axvline(mean - std, color='black', linestyle='--')
    if show_gauss:
        ax.plot(x,y,color="red")
    if show_percent:
        ax.axvline(low_per, color='black', linestyle='-.', label=f'{100-2*percentile}% CI')
        ax.axvline(heigh_per, color='black', linestyle='-.')
    if show_median:
        ax.axvline(median, color='black', linestyle='-', label=f"Median: {median:.3f}")
    ax.set_xlabel(var_name)
    ax.set_ylabel("Frequency")
    ax.legend()

    return fig, ax

def compute_uncertainties(sample, percentile = 5):
    """
    Compute uncertainties from a sample data using percentiles

    Parameters
    ----------
    sample : np.ndarray
        Sample data
    percentile : float, default 5
        Percentile to compute the confidence interval (e.g. 5 for 90% confidence interval)
    
    Returns
    -------
    float
        Lower uncertainty bound
    float
        Upper uncertainty bound
    float
        Symmetric uncertainty (half of the confidence interval)

    Warnings
    --------
    If the sample data are not an array or contain less than 2 elements, the function returns 0 for all uncertainties
    """
    # Check that the sample data are an array and contain more than 1 element
    if not isinstance(sample, np.ndarray) or len(sample) < 2:
        return 0, 0, 0
    
    low_per = np.percentile(sample, percentile)
    high_per = np.percentile(sample, 100-percentile)
    median = np.median(sample)

    return median - low_per, high_per - median, 0.5*(high_per - low_per)