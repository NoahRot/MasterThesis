import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(sample, var_name, bins = 30, show_gauss = False, show_std = False, show_mean = False, show_median = False, show_percent =  False, percentile = 5):
    std = np.std(sample)
    mean = np.mean(sample)
    low_per = np.percentile(sample, percentile)
    heigh_per = np.percentile(sample, 100-percentile)
    median = np.median(sample)

    x = np.linspace(np.min(sample), np.max(sample), 1000)
    y = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
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
        ax.axvline(low_per, color='black', linestyle='-.', label=f'{100-2*percentile}% conf. int.')
        ax.axvline(heigh_per, color='black', linestyle='-.')
    if show_median:
        ax.axvline(median, color='black', linestyle='-', label=f"Median: {median:.3f}")
    ax.set_xlabel(var_name)
    ax.set_ylabel("Frequency")
    ax.legend()

    return fig, ax

def compute_uncertainties(sample, percentile = 5):
    # Check that the sample data are an array and contain more than 1 element
    if not isinstance(sample, np.ndarray) or len(sample) < 2:
        return 0, 0, 0
    
    low_per = np.percentile(sample, percentile)
    high_per = np.percentile(sample, 100-percentile)
    median = np.median(sample)

    return median - low_per, high_per - median, 0.5*(high_per - low_per)