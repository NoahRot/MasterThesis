"""
Crack profile definition and crack profile uncertainties distribution for Monte-Carlo analysis.

This module contains two classes
- CrackProfile
    Represent the profile of the crack with crack length taken at 9 different points
- CrackProfileDistribution
    Describe statistical distributions of the parameters of a crack profile. It can generate
    deterministic crack profile or sample values for crack profile parameters.
This module contains two functions
- crack_profile_distribution
    Create a crack profile distribution using crack profile for the data as well as the
    uncertainties on the parameters
- plot_crack_profile
    Plot the crack profile

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

class CrackProfile(object):
    """
    Representation of the initial crack profile shape.

    Parameters
    ----------
    l_i : ndarray
        Distance along the width
    a_i : ndarray
        Crack length at the different l_i points

    Raises
    ------
    ValueError
        Error if both arrays (l_i and a_i) have not a length of 9.
    """

    def __init__(self, l_i : np.ndarray, a_i : np.ndarray):
        # Check if l_i and a_i are the correct length
        if len(l_i) != 9 or len(a_i) != 9:
            print(f"ERROR: l_i or a_i are not composed of 9 elements. len(l_i)={len(l_i)}, len(a_i)={len(a_i)}")
            raise ValueError("l_i or a_i are not composed of 9 elements. len(l_i)={len(self.l_i)}, len(a_i)={len(self.a_i)}")
        
        self.l_i = l_i
        self.a_i = a_i

    def initial_crack_length(self) -> float:
        """
        Compute the initial crack length a0

        Return
        ------
        float
            Initial crack length a0
        """
        return 0.125*(0.5*(self.a_i[0] + self.a_i[-1]) + np.sum(self.a_i[1:-1], axis=0))
    
class CrackProfileDistribution(object):
    """
    Statistical distribution of the crack profile parameters

    Parameters
    ----------
    l_i : ndarray
        Distance along the width
    l_i_u : float
        Uncertainty on distance measurement
    a_i : ndarray
        Crack length at the different l_i points
    a_i_u : float
        Uncertainty on crack length measurement
    """
    def __init__(self, l_i : np.ndarray, l_i_u : float, a_i : np.ndarray, a_i_u : float):
        # Check if l_i and a_i are the correct length
        if len(l_i) != 9 or len(a_i) != 9:
            print(f"ERROR: l_i or a_i are not composed of 9 elements. len(l_i)={len(self.l_i)}, len(a_i)={len(self.a_i)}")
            raise ValueError("l_i or a_i are not composed of 9 elements. len(l_i)={len(self.l_i)}, len(a_i)={len(self.a_i)}")
        
        self.l_i = np.asarray(l_i, dtype=float)
        self.a_i = np.asarray(a_i, dtype=float)
        self.l_i_u = l_i_u
        self.a_i_u = a_i_u

    def simple(self) -> CrackProfile:
        """
        Create a deterministic CrackProfile instance using the values without uncertainties

        Returns
        -------
        CrackProfile
            CrackProfile object built using nominal crack length measurements
        """
        return CrackProfile(self.l_i, self.a_i)

    def sample(self, nbr_samples : int, rng : Union[np.random.Generator, None] = None) -> CrackProfile:
        """
        Sample values to generate random CrackProfile for Monte-Carlo method

        Parameters
        ----------
        nbr_samples : int
            Number of sampled values
        rng : numpy.random.Generator or None, default=None
            Random number generator. If None, a default generator is created.

        Returns
        -------
        CrackProfile
            CrackProfile object containing sampled parameter arrays
        """
        if rng is None:
            rng = np.random.default_rng()

        a_sample = rng.normal(self.a_i, self.a_i_u, (nbr_samples, 9))
        l_sample = rng.normal(self.l_i, self.l_i_u, (nbr_samples, 9))

        return CrackProfile(np.transpose(l_sample), np.transpose(a_sample))
    
def crack_profile_distribution(crack_profile : CrackProfile, l_i_u : float, a_i_u : float) -> CrackProfileDistribution:
    """
    Create a crack profile distribution from a crack profile given uncertainties

    Parameters
    ----------
    crack_profile : CrackProfile
        Crack profile from data
    l_i_u : float
        Uncertainties on the distance along the width
    a_i_u : float
        Uncertainties on the crack length

    Returns
    -------
    CrackProfileDistribution
        Distribution of the crack profile data
    """
    return CrackProfileDistribution(crack_profile.l_i, l_i_u, crack_profile.a_i, a_i_u)
    
def plot_crack_profile(crack : CrackProfile):
    """
    Plot the crack profile

    Parameters
    ----------
    crack : CrackProfile
        The crack profile data
    """
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(crack.l_i, crack.a_i, marker='+', label="Crack length $a_i$")
    ax.axhline(crack.initial_crack_length(), linestyle="--", color="black", label="Average $a_0$")
    ax.set_xlabel("$l$ [mm]")
    ax.set_ylabel("$a_i$ [mm]")
    ax.legend()