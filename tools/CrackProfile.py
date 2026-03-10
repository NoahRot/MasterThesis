import numpy as np
import matplotlib.pyplot as plt

"""
Class representing the crack profile
Input/Parameters:
 - l_i (array): 9 values at which the crack length have been measured
 - a_i (array): 9 values of crack length
"""
class CrackProfile(object):
    def __init__(self, l_i, a_i):
        if len(l_i) != 9 or len(a_i) != 9:
            print(f"ERROR: l_i or a_i are not composed of 9 elements. len(l_i)={len(l_i)}, len(a_i)={len(a_i)}")
            raise ValueError("l_i or a_i are not composed of 9 elements. len(l_i)={len(self.l_i)}, len(a_i)={len(self.a_i)}")
        self.l_i = l_i
        self.a_i = a_i

    """
    Initial crack length calculation
    Output:
     - (float/array) Initial crack lengh a0
    """
    def initial_crack_length(self):
        return 0.125*(0.5*(self.a_i[0] + self.a_i[-1]) + np.sum(self.a_i[1:-1], axis=0))
    
class CrackProfileDistribution(object):
    def __init__(self, l_i, l_i_u : float, a_i, a_i_u : float):
        if len(l_i) != 9 or len(a_i) != 9:
            print(f"ERROR: l_i or a_i are not composed of 9 elements. len(l_i)={len(self.l_i)}, len(a_i)={len(self.a_i)}")
            raise ValueError("l_i or a_i are not composed of 9 elements. len(l_i)={len(self.l_i)}, len(a_i)={len(self.a_i)}")
        self.l_i = np.asarray(l_i, dtype=float)
        self.a_i = np.asarray(a_i, dtype=float)
        self.l_i_u = l_i_u
        self.a_i_u = a_i_u

    def simple(self) -> CrackProfile:
        return CrackProfile(self.l_i, self.a_i)

    def sample(self, nbr_samples : int, rng : np.random.Generator = None) -> CrackProfile:
        if rng is None:
            rng = np.random.default_rng()

        a_sample = rng.normal(self.a_i, self.a_i_u, (nbr_samples, 9))
        l_sample = rng.normal(self.l_i, self.l_i_u, (nbr_samples, 9))

        return CrackProfile(np.transpose(l_sample), np.transpose(a_sample))
    
def crack_profile_distribution(crack_profile : CrackProfile, l_i_u : float, a_i_u : float):
    return CrackProfileDistribution(crack_profile.l_i, l_i_u, crack_profile.a_i, a_i_u)
    
def plot_crack_profile(crack : CrackProfile):
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(crack.l_i, crack.a_i, marker='+', label="Crack length $a_i$")
    ax.axhline(crack.initial_crack_length(), linestyle="--", color="black", label="Average $a_0$")
    ax.set_xlabel("$l$ [mm]")
    ax.set_ylabel("$a_i$ [mm]")
    ax.legend()