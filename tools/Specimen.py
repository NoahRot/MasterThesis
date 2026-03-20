"""
Specimen definition and specimen uncertainties distribution for Monte-Carlo analysis.

This module contains two classes:
- Specimen
    Represent the geometry of a specimen and its material properties
- SpecimenDistribution
    Describe statistical distributions of the parameters of a specimen. It can generate
    deterministic specimen or sample values for specimen parameters.

Author
------
ROTUNNO Noah

Date
----
2026
"""

import numpy as np
from typing import Union
from tools.CrackProfile import *
from tools.Logger import Logger
from tools.MonteCarlo import compute_uncertainties
from tools.GeometricFunction import geometric_fnc_K

class Specimen(object):
    """
    Representation of the geometry and material of a specimen.

    Parameters
    ----------
    W : float or ndarray
        Width of the specimen [mm]
    S : float or ndarray
        Span [mm]
    B : float or ndarray
        Thickness [mm]
    B_N : float or ndarray
        Net thickness [mm]
    a0 : float or ndarray
        Initial crack length [mm]
    nu : float 
        Poisson ratio [-]
    E : float or ndarray
        Young modulus [MPa]
    eta_pl : float
        Plastic geometry factor used in J-integral calculations [-]
    sigma_YS : float or ndarray
        Yield strength of the material [MPa]

    Attributes
    ----------
    b0 : float or ndarray
        Initial remaining ligament [mm]
    E_plain_strain : float or ndarray
        Plane strain modulus [MPa]
    K_Jc_lim : float or ndarray
        Limiting fracture toughness for validity criteria [MPa mm^0.5]

    Note
    ----
    This class handle float for deterministic calculation and ndarray for 
    statistical Monte-Carlo uncertainties evaluation.
    """
    
    def __init__(self, 
                 W:Union[float,np.ndarray],
                 S:Union[float,np.ndarray], 
                 B:Union[float,np.ndarray], 
                 B_N:Union[float,np.ndarray], 
                 a0:Union[float,np.ndarray], 
                 nu:float, 
                 E:Union[float,np.ndarray],
                 eta_pl:float, 
                 sigma_YS:Union[float,np.ndarray]):
        # Geometry
        self.W = W
        self.S = S
        self.B = B
        self.B_N = B_N
        self.a0 = a0
        self.b0 = W - a0
        self.eta_pl = eta_pl
        self.sigma_YS = sigma_YS

        # Material
        self.nu = nu
        self.E = E
        self.E_plain_strain = E/(1 - self.nu**2)
        self.K_Jc_lim = np.sqrt(self.E*self.b0*self.sigma_YS/(30*(1-self.nu**2)))

    def is_sample(self) -> bool:
        """
        Check if the instance is a sample

        Returns
        -------
        bool
            True if the instance is a sample, False otherwise
        """
        
        return  isinstance(self.W, np.ndarray) or \
                isinstance(self.S, np.ndarray) or \
                isinstance(self.B, np.ndarray) or \
                isinstance(self.B_N, np.ndarray) or \
                isinstance(self.a0, np.ndarray) or \
                isinstance(self.E, np.ndarray) or \
                isinstance(self.sigma_YS, np.ndarray)

class SpecimenDistribution(object):
    """
    Statistical distribution of the specimen parameters.

    Parameters
    ----------
    W : float
        Width of the specimen [mm]
    W_u : float
        Uncertainty on width of the specimen [mm]
    S : float
        Span [mm]
    S_u : float
        Uncertainty on span [mm]
    B : float
        Thickness [mm]
    B_u : float
        Uncertainty on thickness [mm]
    B_N : float
        Net thickness [mm]
    B_N_u : float
        Uncertainty on net thickness [mm]
    nu : float 
        Poisson ratio [-]
    E : float
        Young modulus [MPa]
    E_u : float
        Uncertainty on Young modulus [MPa]
    eta_pl : float
        Plastic geometry factor used in J-integral calculations [-]
    sigma_YS : float
        Yield strength of the material [MPa]
    sigma_YS_u : float
        Uncertainty on yield strength of the material [MPa]
    crack_profile_dist : CrackProfileDistribution
        Statistical distribution of the intial crack length

    Note
    ----
    This class is used to create Specimen instance. It can create both deterministic instance
    using the measured values or statistical to handle Monte-Carlo uncertainty evaluation.
    """

    def __init__(self, 
                 W: float, W_u: float, 
                 S: float, S_u: float, 
                 B: float, B_u: float, 
                 B_N: float, B_N_u: float, 
                 nu: float, 
                 E: float, E_u: float, 
                 eta_pl: float, 
                 sigma_YS: float, sigma_YS_u: float, 
                 crack_profile_dist : CrackProfileDistribution):
        self.W = W
        self.W_u = W_u
        self.S = S
        self.S_u = S_u
        self.B = B
        self.B_u = B_u
        self.B_N = B_N
        self.B_N_u = B_N_u
        self.nu = nu
        self.E = E
        self.E_u = E_u
        self.eta_pl = eta_pl
        self.sigma_YS = sigma_YS
        self.sigma_YS_u = sigma_YS_u
        self.crack_profile_dist = crack_profile_dist

    def simple(self) -> Specimen:
        """
        Create a deterministic Specimen instance using the values without uncertainties

        Returns
        -------
        Specimen
            Specimen object built using nominal geometry, material properties,
            and the deterministic crack profile
        """
        return Specimen(self.W, self.S, self.B, self.B_N, self.crack_profile_dist.simple().initial_crack_length(), self.nu, self.E, self.eta_pl, self.sigma_YS)

    def sample(self, nbr_samples : int, rng : Union[np.random.Generator, None] = None) -> Specimen:
        """
        Sample values to generate random Specimen for Monte-Carlo method

        Parameters
        ----------
        nbr_samples : int
            Number of sampled values
        rng : numpy.random.Generator or None, default=None
            Random number generator. If None, a default generator is created.

        Returns
        -------
        Specimen
            Specimen object containing sampled parameter arrays
        """

        # Check if a rnd generator is provided 
        if rng is None:
            rng = np.random.default_rng()

        # Sample crack profile
        crack_profile = self.crack_profile_dist.sample(nbr_samples, rng)

        # Sample parameters
        W_sampled   = rng.uniform(self.W - self.W_u, self.W + self.W_u, nbr_samples)
        S_sampled   = rng.uniform(self.S - self.S_u, self.S + self.S_u, nbr_samples)
        B_sampled   = rng.uniform(self.B - self.B_u, self.B + self.B_u, nbr_samples)
        B_N_sampled = rng.uniform(self.B_N - self.B_N_u, self.B_N + self.B_N_u, nbr_samples)
        a0_sampled = crack_profile.initial_crack_length()
        E_sampled = rng.normal(self.E, self.E_u, nbr_samples)
        sigma_YS_sample = rng.normal(self.sigma_YS, self.sigma_YS_u, nbr_samples)

        return Specimen(W_sampled, S_sampled, B_sampled, B_N_sampled, a0_sampled, self.nu, E_sampled, self.eta_pl, sigma_YS_sample)
    
def compare_specimen(s1 : Specimen, s2 : Specimen) -> bool:
    """
    Compare two  specimen to check that they are similar

    Parameters
    ----------
    s1 : Specimen
        First specimen
    s2 : Specimen
        Second specimen

    Returns
    -------
    bool
        True if they are similar, False otherwise

    Warnings
    --------
    The initial crack length are different for each specimen and
    thus are not compared in this function.
    """
    return  s1.W ==         s2.W and \
            s1.S ==         s2.S and \
            s1.B ==         s2.B and \
            s1.B_N ==       s2.B_N and \
            s1.eta_pl ==    s2.eta_pl and \
            s1.sigma_YS ==  s2.sigma_YS and \
            s1.nu ==        s2.nu and \
            s1.E ==         s2.E

def log_specimen(specimen : Specimen, logger : Logger):
    """
    Log the specimen data

    Parameters
    ----------
    specimen : Specimen
        The specimen
    logger : Logger
        The logger
    """
    # Check that the specimen is not a sample
    if specimen.is_sample():
        print("ERROR: Can not log data from a sample specimen")
        raise ValueError("ERROR: Can not log data from a sample specimen")

    # -------------------------
    # Specimen geometry
    # -------------------------
    logger.log("\n--- Specimen geometry ---")
    logger.log(f" W       = {specimen.W:.3e} mm (specimen width)")
    logger.log(f" S       = {specimen.S:.3e} mm (span)")
    logger.log(f" B       = {specimen.B:.3e} mm (thickness)")
    logger.log(f" B_N     = {specimen.B_N:.3e} mm (net thickness)")
    logger.log(f" a0      = {specimen.a0:.3e} mm (initial crack length)")
    logger.log(f" b0      = {specimen.b0:.3e} mm (remaining ligament)")
    logger.log(f" f(a0/W) = {geometric_fnc_K(specimen.a0, specimen.W):.3e} (-) (geometric function)")
    logger.log(f" eta_pl  = {specimen.eta_pl:.3f} (-)")

    # -------------------------
    # Material properties
    # -------------------------
    logger.log("\n--- Material properties ---")
    logger.log(f" E        = {specimen.E:.3f} MPa (Young modulus)")
    logger.log(f" E'       = {specimen.E_plain_strain:.3f} MPa (Effective modulus in plain strain)")
    logger.log(f" nu       = {specimen.nu:.3f} (-) (Poisson ratio)")
    logger.log(f" K_Jc lim = {specimen.K_Jc_lim*10**-1.5:.6e} MPa·√m (Maximum K_Jc)")

def log_specimen_uncertainties(specimen : Specimen, specimen_mc : Specimen, logger : Logger):
    """
    Log the specimen data with the uncertainties

    Parameters
    ----------
    specimen : Specimen
        The specimen
    specimen_mc : Specimen
        A specimen sample
    logger : Logger
        The logger
    """
    # Check that the specimen is not a sample and that specimen_mc is a sample
    if specimen.is_sample():
        print("ERROR: Can not log data from a sample specimen")
        raise ValueError("ERROR: Can not log data from a sample specimen")
    if not specimen_mc.is_sample():
        print("ERROR: Can not log uncertainties from a non sample specimen")
        raise ValueError("ERROR: Can not log uncertainties from a non sample specimen")
    
    # -------------------------
    # Specimen geometry
    # -------------------------
    logger.log("\n--- Specimen geometry ---")
    logger.log(f" W       = {specimen.W:.3e} ± {compute_uncertainties(specimen_mc.W)[2]:.3e} mm (specimen width)")
    logger.log(f" S       = {specimen.S:.3e} ± {compute_uncertainties(specimen_mc.S)[2]:.3e} mm (span)")
    logger.log(f" B       = {specimen.B:.3e} ± {compute_uncertainties(specimen_mc.B)[2]:.3e} mm (thickness)")
    logger.log(f" B_N     = {specimen.B_N:.3e} ± {compute_uncertainties(specimen_mc.B_N)[2]:.3e} mm (net thickness)")
    logger.log(f" a0      = {specimen.a0:.3e} ± {compute_uncertainties(specimen_mc.a0)[2]:.3e} mm (initial crack length)")
    logger.log(f" b0      = {specimen.b0:.3e} ± {compute_uncertainties(specimen_mc.b0)[2]:.3e} mm (remaining ligament)")
    logger.log(f" f(a0/W) = {geometric_fnc_K(specimen.a0, specimen.W):.3e} ± {compute_uncertainties(geometric_fnc_K(specimen_mc.a0, specimen_mc.W))[2]:.3e} (-) (geometric function)")
    logger.log(f" eta_pl  = {specimen.eta_pl:.3f} (-)")

    # -------------------------
    # Material properties
    # -------------------------
    logger.log("\n--- Material properties ---")
    logger.log(f" E        = {specimen.E:.3f} ± {compute_uncertainties(specimen_mc.E)[2]:.3f} MPa (Young modulus)")
    logger.log(f" E'       = {specimen.E_plain_strain:.3f} ± {compute_uncertainties(specimen_mc.E_plain_strain)[2]:.3f} MPa (Effective modulus in plain strain)")
    logger.log(f" nu       = {specimen.nu:.3f} (-) (Poisson ratio)")
    logger.log(f" K_Jc lim = {specimen.K_Jc_lim*10**-1.5:.6e} ± {compute_uncertainties(specimen_mc.K_Jc_lim*10**-1.5)[2]:.6e} MPa·√m (Maximum K_Jc)")