from tools.Fracture import Fracture, log_fracture
from tools.Specimen import compare_specimen
from tools.MonteCarlo import compute_uncertainties
from tools.Logger import Logger
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

def single_T_master_curve_analysis(fractures : list[Fracture], T : float, B : float):
    K_min = 20
    Bx = 25.4
    nbr_uncencored_data = 0

    # Load data
    K_Jci = []
    K_Jc_lim = []
    for f in fractures:

        # Check limits
        if f.K_Jc < f.specimen.K_Jc_lim:
            nbr_uncencored_data += 1
            K_Jci.append(f.K_Jc)
        else:
            K_Jci.append(f.specimen.K_Jc_lim)

        K_Jc_lim.append(f.specimen.K_Jc_lim)

    # Check if it is possible to compute the master curve
    if nbr_uncencored_data == 0:
        print("ERROR: No K_Jc in acceptable limit. Impossible to compute master curve.")
        raise ValueError("ERROR: No K_Jc in acceptable limit. Impossible to compute master curve.")

    K_Jci = np.array(K_Jci)
    K_Jc_lim = np.array(K_Jc_lim)

    # size-adjusted to 1T thickness
    K_Jc1T = K_min + (K_Jci - K_min)*(B/Bx)**0.25

    # Compute T0 of the master curve temperature
    K0 = ((np.sum((K_Jc1T*10**-1.5 - K_min*10**-1.5)**4/nbr_uncencored_data))**0.25 + K_min*10**-1.5)/10**-1.5
    K_Jc_med =  K_min + 0.91*(K0 - K_min)
    T_0Q = T - (1/0.019)*np.log((K_Jc_med*10**-1.5 - 30)/70)

    # Check T0 validity
    valid_T0Q = T-T_0Q > -50 and  T-T_0Q < 50

    return K_Jci, K_Jc_lim, K_Jc1T, K0, K_Jc_med, T_0Q, valid_T0Q, nbr_uncencored_data

def master_curve(T : Union[float, np.ndarray], T0 : float):
    return 30 + 70*np.exp(0.019*(T - T0))

def master_curve_tolerance_bounds(T : float, T0 : float, percentile : float = 0.05):
    K_Jc_percentile = 20.0 + (np.log(1.0/(1.0 - percentile)))**0.25 * (11.0 + 77.0*np.exp(0.019*(T-T0)))
    T0_percentile = T - (1/0.019)*np.log((K_Jc_percentile - 30)/70)
    return K_Jc_percentile/10**-1.5, T0_percentile

class MasterCurve(object):
    def __init__(self, fractures : list[Fracture], T:float, percentile:float, fractures_mc : list[Fracture] = None):
        # Check that the fractures are not sampled and check that all specimen are similar
        specimen = None
        K_Jc_err_list = []
        for f in fractures:
            if f.is_sample:
                print("ERROR: Can not comput master curve with sampled fractures. Use experimental fracture only, not Monte-Carlo sampled fractures")
                raise ValueError("ERROR: Can not comput master curve with sampled fractures. Use experimental fracture only, not Monte-Carlo sampled fractures")
            
            if specimen is None:
                specimen = f.specimen
            elif not compare_specimen(specimen, f.specimen):
                print("ERROR: specimens are not all similar")
                raise ValueError("ERROR: specimens are not all similar")
            
        for f in fractures_mc:
            if not f.is_sample:
                print("ERROR: fractures_mc must be composed of sampled values")
                raise ValueError("ERROR: fractures_mc must be composed of sampled values")
            
            K_Jc_err_list.append(compute_uncertainties(f.K_Jc, percentile)[2])
            
        self.fractures = fractures
        self.T = T
        self.percentile = percentile
        self.K_Jc_err = np.array(K_Jc_err_list)
        self.K_Jci, self.K_Jc_lim, self.K_Jc1T, self.K0, self.K_Jc_med, self.T0, self.valid_T0, self.nbr_uncencored_data = single_T_master_curve_analysis(self.fractures, self.T, specimen.B)
        self.K_Jc_med_low, self.T0_low = master_curve_tolerance_bounds(self.T, self.T0, self.percentile/100)
        self.K_Jc_med_high, self.T0_high = master_curve_tolerance_bounds(self.T, self.T0, (100-self.percentile)/100)
        if not self.valid_T0:
            print("WARNING: The temperature computed for the MC is invalid")

    def plot_ld_curves(self):
        fig = plt.figure()
        ax = fig.subplots()
        ax.set_xlabel("$\Delta$ [mm]")
        ax.set_ylabel("$L$ [N]")
        for f in self.fractures:
            if f.test_nbr is not None:
                ax.plot(f.ld.disp, f.ld.load, label="Test " + str(f.test_nbr))
            else:
                ax.plot(f.ld.disp, f.ld.load)
        ax.legend()
        return fig, ax

    def plot_list_K(self, simulation_fracture : Union[Fracture, None] = None):
        bar_color = []
        for i in range(len(self.K_Jci)):
            if self.K_Jci[i] < self.K_Jc_lim[i]:
                bar_color.append("lightgreen")
            else:
                bar_color.append("salmon")

        bar_label = []
        for f in self.fractures:
            if f.test_nbr is None:
                bar_label.append("Test ??")
            else:
                bar_label.append("Test " + str(f.test_nbr))
        fig = plt.figure()
        ax = fig.subplots()
        p = ax.bar(bar_label, self.K_Jci*10**-1.5, yerr=self.K_Jc_err*10**-1.5, edgecolor = "black", color=bar_color, capsize=5)
        if simulation_fracture is not None:
            p2 = ax.bar("Abaqus", simulation_fracture.K_Jc*10**-1.5, edgecolor = "black", color="skyblue")
        ax.bar_label(p, label_type='center', rotation=90)
        if simulation_fracture is not None:
            ax.bar_label(p2, label_type='center', rotation=90)
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylabel("$K_{Jc}$ [MPa m$^{0.5}$]")
        return fig, ax

    def plot_master_curve(self, show_tolerance = False, show_errorbar = False):
        # Plot master curve
        T_master_curve = np.linspace(self.T0-100, self.T0 + 100, 1000)
        K_Jc_master_curve = master_curve(T_master_curve, self.T0)
        K_Jc_master_curve_005 = master_curve(T_master_curve, self.T0_low) #30 + 70*np.exp(0.019*(T_master_curve - T0_005))
        K_Jc_master_curve_095 = master_curve(T_master_curve, self.T0_high) #30 + 70*np.exp(0.019*(T_master_curve - T0_095))
        fig = plt.figure()
        ax = fig.subplots()
        ax.axvline(self.T0, color="black", linestyle="-", label="$T_0$")
        if show_tolerance:
            ax.axvline(self.T0_low, color="black", linestyle="--")
            ax.axvline(self.T0_high, color="black", linestyle="--")
        ax.plot(T_master_curve, K_Jc_master_curve, label="Master curve")
        if show_tolerance:
            ax.plot(T_master_curve, K_Jc_master_curve_005, linestyle="--", color="blue")
            ax.plot(T_master_curve, K_Jc_master_curve_095, linestyle="--", color="blue")
        if show_errorbar:
            ax.errorbar(np.zeros_like(self.K_Jc1T)+self.T, self.K_Jc1T*10**-1.5, np.array(self.K_Jc_err*10**-1.5), label="$K_{Jc(1T)}$", linestyle=" ", marker="x", capsize=5)
        else:
            ax.plot(np.zeros_like(self.K_Jc1T)+self.T, self.K_Jc1T*10**-1.5, label="$K_{Jc(1T)}$", linestyle=" ", marker="x")
        ax.set_xlabel("$T$ [°C]")
        ax.set_ylabel("$K_{Jc(med)}$")
        ax.legend()
        return fig, ax
    
def log_master_curve(mc : MasterCurve, logger : Logger, with_fracture : bool = False):
    logger.log("="*60)
    logger.log("Results for Master Curve")
    logger.log("="*60)

    # -------------------------
    # General info on the fracture tests
    # -------------------------
    logger.log("\n--- General fracture tests ---")
    logger.log(" Tests number:")
    for f in mc.fractures:
        info = "  > "
        if f.test_nbr is None:
            info += "Test ??"
        else:
            info += "Test " + str(f.test_nbr)
        if f.specimen.K_Jc_lim < f.K_Jc:
            info += " Censored"
        else:
            info += " Uncensored"
        logger.log(info)
    logger.log(f" Number of tests = {len(mc.fractures)}")
    logger.log(f" Number of uncensored = {mc.nbr_uncencored_data}")

    # -------------------------
    # Fracture details (in requiered)
    # -------------------------
    if with_fracture:
        for f in mc.fractures:
            log_fracture(f, logger)

    # -------------------------
    # Master curve infos
    # -------------------------
    logger.log("="*60)
    logger.log("Master curve")
    logger.log("="*60)

    logger.log(f" T = {mc.T:.3f} °C")
    logger.log(f" Confidence interval = {100-2*mc.percentile}%")
    logger.log(f" T0 = {mc.T0:.3f}, CI[{mc.T0_low:.3f}, {mc.T0_high:.3f}] °C")
    if mc.valid_T0:
        logger.log(f" Valid T0 = True")
    else:
        logger.log(f" Valid T0 = False")
    logger.log(f" K_Jc(med) = {mc.K_Jc_med*10**-1.5:.6e}, CI[{mc.K_Jc_med_low*10**-1.5:.3f}, {mc.K_Jc_med_high*10**-1.5:.3f}] MPa·√m")

    logger.log("="*60)