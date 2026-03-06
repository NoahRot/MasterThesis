import numpy as np

from tools.Specimen import *
from tools.ElasticRegion import *
from tools.LoadDisplacement import *
from tools.Fracture import *

class FractureMC(object):
    def __init__(self, specimen : SpecimenUncertainties, elastic : ElasticRegionUncertainties, ld : LoadDisplacement, id_computation : int, nbr_mc : int):
        self.id_computation = id_computation
        self.P = ld.load[self.id_computation]

        rng = np.random.default_rng()

        self.a0 = rng.normal(specimen.a0, specimen.a0_u, nbr_mc)
        self.W = rng.normal(specimen.W, specimen.W_u, nbr_mc)
        self.S = rng.normal(specimen.S, specimen.S_u, nbr_mc)
        self.B = rng.normal(specimen.B, specimen.B_u, nbr_mc)
        self.B_N = rng.normal(specimen.B_N, specimen.B_N_u, nbr_mc)
        self.nu = specimen.nu
        self.E = rng.normal(specimen.E, specimen.E_u, nbr_mc)
        self.eta_pl = specimen.eta_pl
        self.b0 = self.W - self.a0

        self.stiffness = rng.normal(elastic.stiffness, elastic.stiffness_u, nbr_mc)
        self.intercept = rng.normal(elastic.intercept, elastic.intercept_u, nbr_mc)

        load, disp = ld.get_LD_sorted()
        self.load_computation = load[self.id_computation]
        self.disp_computation = disp[self.id_computation]
        self.disp_min = np.min(disp)
        self.conditionnal_area = ld.load[0] >= 1e-6

        # Compute area under load-disp curve
        self.A_pl = np.trapz(load[:self.id_computation], disp[:self.id_computation])

        # MC computation
        self.stress_intensity_factor()
        self.J_integral_el()
        self.J_integral_pl()

    def stress_intensity_factor(self) -> float:
        f_geom = geometric_fnc_K(self.a0, self.W)
        self.K = self.P*self.S/(np.sqrt(self.B*self.B_N) * self.W**1.5) * f_geom
        return self.K

    def J_integral_el(self) -> float:
        self.J_el =  self.K**2 * (1.0 - self.nu**2) / self.E
        return self.J_el

    def J_integral_pl(self) -> float:

        # Linear regression in the elastic region
        intercept_2 = -self.stiffness*self.disp_computation + self.load_computation

        # Add the rest of the area using stiffness if needed
        A_pl = self.A_pl
        if self.conditionnal_area:
            x0 = -self.intercept/self.stiffness
            x1 = self.disp_min
            y1 = self.stiffness*x1 + self.intercept
            A_pl += 0.5*(x1 - x0)*y1

        # Remove the triangle area below the index_computation
        x0 = -intercept_2/self.stiffness
        x1 = self.disp_computation
        y1 = self.load_computation
        A_pl -= 0.5*(x1 - x0)*y1

        self.J_pl = self.eta_pl*A_pl/(self.B_N*self.b0)
        return self.J_pl
    
    def plot_mc_results(self, bins: int = 30):
        """
        Plot histograms of Monte Carlo results for fracture parameters.
        
        Parameters:
            K_samples (np.ndarray): Array of stress intensity factor K from MC.
            J_el_samples (np.ndarray): Array of elastic J-integral values from MC.
            J_pl_samples (np.ndarray): Array of plastic J-integral values from MC.
            bins (int): Number of bins for the histograms.
        """
        K_samples = self.K
        J_el_samples = self.J_el
        J_pl_samples = self.J_pl
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # K histogram
        axes[0].hist(K_samples, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(K_samples), color='red', linestyle='--', label=f'Mean: {np.mean(K_samples):.3f}')
        axes[0].set_xlabel("K [MPa·√mm]")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Stress Intensity Factor (K)")
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # J_el histogram
        axes[1].hist(J_el_samples, bins=bins, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(J_el_samples), color='red', linestyle='--', label=f'Mean: {np.mean(J_el_samples):.3f}')
        axes[1].set_xlabel("J_el [MPa·mm]")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Elastic J-integral (J_el)")
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        # J_pl histogram
        axes[2].hist(J_pl_samples, bins=bins, color='salmon', edgecolor='black', alpha=0.7)
        axes[2].axvline(np.mean(J_pl_samples), color='red', linestyle='--', label=f'Mean: {np.mean(J_pl_samples):.3f}')
        axes[2].set_xlabel("J_pl [MPa·mm]")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("Plastic J-integral (J_pl)")
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.5)
