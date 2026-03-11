import numpy as np

from tools.Specimen import *
from tools.ElasticRegion import *
from tools.LoadDisplacement import *
from tools.Fracture import *
from tools.Logger import Logger

class FractureMC(object):
    def __init__(self, specimen : Specimen, elastic : ElasticRegion, ld : LoadDisplacement, id_computation : int):
        print(f"Running Monte Carlo simulation...")
        
        self.id_computation = id_computation
        self.P = ld.load[self.id_computation]

        self.load_computation = ld.load[self.id_computation]
        self.disp_computation = ld.disp[self.id_computation]
        self.disp_min = np.min(ld.disp)
        self.intercept_2 = -elastic.stiffness*self.disp_computation + self.load_computation
        self.conditionnal_area = ld.load[0] >= 1e-6

        # Compute area under load-disp curve (TODO Find a way to compute uncertainty on A_pl)
        self.A_pl = np.trapz(ld.load[:self.id_computation], ld.disp[:self.id_computation])

        # MC computation
        self.K_el = stress_intensity_factor(specimen, self.P)
        self.J_el = J_integral_el(specimen, self.K_el)
        self.J_pl = J_integral_pl(specimen, elastic, self.A_pl, self.load_computation, self.disp_min, self.conditionnal_area)
        self.J_c = self.J_el + self.J_pl
        self.K_Jc = np.sqrt(self.J_c*specimen.E_plain_strain)

        self.J_el_mean = np.mean(self.J_el)
        self.J_el_std = np.std(self.J_el)

        self.J_pl_mean = np.mean(self.J_pl)
        self.J_pl_std = np.std(self.J_pl)

        self.J_c_mean = np.mean(self.J_c)
        self.J_c_std = np.std(self.J_c)

        self.K_el_mean = np.mean(self.K_el)
        self.K_el_std = np.std(self.K_el)

        self.K_Jc_mean = np.mean(self.K_Jc)
        self.K_Jc_std = np.std(self.K_Jc)

        print("Monte Carlo simulation completed.")
    
    def plot_mc_results(self, bins: int = 30):
        # K histogram
        x = np.linspace(np.min(self.K_el), np.max(self.K_el), 1000)
        y = 1/(self.K_el_std * np.sqrt(2 * np.pi)) * np.exp( - (x - self.K_el_mean)**2 / (2 * self.K_el_std**2))
        fig1, ax1 = plt.subplots()
        ax1.hist(self.K_el, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        ax1.axvline(self.K_el_mean, color='red', linestyle='--', label=f'Mean: {self.K_el_mean:.3f}')
        ax1.axvline(self.K_el_mean + self.K_el_std, color='black', linestyle='--', label=f'Std: {self.K_el_std:.3f}')
        ax1.axvline(self.K_el_mean - self.K_el_std, color='black', linestyle='--')
        ax1.plot(x,y,color="red")
        ax1.set_xlabel("K [MPa·√mm]")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Stress Intensity Factor elastic ($K_{el}$)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # J_el histogram
        x = np.linspace(np.min(self.J_el), np.max(self.J_el), 1000)
        y = 1/(self.J_el_std * np.sqrt(2 * np.pi)) * np.exp( - (x - self.J_el_mean)**2 / (2 * self.J_el_std**2))
        fig2, ax2 = plt.subplots()
        ax2.hist(self.J_el, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        ax2.axvline(self.J_el_mean, color='red', linestyle='--', label=f'Mean: {self.J_el_mean:.3f}')
        ax2.axvline(self.J_el_mean + self.J_el_std, color='black', linestyle='--', label=f'Std: {self.J_el_std:.3f}')
        ax2.axvline(self.J_el_mean - self.J_el_std, color='black', linestyle='--')
        ax2.plot(x,y,color="red")
        ax2.set_xlabel("J [MPa·mm]")
        ax2.set_ylabel("Frequency")
        ax2.set_title("J-integral elastic ($J_{el}$)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # J_pl histogram
        x = np.linspace(np.min(self.J_pl), np.max(self.J_pl), 1000)
        y = 1/(self.J_pl_std * np.sqrt(2 * np.pi)) * np.exp( - (x - self.J_pl_mean)**2 / (2 * self.J_pl_std**2))
        fig3, ax3 = plt.subplots()
        ax3.hist(self.J_pl, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        ax3.axvline(self.J_pl_mean, color='red', linestyle='--', label=f'Mean: {self.J_pl_mean:.3f}')
        ax3.axvline(self.J_pl_mean + self.J_pl_std, color='black', linestyle='--', label=f'Std: {self.J_pl_std:.3f}')
        ax3.axvline(self.J_pl_mean - self.J_pl_std, color='black', linestyle='--')
        ax3.plot(x,y,color="red")
        ax3.set_xlabel("J [MPa·mm]")
        ax3.set_ylabel("Frequency")
        ax3.set_title("J-integral plastic ($J_{pl}$)")
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.5)

        # J histogram
        x = np.linspace(np.min(self.J_c), np.max(self.J_c), 1000)
        y = 1/(self.J_c_std * np.sqrt(2 * np.pi)) * np.exp( - (x - self.J_c_mean)**2 / (2 * self.J_c_std**2))
        fig4, ax4 = plt.subplots()
        ax4.hist(self.J_c, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        ax4.axvline(self.J_c_mean, color='red', linestyle='--', label=f'Mean: {self.J_c_mean:.3f}')
        ax4.axvline(self.J_c_mean + self.J_c_std, color='black', linestyle='--', label=f'Std: {self.J_c_std:.3f}')
        ax4.axvline(self.J_c_mean - self.J_c_std, color='black', linestyle='--')
        ax4.plot(x,y,color="red")
        ax4.set_xlabel("J [MPa·mm]")
        ax4.set_ylabel("Frequency")
        ax4.set_title("J-integral ($J$)")
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.5)

def log_fracture_with_uncertainties(logger : Logger, fracture : Fracture, uncertainties : FractureMC, specimen_u : SpecimenDistribution, elastic_u : ElasticRegionDistribution):
    logger.log("="*60)
    logger.log("Results for fracture: ")
    logger.log("="*60)

    # -------------------------
    # Specimen geometry
    # -------------------------
    logger.log("\n--- Specimen geometry ---")
    logger.log(f" W       = {specimen_u.W} pm {specimen_u.W_u} mm (specimen width)")
    logger.log(f" S       = {specimen_u.S} pm {specimen_u.S_u} mm (span)")
    logger.log(f" B       = {specimen_u.B} pm {specimen_u.B_u} mm (thickness)")
    logger.log(f" B_N     = {specimen_u.B_N} pm {specimen_u.B_N_u} mm (net thickness)")
    logger.log(f" a0      = {fracture.specimen.a0} mm (initial crack length)")
    logger.log(f" b0      = {fracture.specimen.b0} mm (remaining ligament)")
    logger.log(f" f(a0/W) = {geometric_fnc_K(fracture.specimen.a0, specimen_u.W)} (-) (geometric function)")
    logger.log(f" eta_pl = {specimen_u.eta_pl:.3f} (-)")

    # -------------------------
    # Material properties
    # -------------------------
    logger.log("\n--- Material properties ---")
    logger.log(f" E      = {specimen_u.E:.3f} pm {specimen_u.E_u:.3f} MPa (Young modulus)")
    logger.log(f" E'     = {fracture.specimen.E_plain_strain:.3f} MPa (Effective modulus in plain strain)")
    logger.log(f" nu     = {specimen_u.nu:.3f} (-) (Poisson ratio)")

    # -------------------------
    # Elastic region detection
    # -------------------------
    logger.log("\n--- Elastic region detection ---")
    logger.log(f" Yield load         = {fracture.ld.load[elastic_u.id_end]} N")
    logger.log(f" Yield displacement = {fracture.ld.disp[elastic_u.id_end]} mm")
    logger.log(f" Elastic end index  = {elastic_u.id_end}")
    logger.log(f" Stiffness (slope)  = {elastic_u.stiffness:.6f} pm {elastic_u.stiffness_u} N/mm")
    logger.log(f" Intercept 1        = {elastic_u.intercept:.6f} pm {elastic_u.intercept_u} (Interception of y-axis for elastic region)")
    logger.log(f" Intercept 2        = {fracture.intercept_2:.6f} (Interception of y-axis for computation point)")

    # -------------------------
    # Load at computation point
    # -------------------------
    logger.log("\n--- Computation point ---")
    logger.log(f" Index used        = {elastic_u.id_end}")
    logger.log(f" Load P            = {fracture.ld.load[elastic_u.id_end]:.3f} N")

    # -------------------------
    # Fracture parameters
    # -------------------------
    logger.log("\n--- Fracture parameters ---")
    logger.log(f" J_el   = {fracture.J_el:.6f} pm {uncertainties.J_el_std:.6f} MPa mm,        {fracture.J_el*1e-3:.6f} pm {uncertainties.J_el_std*1e-3:.6f} MPa m")
    logger.log(f" J_pl   = {fracture.J_pl:.6f} pm {uncertainties.J_pl_std:.6f} MPa mm,        {fracture.J_pl*1e-3:.6f} pm {uncertainties.J_pl_std*1e-3:.6f} MPa m")
    logger.log(f" J_c    = {fracture.J_c:.6f} pm {uncertainties.J_c_std:.6f} MPa mm,        {fracture.J_c*1e-3:.6f} pm {uncertainties.J_c_std*1e-3:.6f} MPa m")
    logger.log(f" K_el   = {fracture.K_el:.6f} pm {uncertainties.K_el_std:.6f} MPa mm^0.5, {fracture.K_el*np.sqrt(1e-3):.6f} pm {uncertainties.K_el_std*np.sqrt(1e-3):.6f} MPa m^0.5")
    logger.log(f" K_Jc   = {fracture.K_Jc:.6f} pm {uncertainties.K_Jc_std:.6f} MPa mm^0.5, {fracture.K_Jc*np.sqrt(1e-3):.6f} pm {uncertainties.K_Jc_std*np.sqrt(1e-3):.6f} MPa m^0.5")

    logger.log("="*60)