import numpy as np

from tools.Specimen import *
from tools.ElasticRegion import *
from tools.LoadDisplacement import *
from tools.Fracture import *

class FractureMC(object):
    def __init__(self, specimen : Specimen, elastic : ElasticRegion, ld : LoadDisplacement, id_computation : int, nbr_mc : int):
        print(f"Running Monte Carlo simulation with {nbr_mc} iterations...")
        
        self.id_computation = id_computation
        self.P = ld.load[self.id_computation]

        load, disp = ld.get_LD_sorted()
        self.load_computation = load[self.id_computation]
        self.disp_computation = disp[self.id_computation]
        self.disp_min = np.min(disp)
        self.intercept_2 = -elastic.stiffness*self.disp_computation + self.load_computation
        self.conditionnal_area = ld.load[0] >= 1e-6

        # Compute area under load-disp curve (TODO Find a way to compute uncertainty on A_pl)
        self.A_pl = np.trapz(load[:self.id_computation], disp[:self.id_computation])

        # MC computation
        self.K = stress_intensity_factor(specimen, self.P)
        self.J_el = J_integral_el(specimen, self.K)
        self.J_pl = J_integral_pl(specimen, elastic, self.A_pl, self.load_computation, self.disp_min, self.conditionnal_area)

        self.J_el_mean = np.mean(self.J_el)
        self.J_el_std = np.std(self.J_el)

        self.J_pl_mean = np.mean(self.J_pl)
        self.J_pl_std = np.std(self.J_pl)

        self.J = self.J_el + self.J_pl
        self.J_mean = np.mean(self.J)
        self.J_std = np.std(self.J)

        self.K_mean = np.mean(self.K)
        self.K_std = np.std(self.K)

        print("Monte Carlo simulation completed.")

#    def stress_intensity_factor(self) -> float:
#        f_geom = geometric_fnc_K(self.a0, self.W)
#        self.K = self.P*self.S/(np.sqrt(self.B*self.B_N) * self.W**1.5) * f_geom
#
#    def J_integral_el(self) -> float:
#        self.J_el =  self.K**2 * (1.0 - self.nu**2) / self.E
#
#    def J_integral_pl(self) -> float:
#
#        # Linear regression in the elastic region
#        intercept_2 = -self.stiffness*self.disp_computation + self.load_computation
#
#        # Add the rest of the area using stiffness if needed
#        A_pl = self.A_pl
#        if self.conditionnal_area:
#            x0 = -self.intercept/self.stiffness
#            x1 = self.disp_min
#            y1 = self.stiffness*x1 + self.intercept
#            A_pl += 0.5*(x1 - x0)*y1
#
#        # Remove the triangle area below the index_computation
#        x0 = -intercept_2/self.stiffness
#        x1 = self.disp_computation
#        y1 = self.load_computation
#        A_pl -= 0.5*(x1 - x0)*y1
#
#        self.J_pl = self.eta_pl*A_pl/(self.B_N*self.b0)
    
    def plot_mc_results(self, bins: int = 30):
        # K histogram
        fig1, ax1 = plt.subplots()
        ax1.hist(self.K, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(self.K_mean, color='red', linestyle='--', label=f'Mean: {self.K_mean:.3f}')
        ax1.axvline(self.K_mean + self.K_std, color='black', linestyle='--', label=f'Std: {self.K_std:.3f}')
        ax1.axvline(self.K_mean - self.K_std, color='black', linestyle='--')
        ax1.set_xlabel("K [MPa·√mm]")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Stress Intensity Factor ($K$)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # J_el histogram
        fig2, ax2 = plt.subplots()
        ax2.hist(self.J_el, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(self.J_el_mean, color='red', linestyle='--', label=f'Mean: {self.J_el_mean:.3f}')
        ax2.axvline(self.J_el_mean + self.J_el_std, color='black', linestyle='--', label=f'Std: {self.J_el_std:.3f}')
        ax2.axvline(self.J_el_mean - self.J_el_std, color='black', linestyle='--')
        ax2.set_xlabel("J [MPa·mm]")
        ax2.set_ylabel("Frequency")
        ax2.set_title("J-integral elastic ($J_{el}$)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # J_pl histogram
        fig3, ax3 = plt.subplots()
        ax3.hist(self.J_pl, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.axvline(self.J_pl_mean, color='red', linestyle='--', label=f'Mean: {self.J_pl_mean:.3f}')
        ax3.axvline(self.J_pl_mean + self.J_pl_std, color='black', linestyle='--', label=f'Std: {self.J_pl_std:.3f}')
        ax3.axvline(self.J_pl_mean - self.J_pl_std, color='black', linestyle='--')
        ax3.set_xlabel("J [MPa·mm]")
        ax3.set_ylabel("Frequency")
        ax3.set_title("J-integral plastic ($J_{pl}$)")
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.5)

        # J histogram
        fig4, ax4 = plt.subplots()
        ax4.hist(self.J, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        ax4.axvline(self.J_mean, color='red', linestyle='--', label=f'Mean: {self.J_mean:.3f}')
        ax4.axvline(self.J_mean + self.J_std, color='black', linestyle='--', label=f'Std: {self.J_std:.3f}')
        ax4.axvline(self.J_mean - self.J_std, color='black', linestyle='--')
        ax4.set_xlabel("J [MPa·mm]")
        ax4.set_ylabel("Frequency")
        ax4.set_title("J-integral ($J$)")
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.5)

def report_with_uncertainties(report_name : str, fracture : Fracture, uncertainties : FractureMC, specimen_u : SpecimenDistribution, elastic_u : ElasticRegionDistribution):
    try:
        with open(report_name, "w") as f:

            f.write("="*60 + "\n")
            f.write("Fracture report\n")
            f.write("="*60 + "\n")

            # -------------------------
            # Specimen geometry
            # -------------------------
            f.write("\n--- Specimen geometry ---\n")
            f.write(f" W   = {specimen_u.W:.3e} pm {specimen_u.W_u} mm (specimen width)\n")
            f.write(f" S   = {specimen_u.S:.3e} pm {specimen_u.S_u:.3e} mm (span)\n")
            f.write(f" B   = {specimen_u.B:.3e} pm {specimen_u.B_u:.3e} mm (thickness)\n")
            f.write(f" B_N = {specimen_u.B_N:.3e} pm {specimen_u.B_N_u:.3e} mm (net thickness)\n")
            f.write(f" a0  = {specimen_u.a0:.3e} pm {specimen_u.a0_u:.3e} mm (initial crack length)\n")
            f.write(f" b0  = {fracture.specimen.b0:.3e} mm (remaining ligament)\n")

            # -------------------------
            # Material properties
            # -------------------------
            f.write("\n--- Material properties ---\n")
            f.write(f" E  = {specimen_u.E:.3f} pm {specimen_u.E_u:.3f} MPa (Young modulus)\n")
            f.write(f" nu = {specimen_u.nu:.3f} (-) (Poisson ratio)\n")
            f.write(f" eta_pl = {specimen_u.eta_pl:.3f} (-)\n")

            # -------------------------
            # Elastic region detection
            # -------------------------
            intercept_2 = -fracture.elastic.stiffness*fracture.ld.disp[fracture.id_computation] + fracture.ld.load[fracture.id_computation]
            f.write("\n--- Elastic region detection ---\n")
            f.write(f" Yield load         = {fracture.ld.load[fracture.elastic.id_end]:.3e} N\n")
            f.write(f" Yield displacement = {fracture.ld.disp[fracture.elastic.id_end]:.3e} mm\n")
            f.write(f" Elastic end index  = {fracture.elastic.id_end}\n")
            f.write(f" Stiffness (slope)  = {fracture.elastic.stiffness:.6e} pm {elastic_u.stiffness_u} N/mm\n")
            f.write(f" Intercept 1        = {fracture.elastic.intercept:.6e} pm {elastic_u.intercept_u}\n")
            f.write(f" Intercept 2        = {intercept_2:.6e}\n")

            # -------------------------
            # Load at computation point
            # -------------------------
            f.write("\n--- Computation point ---\n")
            f.write(f" Index used        = {fracture.id_computation}\n")
            f.write(f" Load P            = {fracture.ld.load[fracture.id_computation]:.6e} N\n")

            # -------------------------
            # Fracture parameters
            # -------------------------
            f.write("\n--- Fracture parameters ---\n")
            f.write(f" K      = {fracture.K:.6e} pm {uncertainties.K_std:.6e} MPa mm^0.5, {fracture.K*np.sqrt(1e-3):.6e} MPa m^0.5\n")
            f.write(f" J_el   = {fracture.J_el:.6e} pm {uncertainties.J_el_std:.6e} MPa mm^0.5\n")
            f.write(f" J_pl   = {fracture.J_pl:.6e} pm {uncertainties.J_pl_std:.6e} MPa mm^0.5\n")
            f.write(f" J_tot  = {fracture.J:.6e} pm {uncertainties.J_std:.6e} MPa mm^0.5\n")

            f.write("="*60 + "\n")

        print(f"Report successfully written to {report_name}")

    except Exception as e:
        print(f"ERROR: Cannot create report file {report_name}")
        print(f"Reason: {e}")