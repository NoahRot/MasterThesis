"""
A class that is used to analyze load displacement data
"""

import numpy as np
import scipy as sci
import os
import matplotlib.pyplot as plt
from tools.load_disp_tool import *
from tools.plt_spec import *

"""
A class that is used to analyze the load displacement curve.
Input-Parameters:
 - path (str) : path to the data
 - W (float) : width of the specimen [micron]
 - S (float) : span [micron]
 - B (float) : thickness [micron]
 - B_N (float) : net thickness [micron]
 - a0 (float) : initial crack length [micron]
 - nu (float): Poisson ratio [-]
 - E (float): Young modulus [MPa]
 - eta_pl (float): ??? parameters to compute the J-elastic
 - id_computation (int): index of the computation point
 - min_point (int): minimum number of points used to determin the elastic region
 - r2_threshold (float): R^2 threshold used to determin the elastic region
 - file_type (float): Type of file to analyze ('abaqus' or 'experiment')
Parameters:
 - b0 (float): remaining ligament (W - a0)
 - t (array[float]): time array
 - load (array[float]): load array
 - disp (arra[float]): displacement array
 - P (float): load at the computation point
 - id_elastic_end (int): index of the end of the elastic limit
 - stiffness (float): stiffness approximate with linear regression in the elastic region
 - intercept_1 (float): interception value for the linear curve in the elastic region
 - intercept_2 (float): interception value for the linear curve passing by the computation point
 - K (float): stress intensity factor
 - J_el (float): elastic part of the J-integral
 - J_pl (float): plastic part of the J-integral
 - J (float): J-integral
 - fig (Figure): figure (detailed). Only exist if the method plot_details has been called.  
"""
class Analyzer_LD:
    def __init__(self, path:str, W:float, S:float, B:float, B_N:float, a0:float, nu:float, E:float, eta_pl:float, id_computation:int = -1, min_point:int = 3, r2_threshold = 0.995, file_type : str = "abaqus"):
        # Physical parameters
        self.path = path
        self.W = W
        self.S = S
        self.B = B
        self.B_N = B_N
        self.a0 = a0
        self.nu = nu
        self.E = E
        self.eta_pl = eta_pl
        self.b0 = W - a0

        # Numerical parameters
        self.id_computation = id_computation
        self.min_point = min_point
        self.r2_threshold = r2_threshold
        self.file_type = file_type

        # Set matplotlib specification
        create_sns_palette()
        init_plt(latex=False)

        # Load data
        self.t, self.load, self.disp = load_load_disp_data(self.path, self.file_type)
        self.P = self.load[id_computation]

        # Determine elastic region
        self.id_elastic_end, self.stiffness, self.intercept_1 = elastic_region_determination_r2_method(self.load, self.disp, self.min_point, self.r2_threshold)
        self.intercept_2 = -self.stiffness*self.disp[self.id_computation] + self.load[self.id_computation]

        # Compute stress intensity foactor and J-integral
        self.K = stress_intensity_factor(self.P, self.S, self.B, self.B_N, self.W, self.a0)
        self.J_el = J_integral_el(self.K, self.nu, self.E)
        self.J_pl = J_integral_pl(self.eta_pl, self.B_N, self.b0, self.id_computation, self.load, self.disp, self.stiffness)
        self.J = self.J_el + self.J_pl 

        # Figures
        self.fig = None

    def print_all(self):
        print("="*60)
        print("Results for data: " + os.path.abspath(self.path))
        print("="*60)

        # -------------------------
        # Specimen geometry
        # -------------------------
        print("\n--- Specimen geometry ---")
        print(f" W   = {self.W:.3e} µm (specimen width)")
        print(f" S   = {self.S:.3e} µm (span)")
        print(f" B   = {self.B:.3e} µm (thickness)")
        print(f" B_N = {self.B_N:.3e} µm (net thickness)")
        print(f" a0  = {self.a0:.3e} µm (initial crack length)")
        print(f" b0  = {self.b0:.3e} µm (remaining ligament)")

        # -------------------------
        # Material properties
        # -------------------------
        print("\n--- Material properties ---")
        print(f" E  = {self.E:.3f} MPa (Young modulus)")
        print(f" nu = {self.nu:.3f} (-) (Poisson ratio)")
        print(f" eta_pl = {self.eta_pl:.3f} (-)")

        # -------------------------
        # Elastic region detection
        # -------------------------
        print("\n--- Elastic region detection ---")
        print(f" Yield load         = {self.load[self.id_elastic_end]*1e-6} N")
        print(f" Yield displacement = {self.disp[self.id_elastic_end]} micron")
        print(f" Elastic end index  = {self.id_elastic_end}")
        print(f" Stiffness (slope)  = {self.stiffness:.6e} N/m")
        print(f" Intercept 1        = {self.intercept_1:.6e}")
        print(f" Intercept 2        = {self.intercept_2:.6e}")
        print(f" R² threshold used  = {self.r2_threshold}")

        # -------------------------
        # Load at computation point
        # -------------------------
        print("\n--- Computation point ---")
        print(f" Index used        = {self.id_computation}")
        print(f" Load P            = {self.P:.6e} µN")

        # -------------------------
        # Fracture parameters
        # -------------------------
        print("\n--- Fracture parameters ---")
        print(f" K      = {self.K:.6e} MPa·√µm")
        print(f" J_el   = {self.J_el:.6e} MPa·µm")
        print(f" J_pl   = {self.J_pl:.6e} MPa·µm")
        print(f" J_tot  = {self.J:.6e} MPa·µm")

        print("="*60)

    def report(self, report_name):
        try:
            with open(report_name, "w") as f:

                f.write("="*60 + "\n")
                f.write("Results for data: " + os.path.abspath(self.path) + "\n")
                f.write("="*60 + "\n")

                # -------------------------
                # Specimen geometry
                # -------------------------
                f.write("\n--- Specimen geometry ---\n")
                f.write(f" W   = {self.W:.3e} micron (specimen width)\n")
                f.write(f" S   = {self.S:.3e} micron (span)\n")
                f.write(f" B   = {self.B:.3e} micron (thickness)\n")
                f.write(f" B_N = {self.B_N:.3e} micron (net thickness)\n")
                f.write(f" a0  = {self.a0:.3e} micron (initial crack length)\n")
                f.write(f" b0  = {self.b0:.3e} micron (remaining ligament)\n")

                # -------------------------
                # Material properties
                # -------------------------
                f.write("\n--- Material properties ---\n")
                f.write(f" E  = {self.E:.3f} MPa (Young modulus)\n")
                f.write(f" nu = {self.nu:.3f} (-) (Poisson ratio)\n")
                f.write(f" eta_pl = {self.eta_pl:.3f} (-)\n")

                # -------------------------
                # Elastic region detection
                # -------------------------
                f.write("\n--- Elastic region detection ---\n")
                f.write(f" Yield load         = {self.load[self.id_elastic_end]*1e-6} N\n")
                f.write(f" Yield displacement = {self.disp[self.id_elastic_end]} micron\n")
                f.write(f" Elastic end index  = {self.id_elastic_end}\n")
                f.write(f" Stiffness (slope)  = {self.stiffness:.6e} N/m\n")
                f.write(f" Intercept 1        = {self.intercept_1:.6e}\n")
                f.write(f" Intercept 2        = {self.intercept_2:.6e}\n")
                f.write(f" R^2 threshold used  = {self.r2_threshold}\n")

                # -------------------------
                # Load at computation point
                # -------------------------
                f.write("\n--- Computation point ---\n")
                f.write(f" Index used        = {self.id_computation}\n")
                f.write(f" Load P            = {self.P:.6e}  micro N\n")

                # -------------------------
                # Fracture parameters
                # -------------------------
                f.write("\n--- Fracture parameters ---\n")
                f.write(f" K      = {self.K:.6e} MPa micron^0.5\n")
                f.write(f" J_el   = {self.J_el:.6e} MPa micron^0.5\n")
                f.write(f" J_pl   = {self.J_pl:.6e} MPa micron^0.5\n")
                f.write(f" J_tot  = {self.J:.6e} MPa micron^0.5\n")

                f.write("="*60 + "\n")

            print(f"Report successfully written to {report_name}")

        except Exception as e:
            print(f"ERROR: Cannot create report file {report_name}")
            print(f"Reason: {e}")

    def plot_details(self):
        # Define the stiffness curves
        x1 = np.linspace(0, (np.max(self.load)-self.intercept_1)/self.stiffness, 2)
        x2 = np.linspace(-self.intercept_2/self.stiffness, self.disp[self.id_computation], 2)
        y1 = self.stiffness*x1 + self.intercept_1
        y2 = self.stiffness*x2 + self.intercept_2

        # Define area curves
        x = np.copy(self.disp)
        y_up = np.copy(self.load)
        x_c = -self.intercept_2/self.stiffness
        idx = np.searchsorted(self.disp, x_c)
        y_c = y_up[idx-1] + (y_up[idx]-y_up[idx-1])/(x[idx]-x[idx-1])*(x_c-x[idx-1])
        x = np.insert(x, idx, x_c)
        y_up = np.insert(y_up, idx, y_c)
        y_down = self.stiffness*x + self.intercept_2
        y_down = np.maximum(y_down, np.zeros_like(y_down))

        # Plot the results
        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(self.disp, self.load, "bx", label="$L-\Delta$ curve")
        ax.plot(self.disp[self.id_computation], self.load[self.id_computation], "rx", label="Computation point")
        ax.vlines(self.disp[self.id_elastic_end], np.min(self.load), np.max(self.load), color="k", linestyles=":", label="End of elastic region")
        ax.plot(x1, y1, "k--", label="Stiffness")
        ax.plot(x2, y2, "k--")
        ax.fill_between(x, y_down, y_up, alpha=0.2, label="$A_{pl}$")

        #ax.plot(-self.intercept_1/self.stiffness, 0.0, "kx")
        #ax.plot(-self.intercept_2/self.stiffness, 0.0, "kx")
        #ax.plot(self.disp[self.id_computation], self.load[self.id_computation], "kx")
        #ax.plot(self.disp[self.id_computation], 0.0, "kx")

        ax.set_xlabel("$\Delta$ [μm]")
        ax.set_ylabel("$L$ [μN]")
        ax.set_title("$L-\Delta$ curve")
        ax.legend()

        self.fig = fig

    def plot(self, curve_name = None, ax = None):
        if (ax == None):
            fig = plt.figure()
            ax = fig.subplots()
            ax.set_xlabel("$\Delta$ [μm]")
            ax.set_ylabel("$L$ [μN]")
            ax.set_title("$L-\Delta$ curve")

        if (curve_name == None):
            ax.plot(self.disp, self.load)
        else:
            ax.plot(self.disp, self.load, label=curve_name)

        return ax
    
    def plot_time(self):
        fig = plt.figure()
        ax = fig.subplots(2,2)
        ax[0,0].plot(self.disp,self.load)
        ax[0,0].set_xlabel("$\Delta$ [μm]")
        ax[0,0].set_ylabel("$L$ [μN]")

        ax[0,1].plot(self.t, self.disp)
        ax[0,1].set_xlabel("$t$ [t]")
        ax[0,1].set_ylabel("$\Delta$ [μm]")

        ax[1,0].plot(self.t, self.load)
        ax[1,0].set_xlabel("$t$ [t]")
        ax[1,0].set_ylabel("$L$ [μN]")
    
    def savefig(self, name):
        if self.fig != None:
            self.fig.savefig(name)
            print(f"Figure successfully saved to {name}")