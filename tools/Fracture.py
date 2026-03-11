from tools.LoadDisplacement import *
from tools.Specimen import *
from tools.ElasticRegion import *
import os

"""
Compute the geometric function for the stress intensity factor computation
Input:
- a0 (float): initial crack length
- W (float): width of the specimen
Output:
 - (float): geometric function
"""
def geometric_fnc_K(a0 : float, W : float) -> float:
    r = a0/W
    return 3.0*np.sqrt(r)*(1.99 - r*(1.0-r)*(2.15 - 3.93*r + 2.7*r**2))/(2.0*(1.0 + 2.0*r)*(1.0-r)**1.5)

"""
Compute the stress intensity factor
Input:
 - specimen (Specimen): The specimen to study
 - P (float/array): The load at the computationi point
Output:
 - (float) stress intensity factor
"""
def stress_intensity_factor(specimen : Specimen, P) -> float:
    f_geom = geometric_fnc_K(specimen.a0, specimen.W)
    K = P*specimen.S/(np.sqrt(specimen.B*specimen.B_N) * specimen.W**1.5) * f_geom
    return K

"""
Compute the elastic J-integral
Input:
 - specimen (Specimen): The specimen to study
 K (float/array): stress intensity factor
Output:
 - (float) J-integral elastic
"""
def J_integral_el(specimen : Specimen, K) -> float:
    J_el = K**2 * (1.0 - specimen.nu**2) / specimen.E
    return J_el

"""
Compute the plastic J-integral
Input:
 - specimen (Specimen): The specimen to study
 - elastic (ElasticRegion): Elastic region of the test
 - A_total (float/array): Area under the LD curve
 - disp_computation (float/array): displacement at computation point
 - load_computation (float/array): load at computation point
 - intercept_2 (float/array): intercept point for the second stiffness curve
 - min_disp (float/array): minimum dispalcement of the test
 - conditional_area (float/array): should the conditional area be computed 
Output:
 - (float) J-integral elastic
"""
def J_integral_pl(specimen : Specimen, elastic : ElasticRegion, A_total, load_computation, min_disp, conditional_area) -> float:
    # Add the rest of the area using stiffness if needed
    A_pl = A_total
    if conditional_area:
        A_pl += 0.5*elastic.stiffness*min_disp**2

    # Remove the triangle area below the index_computation
    A_pl -= 0.5*load_computation**2/elastic.stiffness

    J_pl = specimen.eta_pl*A_pl/(specimen.B_N*specimen.b0)
    return J_pl

"""
A class that represent the fracture test.
Input-Parameters:
 - specimen (Specimen): The tested specimen 
 - elastic (ElasticRegion): The elastic region of the test
 - ld (LoadDispalcement): The load-displacement data
 - id_computation (int): The index of the computed point
Parameters:
 - K (float): The stress intensity factor
 - J_el (float): The elastic part of the J-integral
 - J_pl (float): The plastic part of the J-integral
 - J (float): The J-integral
"""
class Fracture(object):
    def __init__(self, specimen : Specimen, elastic : ElasticRegion, ld : LoadDisplacement, id_computation : int):
        self.specimen = specimen
        self.elastic = elastic
        self.ld = ld
        self.id_computation = id_computation

        load, disp = self.ld.get_LD_sorted()                                            # Get the sorted load-displacement
        self.A_total = np.trapz(load[:self.id_computation+1], disp[:self.id_computation+1]) # Compute the Area under the LD curve
        self.disp_computation = disp[id_computation]                                    # Get the displacement at the computation point
        self.load_computation = load[id_computation]                                    # Get the load at the computation point
        self.intercept_2 = -self.elastic.stiffness*disp[self.id_computation] + load[self.id_computation] # Get the intercept point for the second stiffness curve
        self.min_disp = np.min(disp)                                                    # Get the minimum value of the displacement
        self.conditional_area = self.ld.load[0] >= 1e-6                                 # Condition if additionnal area should be computed

        self.K = stress_intensity_factor(specimen, self.load_computation)
        self.J_el = J_integral_el(specimen, self.K)
        self.J_pl = J_integral_pl(specimen, elastic, self.A_total, self.load_computation, self.min_disp, self.conditional_area)
        self.J = self.J_el + self.J_pl
    
    """
    Create a detailed plot of the load-displacement curve with all the relevant point curve and information
    Input:
     - save_fig (bool): Do you want to save the figure
     - fig_name (str): path and name of the figure
    Output:
     - fig (Figure): The figure
     - ax (Axes): The axes used for the plot
    """
    def plot_details(self, save_fig : bool = False, fig_name : str = None):
        load, disp = self.ld.get_LD_sorted()
        #load, disp = self.ld.load, self.ld.disp

        fig = plt.figure()
        ax = fig.subplots()

        # Plot LD curve
        ax.plot(disp, load, label="$L-\Delta$")

        # Plot stiffness and elastic region
        x = np.linspace((0.0-self.elastic.intercept)/self.elastic.stiffness, (np.max(load)-self.elastic.intercept)/self.elastic.stiffness, 2)
        y = self.elastic.stiffness*x + self.elastic.intercept
        ax.plot(x, y, linestyle="--", color="black", label="Stiffness")
        x = np.linspace((0.0-self.intercept_2)/self.elastic.stiffness, (np.max(load)-self.intercept_2)/self.elastic.stiffness, 2)
        y = self.elastic.stiffness*x + self.intercept_2
        ax.plot(x, y, linestyle="--", color="black")
        ax.axvline(self.ld.disp[self.elastic.id_end], linestyle="-", color="black", label="Elastic region")

        # Add special points
        x0 = -self.intercept_2/self.elastic.stiffness
        x1 = disp[self.id_computation]
        y1 = load[self.id_computation]
        ax.plot(x0, 0, color="red", marker="x")
        ax.plot(x1, 0, color="red", marker="x")
        ax.plot(x1, y1, color="red", marker="x")

        # Add area
        y_bottom = np.maximum(np.zeros_like(load), disp*self.elastic.stiffness+self.intercept_2)
        ax.fill_between(disp, load, y_bottom, color="blue", alpha=0.2, label="$A_{pl1}$")
        if self.ld.load[0] >= 1e-6:
            x0 = -self.elastic.intercept/self.elastic.stiffness
            x1 = np.min(disp)
            y1 = self.elastic.stiffness*x1 + self.elastic.intercept
            ax.plot(x0, 0, color="green", marker="x")
            ax.plot(x1, 0, color="green", marker="x")
            ax.plot(x1, y1, color="green", marker="x")
            ax.fill_between(np.array([x0, x1]), np.array([0.0, 0.0]), np.array([0.0, y1]), color="green", alpha=0.2, label="$A_{pl2}$")

        ax.set_xlabel("$\Delta$ [mm]")
        ax.set_ylabel("$L$ kN")
        ax.legend()

        if save_fig:
            if fig_name == None:
                print("ERROR: Can not create the figure. The figure name is None")
                raise ValueError("Can not create the figure. The figure name is None")
            fig.savefig(fig_name)

        return fig, ax
    
    """
    Print all the data related to the fracture
    """
    def print_all(self):
        print("="*60)
        print("Results for fracture: ")
        print("="*60)

        # -------------------------
        # Specimen geometry
        # -------------------------
        print("\n--- Specimen geometry ---")
        print(f" W   = {self.specimen.W:.3e} mm (specimen width)")
        print(f" S   = {self.specimen.S:.3e} mm (span)")
        print(f" B   = {self.specimen.B:.3e} mm (thickness)")
        print(f" B_N = {self.specimen.B_N:.3e} mm (net thickness)")
        print(f" a0  = {self.specimen.a0:.3e} mm (initial crack length)")
        print(f" b0  = {self.specimen.b0:.3e} mm (remaining ligament)")

        # -------------------------
        # Material properties
        # -------------------------
        print("\n--- Material properties ---")
        print(f" E  = {self.specimen.E:.3f} MPa (Young modulus)")
        print(f" nu = {self.specimen.nu:.3f} (-) (Poisson ratio)")
        print(f" eta_pl = {self.specimen.eta_pl:.3f} (-)")

        # -------------------------
        # Elastic region detection
        # -------------------------
        print("\n--- Elastic region detection ---")
        print(f" Yield load         = {self.ld.load[self.elastic.id_end]} N")
        print(f" Yield displacement = {self.ld.disp[self.elastic.id_end]} mm")
        print(f" Elastic end index  = {self.elastic.id_end}")
        print(f" Stiffness (slope)  = {self.elastic.stiffness:.6e} N/mm")
        print(f" Intercept 1        = {self.elastic.intercept:.6e}")
        print(f" Intercept 2        = {self.intercept_2:.6e}")

        # -------------------------
        # Load at computation point
        # -------------------------
        print("\n--- Computation point ---")
        print(f" Index used        = {self.id_computation}")
        print(f" Load P            = {self.ld.load[self.id_computation]:.6e} N")

        # -------------------------
        # Fracture parameters
        # -------------------------
        print("\n--- Fracture parameters ---")
        print(f" K      = {self.K:.6e} MPa·√mm, {self.K*np.sqrt(1e-3):.6e} MPa·√m")
        print(f" J_el   = {self.J_el:.6e} MPa·mm")
        print(f" J_pl   = {self.J_pl:.6e} MPa·mm")
        print(f" J_tot  = {self.J:.6e} MPa·mm")

        print("="*60)

    """
    Create a report (text file) containing the data of the fracture
    Input:
     - report_name (str): Path and name of the fiel of the report
    """
    def report(self, report_name : str):
        try:
            with open(report_name, "w") as f:

                f.write("="*60 + "\n")
                f.write("Fracture report\n")
                f.write("="*60 + "\n")

                # -------------------------
                # Specimen geometry
                # -------------------------
                f.write("\n--- Specimen geometry ---\n")
                f.write(f" W   = {self.specimen.W:.3e} mm (specimen width)\n")
                f.write(f" S   = {self.specimen.S:.3e} mm (span)\n")
                f.write(f" B   = {self.specimen.B:.3e} mm (thickness)\n")
                f.write(f" B_N = {self.specimen.B_N:.3e} mm (net thickness)\n")
                f.write(f" a0  = {self.specimen.a0:.3e} mm (initial crack length)\n")
                f.write(f" b0  = {self.specimen.b0:.3e} mm (remaining ligament)\n")

                # -------------------------
                # Material properties
                # -------------------------
                f.write("\n--- Material properties ---\n")
                f.write(f" E  = {self.specimen.E:.3f} MPa (Young modulus)\n")
                f.write(f" nu = {self.specimen.nu:.3f} (-) (Poisson ratio)\n")
                f.write(f" eta_pl = {self.specimen.eta_pl:.3f} (-)\n")

                # -------------------------
                # Elastic region detection
                # -------------------------
                f.write("\n--- Elastic region detection ---\n")
                f.write(f" Yield load         = {self.ld.load[self.elastic.id_end]} N\n")
                f.write(f" Yield displacement = {self.ld.disp[self.elastic.id_end]} mm\n")
                f.write(f" Elastic end index  = {self.elastic.id_end}\n")
                f.write(f" Stiffness (slope)  = {self.elastic.stiffness:.6e} N/mm\n")
                f.write(f" Intercept 1        = {self.elastic.intercept:.6e}\n")
                f.write(f" Intercept 2        = {self.intercept_2:.6e}\n")

                # -------------------------
                # Load at computation point
                # -------------------------
                f.write("\n--- Computation point ---\n")
                f.write(f" Index used        = {self.id_computation}\n")
                f.write(f" Load P            = {self.ld.load[self.id_computation]:.6e} N\n")

                # -------------------------
                # Fracture parameters
                # -------------------------
                f.write("\n--- Fracture parameters ---\n")
                f.write(f" K      = {self.K:.6e} MPa mm^0.5, {self.K*np.sqrt(1e-3):.6e} MPa m^0.5\n")
                f.write(f" J_el   = {self.J_el:.6e} MPa mm^0.5\n")
                f.write(f" J_pl   = {self.J_pl:.6e} MPa mm^0.5\n")
                f.write(f" J_tot  = {self.J:.6e} MPa mm^0.5\n")

                f.write("="*60 + "\n")

            print(f"Report successfully written to {report_name}")

        except Exception as e:
            print(f"ERROR: Cannot create report file {report_name}")
            print(f"Reason: {e}")