"""
Compute fracture related values

This module contains one class
- Fracture
    Compute and store the data related to the fracture
This module contains five functions
- geometric_fnc_K
    Compute the geometric function of the stress intensity factor
- stress_intensity_factor
    Compute the elastic stress intensity factor
- J_integral_el
    Compute the elastic J-integral
- J_integral_pl
    Compute the plastic J-integral
- A_plastic
    Plastic area computation
    
Author
------
ROTUNNO Noah

Date
----
2026
"""

from tools.LoadDisplacement import *
from tools.Specimen import *
from tools.ElasticRegion import *
from tools.Logger import Logger
from tools.GeometricFunction import geometric_fnc_K

def stress_intensity_factor_elastic(specimen : Specimen, P : Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the elastic stress intensity factor

    Parameters
    ----------
    specimen : Specimen
        Specimen
    P : float or ndarray
        Load at computation point [N]

    Returns
    -------
    float or ndarray
        Elastic stress intensity factor K_el [MPa mm^0.5]
    """
    f_geom = geometric_fnc_K(specimen.a0, specimen.W)
    K = P*specimen.S/(np.sqrt(specimen.B*specimen.B_N) * specimen.W**1.5) * f_geom
    return K

def stress_intensity_factor_Jc(J_c : Union[float, np.ndarray], specimen : Specimen) -> Union[float, np.ndarray]:
    """
    Compute the total stress intensity factor

    Parameters
    ----------
    J_c : float or ndarray
        J-integral [MPa mm]
    specimen : Specimen
        Specimen

    Returns
    -------
    float or ndarray
        Stress intensity factor K_el [MPa mm^0.5]
    """
    return np.sqrt(J_c*specimen.E_plain_strain)

def J_integral_el(specimen : Specimen, K : Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the elastic J-integral

    Parameters
    ----------
    specimen : Specimen
        Specimen
    K : float or ndarray
        Stress intensity factor [MPa mm^0.5]

    Returns
    -------
    float or ndarray
        Elastic J-integral [MPa mm]
    """
    J_el = K**2 * (1.0 - specimen.nu**2) / specimen.E
    return J_el

def J_integral_pl(specimen : Specimen, A_pl : Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the plastic J-integral

    Parameters
    ----------
    specimen : Specimen
        Specimen
    A_pl : float or ndarray
        Plastic area [N mm]

    Returns
    -------
    float or ndarray
        Plastic J-integral [MPa mm]
    """
    J_pl = specimen.eta_pl*A_pl/(specimen.B_N*specimen.b0)
    return J_pl

def A_plastic(ld : LoadDisplacement, stiffness : Union[float, np.ndarray], id_computation : int, conditional_area : bool) -> Union[float, np.ndarray]:
    """
    Compute plastic area

    Parameters
    ----------
    ld : LoadDisplacement
        Load-displacement data
    stiffness : float or ndarray
        Stiffness [N/mm]
    id_computation : int
        Index of the computation point on the LD curve
    conditional_area : bool
        Should the area before the beginning of the test be computed

    Returns
    -------
    float or ndarray
        The plastic area [N mm]
    """
    # Compute total area under curve
    A_pl1 = np.trapezoid(ld.load[:id_computation], ld.disp[:id_computation])

    # Add the part before the beginning of LD curve if requiered
    if conditional_area:
        A_pl2 = 0.25*((stiffness**2)*(np.min(ld.disp)**2) + np.min(ld.load)**2)/stiffness
    else:
        A_pl2 = 0

    # Remove part under the stiffness at fracture point
    A_rm = 0.5/stiffness*ld.load[id_computation]**2

    return A_pl1 + A_pl2 - A_rm

class Fracture(object):
    """
    Compute the fracture parameters from a test or a simulation

    Parameters
    ----------
    specimen : Specimen
        Specimen
    elastic : ElasticRegion
        Elastic region
    ld : LoadDisplacement
        Load-displacement data
    id_computation : int
        Index of the computation point on the LD curve
    test_nbr : int or None, default None
        Test identification number

    Attributes
    ----------
    is_sample : bool
        Is this fracture instance a sample or not
    conditional_area : bool
        Should the area before the beginning of the LD curve be computed in plastic area
    A_pl : float
        Plastic area [N mm]
    intercept_2 : float
        Interception with y-axis for the stiffness passing by the computation point
    K_el : float
        Elastic stress intensity factor [MPa mm^0.5]
    J_el : float
        Elastic J-integral
    J_pl : float
        Plastic J-integral
    J_c : float
        J-integral
    K_Jc : float 
        Stress intensity factor

    Warnings
    --------
    You can create a sampled fracture using sampled specimen and elastic region. However, you can not create
    a fracture with a sampled specimen and a non sampled elastic region (or vice-versa).
    """
    
    def __init__(self, specimen : Specimen, elastic : ElasticRegion, ld : LoadDisplacement, id_computation : int, test_nbr : Union[int, None] = None):
        # Check if it is a sample (for Monte-Carlo) or single value
        if specimen.is_sample() and elastic.is_sample():
            self.is_sample = True
        elif not specimen.is_sample() and not elastic.is_sample():
            self.is_sample = False
        else:
            print(f"ERROR: Fracture must be or a sample or not. Can not pass a sample specimen and a non-sample elastic region (or vice-versa)")
            raise ValueError("ERROR: Fracture must be or a sample or not. Can not pass a sample specimen and a non-sample elastic region (or vice-versa)")

        self.specimen = specimen
        self.elastic = elastic
        self.ld = ld
        self.id_computation = id_computation
        self.test_nbr = test_nbr

        self.conditional_area = self.ld.load[0] >= 1e-6
        self.A_pl = A_plastic(self.ld, self.elastic.stiffness, self.id_computation, self.conditional_area)
        self.intercept_2 = -self.elastic.stiffness*self.ld.disp[self.id_computation] + self.ld.load[self.id_computation] # Get the intercept point for the second stiffness curve

        self.K_el = stress_intensity_factor_elastic(specimen, self.ld.load[id_computation])
        self.J_el = J_integral_el(specimen, self.K_el)
        self.J_pl = J_integral_pl(specimen, self.A_pl)
        self.J_c = self.J_el + self.J_pl
        self.K_Jc = stress_intensity_factor_Jc(self.J_c, specimen)

    def plot_details(self, save_fig : bool = False, fig_name : Union[str,None] = None) -> Union[plt.Figure, plt.Axes]:
        """
        Create a detailed plot of the load-displacement curve with all the relevant point curve and information

        Parameters
        ----------
        save_fig : bool, default=False
            Save the figure
        fig_name : str or None, default=None
            Name and path of the figure to save

        Returns
        -------
        plt.Figure
            Figure created
        plt.Axes
            Axe created
        """

        fig = plt.figure()
        ax = fig.subplots()

        # Plot LD curve
        ax.plot(self.ld.disp, self.ld.load, label="$L-\Delta$")

        # Plot stiffness and elastic region
        x = np.linspace((0.0-self.elastic.intercept)/self.elastic.stiffness, (np.max(self.ld.load)-self.elastic.intercept)/self.elastic.stiffness, 2)
        y = self.elastic.stiffness*x + self.elastic.intercept
        ax.plot(x, y, linestyle="--", color="black", label="Stiffness")
        x = np.linspace((0.0-self.intercept_2)/self.elastic.stiffness, (np.max(self.ld.load)-self.intercept_2)/self.elastic.stiffness, 2)
        y = self.elastic.stiffness*x + self.intercept_2
        ax.plot(x, y, linestyle="--", color="black")
        ax.axvline(self.ld.disp[self.elastic.id_end], linestyle="-", color="black", label="Elastic region")

        # Add special points
        x0 = -self.intercept_2/self.elastic.stiffness
        x1 = self.ld.disp[self.id_computation]
        y1 = self.ld.load[self.id_computation]
        ax.plot(x0, 0, color="red", marker="x")
        ax.plot(x1, 0, color="red", marker="x")
        ax.plot(x1, y1, color="red", marker="x")

        # Add area
        y_bottom = np.maximum(np.zeros_like(self.ld.load), self.ld.disp*self.elastic.stiffness+self.intercept_2)
        ax.fill_between(self.ld.disp, self.ld.load, y_bottom, color="blue", alpha=0.2, label="$A_{pl1}$")
        if self.ld.load[0] >= 1e-6:
            x0 = -self.elastic.intercept/self.elastic.stiffness
            x1 = np.min(self.ld.disp)
            y1 = self.elastic.stiffness*x1 + self.elastic.intercept
            ax.plot(x0, 0, color="green", marker="x")
            ax.plot(x1, 0, color="green", marker="x")
            ax.plot(x1, y1, color="green", marker="x")
            ax.fill_between(np.array([x0, x1]), np.array([0.0, 0.0]), np.array([0.0, y1]), color="green", alpha=0.2, label="$A_{pl2}$")

        ax.set_xlabel("$\Delta$ [mm]")
        ax.set_ylabel("$L$ [N]")
        ax.legend()

        if save_fig:
            if fig_name == None:
                print("ERROR: Can not create the figure. The figure name is None")
                raise ValueError("Can not create the figure. The figure name is None")
            fig.savefig(fig_name)

        return fig, ax

def log_fracture_data(fracture : Fracture, logger : Logger):
    """
    Log all the data related to the fracture

    Parameters
    ----------
    logger : Logger
        Logger to log the data
    """

    # Check if the fracture is no sampled
    if fracture.is_sample:
        print("ERROR: Can not log fracture data from a sampled fracture.")
        raise ValueError("ERROR: Can not log fracture data from a sampled fracture.")

    logger.log("="*60)
    if fracture.test_nbr is None:
        logger.log("Results for fracture")
    else:
        logger.log("Results for fracture. Test " + str(fracture.test_nbr))
    logger.log("="*60)

    # -------------------------
    # Specimen geometry
    # -------------------------
    logger.log("\n--- Specimen geometry ---")
    logger.log(f" W       = {fracture.specimen.W:.3e} mm (specimen width)")
    logger.log(f" S       = {fracture.specimen.S:.3e} mm (span)")
    logger.log(f" B       = {fracture.specimen.B:.3e} mm (thickness)")
    logger.log(f" B_N     = {fracture.specimen.B_N:.3e} mm (net thickness)")
    logger.log(f" a0      = {fracture.specimen.a0:.3e} mm (initial crack length)")
    logger.log(f" b0      = {fracture.specimen.b0:.3e} mm (remaining ligament)")
    logger.log(f" f(a0/W) = {geometric_fnc_K(fracture.specimen.a0, fracture.specimen.W):.3e} (-) (geometric function)")
    logger.log(f" eta_pl  = {fracture.specimen.eta_pl:.3f} (-)")

    # -------------------------
    # Material properties
    # -------------------------
    logger.log("\n--- Material properties ---")
    logger.log(f" E      = {fracture.specimen.E:.3f} MPa (Young modulus)")
    logger.log(f" E'     = {fracture.specimen.E_plain_strain:.3f} MPa (Effective modulus in plain strain)")
    logger.log(f" nu     = {fracture.specimen.nu:.3f} (-) (Poisson ratio)")
    logger.log(f" K_Jc lim = {fracture.specimen.K_Jc_lim:.3f} MPa mm^0.5, {fracture.specimen.K_Jc_lim*np.sqrt(1e-3):.3f} MPa m^0.5 (Maximum K_Jc)")

    # -------------------------
    # Elastic region detection
    # -------------------------
    logger.log("\n--- Elastic region detection ---")
    logger.log(f" Yield load         = {fracture.ld.load[fracture.elastic.id_end]} N")
    logger.log(f" Yield displacement = {fracture.ld.disp[fracture.elastic.id_end]} mm")
    logger.log(f" Elastic end index  = {fracture.elastic.id_end}")
    logger.log(f" Stiffness (slope)  = {fracture.elastic.stiffness:.6e} N/mm")
    logger.log(f" Intercept 1        = {fracture.elastic.intercept:.6e} (Interception of y-axis for elastic region)")
    logger.log(f" Intercept 2        = {fracture.intercept_2:.6e} (Interception of y-axis for computation point)")
    logger.log(f" A plastic          = {fracture.A_pl:.6e} (plastic area)")

    # -------------------------
    # Load at computation point
    # -------------------------
    logger.log("\n--- Computation point ---")
    logger.log(f" Index used        = {fracture.id_computation}")
    logger.log(f" Load P            = {fracture.ld.load[fracture.id_computation]:.6e} N")

    # -------------------------
    # Fracture parameters
    # -------------------------
    logger.log("\n--- Fracture parameters ---")
    logger.log(f" J_el   = {fracture.J_el:.6e} MPa mm, {fracture.J_el*1e-3:.6e} MPa m")
    logger.log(f" J_pl   = {fracture.J_pl:.6e} MPa mm, {fracture.J_pl*1e-3:.6e} MPa m")
    logger.log(f" J_c    = {fracture.J_c:.6e} MPa mm, {fracture.J_c*1e-3:.6e}  MPa m")
    logger.log(f" K_el   = {fracture.K_el:.6e} MPa mm^0.5, {fracture.K_el*np.sqrt(1e-3):.6e} MPa m^0.5")
    logger.log(f" K_Jc   = {fracture.K_Jc:.6e} MPa mm^0.5, {fracture.K_Jc*np.sqrt(1e-3):.6e} MPa m^0.5")

    logger.log("="*60)