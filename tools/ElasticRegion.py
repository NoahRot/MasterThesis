"""
Elastic region definition and elastic region uncertainties distribution for Monte-Carlo analysis

This module contains two classes:
- ElasticRegion
    Represent the elatic region properties
- ElasticRegionDistribution
    Describe statistical distributions of the parameters of an elastic region distribution. It can generate
    deterministic elatic region or sample values for elatic region parameters.
This module contains four functions
- elastic_region_distribution
    Create an elastic region distribution from an already determind elastic distribution and a LD curve
- offset_LD_according_to_stiffness
    Offset the LD curve such that the stiffness pass ba the origin. Correct also the elastic region according to this offset
- elastic_region_determination_r2_max
    Determine the elastic region using maximum R^2 value (good for experimental data)
- elastic_region_determination_r2_method
    Determine the elastic region using a R^2 threshold (good for simulation data)
    
Author
------
ROTUNNO Noah

Date
----
2026
"""

from tools.LoadDisplacement import *
import scipy as sci

class ElasticRegion(object):
    """
    Representation of the elastic region of a test sample

    Parameters
    ----------
    id_end : int
        Index of the end of the elastic region (index valid on the LD curve)
    stiffness : float
        Stiffness [N/mm]
    intercept : float
        Interception with the y-axis when stiffness is evaluated [N]
    """
    
    def __init__(self, id_end : int, stiffness : float, intercept : float):
        self.id_end = id_end
        self.stiffness = stiffness
        self.intercept = intercept

class ElasticRegionDistribution(object):
    """
    Statistical distribution of the specimen parameters.

    Parameters
    ----------
    id_end : int
        Index of the end of the elastic region (index valid on the LD curve)
    stiffness : float
        Stiffness [N/mm]
    stiffness_u : float
        Uncertainty on stiffness [N/mm]
    intercept : float
        Interception with the y-axis when stiffness is evaluated [N]
    intercept_u : float
        Uncertainty on interception with the y-axis [N]
    """

    def __init__(self, id_end : int, stiffness : float, stiffness_u : float, intercept : float, intercept_u : float):
        self.id_end = id_end 
        self.stiffness = stiffness 
        self.stiffness_u = stiffness_u
        self.intercept = intercept
        self.intercept_u = intercept_u

    def simple(self) -> ElasticRegion:
        """
        Create a deterministic elastic region instance using the values without uncertainties

        Returns
        -------
        ElasticRegion
            ElasticRegion object built using nominal parameters
        """
        return ElasticRegion(self.id_end, self.stiffness, self.intercept)

    def sample(self, nbr_samples : int, rng : np.random.Generator = None) -> ElasticRegion:
        """
        Sample values to generate random ElasticRegion for Monte-Carlo method

        Parameters
        ----------
        nbr_samples : int
            Number of sampled values
        rng : numpy.random.Generator or None, default=None
            Random number generator. If None, a default generator is created.

        Returns
        -------
        ElasticRegion
            ElasticRegion object containing sampled parameter arrays
        """
        if rng is None:
            rng = np.random.default_rng()

        stiffness = rng.normal(self.stiffness, self.stiffness_u, nbr_samples)
        intercept = rng.normal(self.intercept, self.intercept_u, nbr_samples)
        return ElasticRegion(self.id_end, stiffness, intercept)

def elastic_region_distribution(ld : LoadDisplacement, elastic : ElasticRegion) -> ElasticRegionDistribution:
    """
    Compute the elastic region distribution from a load-displacment curve and an elastic region

    Parameters
    ----------
    ld : LoadDisplacement
        Load-displacement data
    elastic : ElasticRegion
        Elastic region data

    Returns
    -------
    ElasticRegionDistribution:
        Distribution of the elastic region data
    """

    # Compute the correct values for stiffness and intercept
    elastic_end = elastic.id_end
    result = sci.stats.linregress(ld.disp[:elastic_end], ld.load[:elastic_end])

    stiffness = result.slope
    intercept = result.intercept

    stiffness_u = result.stderr
    intercept_u = result.intercept_stderr

    return ElasticRegionDistribution(elastic_end, stiffness, stiffness_u, intercept, intercept_u)

def offset_LD_according_to_stiffness(ld : LoadDisplacement, elastic : ElasticRegion) -> Union[LoadDisplacement, ElasticRegion]:
    """
    Offset the load-displacement data such that, according to elastic region, the elastic curve pass by the origin. This also change the 
    elastic region interception parameters.

    Parameters
    ----------
    ld : LoadDisplacement
        Load-displacement data
    elastic : ElasticRegion
        Elastic region data

    Returns
    -------
    LoadDisplacement
        Load-displacement data with offset on displacement
    ElasticRegion
        Elastic region corected according to the offset

    Warnings
    --------
    Both load-dispalcement and elastic region are modified. Using new load-displacement with old elastic region
    would leads to errors in future the computations
    """
    offset = elastic.intercept/elastic.stiffness
    ld.disp += offset
    elastic.intercept = 0
    return ld, elastic

def elastic_region_determination_r2_max(ld : LoadDisplacement, min_points : int = 10, debug_plot : bool = False) -> ElasticRegion:
    """
    Determine the elastic region using the maximum value of R^2

    Parameters
    ----------
    ld : LoadDisplacement
        Load-displacement data
    min_points : int, default=10
        Minimum number of points used for the linear regression
    debug_plot : bool, default=False
        If true, plot a figures containing graphs that show the R^2 evaluation for different number of indices

    Returns
    -------
    ElasticRegion
        The determined elastic region

    Note
    ----
    This method is good to determine the elastic region from experimental data
    """
    # Use R^2 to find the stiffness
    r2 = 0.0
    elastic_end = 0
    r2_list = []

    # Make the list of the R^2 values
    for i in range(min_points, len(ld.disp)):
        # Evaluate slope and compute R^2
        slope, intercept, r_value, p_value, std_err = sci.stats.linregress(ld.disp[:i], ld.load[:i])
        r2 = r_value**2
        r2_list.append(r2)

    # Get the maximum of R^2
    r2_list = np.array(r2_list)
    elastic_end = np.argmax(r2_list)

    # Compute the correct values for stiffness and intercept
    stiffness, intercept, r_value, p_value, std_err = sci.stats.linregress(ld.disp[:elastic_end], ld.load[:elastic_end])

    # Debug figure
    if debug_plot:
        fig2 = plt.figure()
        ax2 = fig2.subplots()
        ax2.plot(r2_list)
        ax2.vlines(elastic_end, np.min(r2_list), np.max(r2_list), linestyles="--", colors="red")
        ax2.set_ylabel("$R^2$")
        ax2.set_title("$R^2$ value evaluation - $(R^2)_{max}$ method")

    return ElasticRegion(elastic_end, stiffness, intercept)

def elastic_region_determination_r2_method(ld : LoadDisplacement, min_points : int = 3, r2_threshold : float = 0.995, debug_plot : bool = False) -> ElasticRegion:
    """
    Determine the elastic region using an R^2 threshold

    Parameters
    ----------
    ld : LoadDisplacement
        Load-displacement data
    min_points : int, default=3
        minimum number of points used for linear regression
    r2_threshold : float, default=0.995
        Threshold of R^2
    debug_plot : bool, default=False
        If true, plot a figures containing graphs that show the R^2 evaluation for different number of indices

    Returns
    -------
    ElasticRegion
        The determined elastic region

    Note
    ----
    This method is good to determine the elastic region from abaqus data
    """
    # Variables
    r2 = 0.0
    elastic_end = 0
    r2_list = []

    # Search where the criteria is satisfied (R^2 < threshold)
    for i in range(min_points, len(ld.disp)):
        # Evaluate slope and compute R^2
        slope, intercept, r_value, p_value, std_err = sci.stats.linregress(ld.disp[:i], ld.load[:i])
        r2 = r_value**2
        r2_list.append(r2)

        # Check criteria
        if r2 < r2_threshold:
            elastic_end = i - 1
            break
    
    # Print error if can not find the elastic region
    if elastic_end == 0:
        print("ERROR: Can not determine the elastic region")
        raise ValueError("Can not determine the elastic region")

    # Compute final stiffnes and intercept
    stiffness, intercept, r_value, p_value, std_err = sci.stats.linregress(ld.disp[:elastic_end], ld.load[:elastic_end])

    # Debug R^2 evolution
    if debug_plot:
        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(r2_list)
        ax.set_ylabel("$R^2$")
        ax.set_title("$R^2$ value evaluation - $(R^2)$ threshold method")

    return ElasticRegion(elastic_end, stiffness, intercept)