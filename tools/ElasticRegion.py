from tools.LoadDisplacement import *
import scipy as sci

"""
Class representing the elastic region
Input-Parameters:
 - id_end (int): index of the end of the plastic region
 - stiffness (float): stiffness (slope in the elastic region)
 - intercept (float): interception of the stiffness line with the y-axis
"""
class ElasticRegion(object):
    def __init__(self, id_end : int, stiffness : float, intercept : float):
        self.id_end = id_end
        self.stiffness = stiffness
        self.intercept = intercept

class ElasticRegionUncertainties(ElasticRegion):
    def __init__(self,
                 id_end: int,
                 stiffness: float, stiffness_u: float,
                 intercept: float, intercept_u: float):

        super().__init__(id_end, stiffness, intercept)

        # uncertainties (standard deviations)
        self.stiffness_u = stiffness_u
        self.intercept_u = intercept_u

"""
Offset the load-displacement data such that the stiffness curve pass by the origin (0;0)
Input:
 - ld (LoadDisplacement): Load-displacement data
 - elastic (ElasticRegion): Elastic region
Ouput:
 - ld (LoadDisplacement): Load-displacement data with an offset
 - elastic (ElasticRegion): Elastic region modified
"""
def offset_LD_according_to_stiffness(ld : LoadDisplacement, elastic : ElasticRegion):
    offset = elastic.intercept/elastic.stiffness
    ld.disp += offset
    elastic.intercept = 0
    return ld, elastic

"""
Determine the elastic region using the maximum value of R^2
Input:
 - ld (LoadDisplacement): Load-displacement data
 - min_points (int): minimum number of points used initially to fit the curve
 - debug_plot (bool): If true, plot a figures of the progression of R^2
Output:
 - (ElasticRegion): Elastic region
"""
def elastic_region_determination_r2_max(ld : LoadDisplacement, min_points : int = 10, debug_plot : bool = False):
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

"""
Determine the elastic region using an R^2 threshold
Input:
 - ld (LoadDisplacement): Load-displacement data
 - min_points (int): minimum number of points used initially to fit the curve
 - debug_plot (bool): If true, plot a figures of the progression of R^2
Output:
 - (ElasticRegion): Elastic region
"""
def elastic_region_determination_r2_method(ld : LoadDisplacement, min_points : int = 3, r2_threshold : float = 0.995, debug_plot = False) -> list[int, float, float]:
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
        quit()

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