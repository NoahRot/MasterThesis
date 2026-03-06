from tools.LoadDisplacement import *
import scipy as sci

class ElasticRegion(object):
    def __init__(self, id_end, stiffness, intercept):
        self.id_end = id_end
        self.stiffness = stiffness
        self.intercept = intercept

def elastic_region_determination_r2_max(ld : LoadDispalcement, min_points : int = 10, debug_plot = False):
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
        ax2.set_title("$R^2$ value evaluation")

    return ElasticRegion(elastic_end, stiffness, intercept)

"""
Determine the elastic region using the R^2 method.
Input:
 - load : (np.array[float]) array containing the load data
 - disp : (np.array[float]) array containing the displacement data
 - min_points : (int) minimum number of points used to compute the first linear regression. Must be >= 2
 -  r2_threshold : (float) threshold for R^2. If R^2 > threshold, then consider that it goes from elastic to plastic.
Output:
 - (int) index at which the elastic region stop 
"""
def elastic_region_determination_r2_method(ld : LoadDispalcement, min_points : int = 3, r2_threshold : float = 0.995, debug_plot = False) -> list[int, float, float]:
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
        ax.plot(r2_list, "b+")
        ax.set_ylabel("$R^2$")

    return ElasticRegion(elastic_end, stiffness, intercept)