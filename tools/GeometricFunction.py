import numpy as np
from typing import Union 

def geometric_fnc_K(a0 : Union[float, np.ndarray], W : Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the geometric function for the stress intensity factor computation

    Parameters
    ----------
    a0 : float or ndarray
        Initial crack length [mm]
    W : float or ndarray
        Width of the specimen [mm]

    Returns
    -------
    float or ndarray
        Geometric value [-]
    """
    r = a0/W
    return 3.0*np.sqrt(r)*(1.99 - r*(1.0-r)*(2.15 - 3.93*r + 2.7*r**2))/(2.0*(1.0 + 2.0*r)*(1.0-r)**1.5)