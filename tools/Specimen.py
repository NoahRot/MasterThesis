
"""
Specimen class.
Input-Parameters:
 - W (float) : width of the specimen [mm]
 - S (float) : span [mm]
 - B (float) : thickness [mm]
 - B_N (float) : net thickness [mm]
 - a0 (float) : initial crack length [mm]
 - nu (float): Poisson ratio [-]
 - E (float): Young modulus [MPa]
 - eta_pl (float): ??? parameters to compute the J-elastic
Parameters:
 - b0 (float): remaining ligament (W - a0)
"""
class Specimen(object):
    def __init__(self, W, S, B, B_N, a0, nu, E, eta_pl):
        # Physical parameters
        self.W = W
        self.S = S
        self.B = B
        self.B_N = B_N
        self.a0 = a0
        self.nu = nu
        self.E = E
        self.eta_pl = eta_pl
        self.b0 = W - a0