
"""
Specimen class. Store parameters of the specimen geometry and material.
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

class SpecimenUncertainties(Specimen):

    def __init__(self,
                 W, W_u,
                 S, S_u,
                 B, B_u,
                 B_N, B_N_u,
                 a0, a0_u,
                 nu,
                 E, E_u,
                 eta_pl):

        super().__init__(W, S, B, B_N, a0, nu, E, eta_pl)

        self.W_u = W_u
        self.S_u = S_u
        self.B_u = B_u
        self.B_N_u = B_N_u
        self.a0_u = a0_u
        self.E_u = E_u