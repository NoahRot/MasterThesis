import numpy as np

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

class SpecimenDistribution(object):
    def __init__(self, W, W_u, S, S_u, B, B_u, B_N, B_N_u, a0, a0_u, nu, E, E_u, eta_pl):
        self.W = W
        self.W_u = W_u
        self.S = S
        self.S_u = S_u
        self.B = B
        self.B_u = B_u
        self.B_N = B_N
        self.B_N_u = B_N_u
        self.a0 = a0
        self.a0_u = a0_u
        self.nu = nu
        self.E = E
        self.E_u = E_u
        self.eta_pl = eta_pl

    def simple(self) -> Specimen:
        return Specimen(self.W, self.S, self.B, self.B_N, self.a0, self.nu, self.E, self.eta_pl)

    def sample(self, nbr_samples : int, rng : np.random.Generator = None) -> Specimen:
        if rng is None:
            rng = np.random.default_rng()

        W_sampled = rng.uniform(self.W - self.W_u, self.W + self.W_u, nbr_samples)
        S_sampled = rng.uniform(self.S - self.S_u, self.S + self.S_u, nbr_samples)
        B_sampled = rng.uniform(self.B - self.B_u, self.B + self.B_u, nbr_samples)
        B_N_sampled = rng.uniform(self.B_N - self.B_N_u, self.B_N + self.B_N_u, nbr_samples)
        a0_sampled = rng.normal(self.a0, self.a0_u, nbr_samples)
        E_sampled = rng.normal(self.E, self.E_u, nbr_samples)

        return Specimen(W_sampled, S_sampled, B_sampled, B_N_sampled, a0_sampled, self.nu, E_sampled, self.eta_pl)