import numpy as np

from tools.Specimen import *
from tools.ElasticRegion import *
from tools.LoadDisplacement import *
from tools.Fracture import *

class FractureMC(object):
    def __init__(self, specimen : SpecimenUncertainties, elastic : ElasticRegionUncertainties, ld : LoadDisplacement, id_computation : int, nbr_mc : int):
        self.P = ld.load[self.id_computation]

        self.a0 = np.random.normal(specimen.a0, specimen.a0_u, nbr_mc)
        self.W = np.random.normal(specimen.W, specimen.W_u, nbr_mc)
        self.S = np.random.normal(specimen.S, specimen.S_u, nbr_mc)
        self.B = np.random.normal(specimen.B, specimen.B_u, nbr_mc)
        self.B_N = np.random.normal(specimen.B_N, specimen.B_N_u, nbr_mc)
        self.nu = np.random.normal(specimen.nu, specimen.nu_u, nbr_mc)
        self.E = np.random.normal(specimen.E, specimen.E_u, nbr_mc)

    def stress_intensity_factor(self) -> float:
        f_geom = geometric_fnc_K(self.a0, self.W)
        self.K = self.P*self.S/(np.sqrt(self.B*self.B_N) * self.W**1.5) * f_geom

    def J_integral_el(self) -> float:
        return self.K**2 * (1.0 - self.nu**2) / self.E

    def J_integral_pl(self) -> float:
        load, disp = self.ld.get_LD_sorted()

        # Linear regression in the elastic region
        intercept_2 = -self.elastic.stiffness*disp[self.id_computation] + load[self.id_computation]

        # Compute area under load-disp curve
        A_pl = np.trapz(load[:self.id_computation], disp[:self.id_computation])

        # Add the rest of the area using stiffness if needed
        if self.ld.load[0] >= 1e-6:
            x0 = -self.elastic.intercept/self.elastic.stiffness
            x1 = np.min(disp)
            y1 = self.elastic.stiffness*x1 + self.elastic.intercept
            A_pl += 0.5*(x1 - x0)*y1

        # Remove the triangle area below the index_computation
        x0 = -intercept_2/self.elastic.stiffness
        x1 = disp[self.id_computation]
        y1 = load[self.id_computation]
        A_pl -= 0.5*(x1 - x0)*y1

        return self.specimen.eta_pl*A_pl/(self.specimen.B_N*self.specimen.b0)