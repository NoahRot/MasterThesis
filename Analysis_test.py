from tools.Analyzer_LD import *
import os

# -------------------------------
# Input parameters
# -------------------------------
# Speciemen parameters
W = 3e3 # specimen width [micron]
S = 8*W # span [micron]
B = 4e3 # specimen thickness [micron]
B_N = B # specimen net thickness [micron]
a0 = 1500 # initial crack length [micron]

# Material data
nu = 0.3 # Poisson ratio
E = 210250 # Young modulus [MPa]

# IDK
eta_pl = 1.9 # ??? parameters to compute the J-elastic. It is 1.9 because we use the load-displacement curve

# Numerial parameters
index_computation = -1 # Index at which the computation will be performed
min_point = 3 # Minimum number of points to compute the elastic region
r2_threshold = 0.998 # R^2 threshold for the computation of the elastic region

# -------------------------------
# Analysis
# -------------------------------

# Paths
"""
path1 = "../../Abaqus/Charpy Models/charpy_05micron_L=1500micron_R=2mm"
file_name1 = "load_disp3.rpt"
full_path1 = os.path.join(path1, file_name1)
analysis1 = Analyzer_LD(full_path1, W, S, B, B_N, a0, nu, E, eta_pl, index_computation, min_point, r2_threshold, "abaqus")

analysis1.print_all()
analysis1.plot_details()
"""

path2 = "../data/3_points_bending"
file_name2 = "sample3_m120C.csv"
full_path2 = os.path.join(path2, file_name2)
analysis2 = Analyzer_LD(full_path2, W, S, B, B_N, a0, nu, E, eta_pl, index_computation, min_point, r2_threshold, "experiment")

analysis2.print_all()
analysis2.plot_time()

plt.show()