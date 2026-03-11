from tools.reader import *
from tools.CrackProfile import *
from tools.LoadDisplacement import *
from tools.plt_spec import *
from tools.Specimen import *
from tools.ElasticRegion import *
from tools.Fracture import *
from tools.MonteCarlo import *
import os

init_plt(latex=False)
create_sns_palette()

# -------------------------------
# Input parameters
# -------------------------------
# Speciemen parameters
W = 3 # specimen width [mm]
S = 4*W # span [mm]
B = 4 # specimen thickness [mm]
B_N = B # specimen net thickness [mm]
a0 = 1.5 # initial crack length [mm]

# Material data
nu = 0.3 # Poisson ratio
E = 210250 # Young modulus [MPa]

# IDK
eta_pl = 2.0 # ??? parameters to compute the J-elastic. It is 1.9 because we use the load-displacement curve

id_computation = -1
nbr_sample = 100000

# -------------------------------
# Path to files
# -------------------------------
path = "C:\\Users\\rotunn_n\\Documents\\PDM\\data\\3_points_bending"

# vvv Change this vvv 
test_nbr = 6
# ^^^ Change this ^^^

test1 = "sample1_m120C.csv"
test2 = "sample2_m120C.csv"
test3 = "sample3_m120C.csv"
test4 = "sample4_m120C.csv"
test6 = "sample6_m120C.csv"
test7 = "sample7_m120C.csv"

crack1 = "EU97C1_crack_length.xlsx"
crack2 = "EU97C2_crack_length.xlsx"
crack3 = "EU97C3_crack_length.xlsx"
crack4 = "EU97C4_crack_length.xlsx"
crack6 = "EU97C6_crack_length.xlsx"
crack7 = "EU97C7_crack_length.xlsx"

# -------------------------------
# Load data, treat and compute SIF and J-integral
# -------------------------------
full_path = os.path.join(path, "sample" + str(test_nbr) + "_m120C.csv")
crack_path = os.path.join(path, "EU97C" + str(test_nbr) + "_crack_length.xlsx")

ld = experiment_LD_reader(full_path)
crack_profile = crack_profile_reader(crack_path)

specimen = Specimen(W, S, B, B_N, crack_profile.initial_crack_length(), nu, E, eta_pl)

ld = experimental_LD_treatment(ld, 5, True)
elastic_region = elastic_region_determination_r2_max(ld, 10, True)
ld, elastic_region = offset_LD_according_to_stiffness(ld, elastic_region)

fracture = Fracture(specimen, elastic_region, ld, id_computation)
fracture.print_all()
fracture.plot_details(True, "fig/test" + str(test_nbr) + ".svg")
fracture.report("report/test" + str(test_nbr) + ".txt")

# -------------------------------
# Compute uncertainties
# -------------------------------
crack_profile_u = crack_profile_distribution(crack_profile, 0.01, 0.01)

specimen_u = SpecimenDistribution(
    W = W,       # width [mm]
    W_u = 0.01,    # uncertainty in width [mm]
    S = S,      # span [mm] (4*W)
    S_u = 0.05,    # uncertainty in span [mm]
    B = B,       # thickness [mm]
    B_u = 0.01,    # uncertainty in thickness [mm]
    B_N = B_N,     # net thickness [mm]
    B_N_u = 0.01,  # uncertainty in net thickness [mm]
    nu = nu,      # Poisson ratio [-]
    E = E,    # Young modulus [MPa]
    E_u = 1000,    # uncertainty in Young modulus [MPa]
    eta_pl = eta_pl,  # eta_pl [-]
    crack_profile_dist = crack_profile_u
)

elastic_u = elastic_region_distribution(ld, elastic_region)

rng = np.random.default_rng()
mc = FractureMC(specimen_u.sample(nbr_sample, rng), elastic_u.sample(nbr_sample, rng), ld, id_computation)
mc.plot_mc_results(100)
report_with_uncertainties("report/test_u" + str(test_nbr) + ".txt", fracture, mc, specimen_u, elastic_u)

# -------------------------------
# Load and compute Abaqus data 
# -------------------------------
#abaqus_path = "C:\\Users\\rotunn_n\\Documents\\Abaqus\\Charpy Models"
#abaqus_rpt_1 = "charpy_1micron_L=1500micron_R=2mm\\load_disp.rpt"
#abaqus_rpt_2 = "charpy_05micron_L=1500micron_R=2mm\\load_disp.rpt"
#abaqus_rpt_3 = "charpy_01micron_L=1500micron_R=2mm\\load_disp.rpt"
#
#ld2 = abaqus_LD_reader(os.path.join(abaqus_path, abaqus_rpt_3))
#elastic_region = elastic_region_determination_r2_method(ld2, 3, 0.999, False)
#fracture = Fracture(specimen, elastic_region, ld2, id_computation)
#fracture.print_all()
#fracture.plot_details()
#fracture.report("report\\simulation01.txt")

plt.show()