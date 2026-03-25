from tools.reader import *
from tools.CrackProfile import *
from tools.LoadDisplacement import *
from tools.plt_spec import *
from tools.Specimen import *
from tools.ElasticRegion import *
from tools.Fracture import *
from tools.MonteCarlo import *
from tools.MasterCurve import *
import os
import sys

init_plt(latex=False)
create_sns_palette()

# -------------------------------
# Input parameters
# -------------------------------
# Speciemen Geometry
W = 4 # specimen width [mm]
S = 4*W # span [mm]
B = 3 # specimen thickness [mm]
B_N = B # specimen net thickness [mm]
eta_pl = 1.9 # parameters to compute the J-elastic.
sigma_YS = 700

# Speciemen Material data
nu = 0.3 # Poisson ratio
E = 210250 # Young modulus [MPa]

id_computation = -1
nbr_sample = 100000

T = -120 # Temperature of the tests [°C]
test_number = 4
begin = -1
end = -1

if len(sys.argv) > 1:
    try:
        test_number =  int(sys.argv[1])
    except:
        print("WARNING: Can not get the index of the test as an argument. It can not be converted into 'int'")

if len(sys.argv) > 2:
    try:
        begin =  int(sys.argv[2])
    except:
        print("WARNING: Can not get the argument for the beginning. It can not be converted into 'int'")

if len(sys.argv) > 3:
    try:
        end =  int(sys.argv[3])
    except:
        print("WARNING: Can not get the argument for the beginning. It can not be converted into 'int'")

# -------------------------------
# Path to files
# -------------------------------
path = "C:\\Users\\rotunn_n\\Documents\\PDM\\data\\3_points_bending"
#path = "data_test"
test_name = ["sample", "_m120C.csv"]
crack_name = ["EU97C", "_crack_length.xlsx"]
report_name = ["report/test", ".txt"]
figure_name = ["fig/test", ".svg"]

ld_path     = os.path.join(path, test_name[0]) + str(test_number) + test_name[1]       # load-disp data
crack_path  = os.path.join(path, crack_name[0]) + str(test_number) + crack_name[1]     # crack data
report_path = report_name[0] + str(test_number) + report_name[1]                       # report path and name
figure_path = figure_name[0] + str(test_number) + figure_name[1]                       # figure path and name

# -------------------------------
# Logger
# -------------------------------

logger = Logger("txt", report_path)

# -------------------------------
# Specimen
# -------------------------------
specimen_u = SpecimenDistribution(
    W = W,       # width [mm]
    W_u = 0.001,    # uncertainty in width [mm]
    S = S,      # span [mm] (4*W)
    S_u = 0.001,    # uncertainty in span [mm]
    B = B,       # thickness [mm]
    B_u = 0.001,    # uncertainty in thickness [mm]
    B_N = B_N,     # net thickness [mm]
    B_N_u = 0.001,  # uncertainty in net thickness [mm]
    nu = nu,      # Poisson ratio [-]
    E = E,    # Young modulus [MPa]
    E_u = 10,    # uncertainty in Young modulus [MPa]
    eta_pl = eta_pl,  # eta_pl [-],
    sigma_YS = sigma_YS, # Yield strength [MPa]
    sigma_YS_u = 1, # uncertainty in Yield strength [MPa]
    crack_profile_dist = None # Initial Crack profile 
)

# -------------------------------
# Experimental
# -------------------------------
print("="*60)
print(" Test " + str(test_number) + " analysis ")
print("="*60)

# -------------------------------
# Load data, treat and compute SIF and J-integral
# -------------------------------
print("Analysis of fracture")
ld = experiment_LD_reader(ld_path)
crack_profile = crack_profile_reader(crack_path)
specimen = Specimen(W, S, B, B_N, crack_profile.initial_crack_length(), nu, E, eta_pl, sigma_YS)

fig = plt.figure()
ax = fig.subplots()
ax.plot(ld.load)
ld = experimental_LD_treatment(ld, 5, begin, end, True)
elastic_region = elastic_region_determination_r2_max(ld, 10, True)
ld, elastic_region = offset_LD_according_to_stiffness(ld, elastic_region)

fracture = Fracture(specimen, elastic_region, ld, id_computation, test_number)

# -------------------------------
# Compute uncertainties
# -------------------------------
rng = np.random.default_rng()
specimen_u.crack_profile_dist = crack_profile_distribution(crack_profile, 0.001, 0.001)
specimen_mc = specimen_u.sample(nbr_sample, rng)
elastic_mc = elastic_region_distribution(ld, elastic_region).sample(nbr_sample, rng)
fracture_mc = Fracture(specimen_mc, elastic_mc, ld, id_computation, test_number)

fracture.plot_details(True, figure_path)

log_fracture_uncertainties(fracture, fracture_mc, logger)

plt.show()