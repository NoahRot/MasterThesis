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

init_plt(latex=False, background_fig_color="white", background_axe_color="white")
create_sns_palette("bright", "seaborn", True)

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
percentile = 5

T = -120 # Temperature of the tests [°C]

# -------------------------------
# Path to files
# -------------------------------
path = "C:\\Users\\rotunn_n\\Documents\\PDM\\data\\3_points_bending"
#path = "data_test"

list_test = [1, 3, 4, 6, 7, 8, 9, 10, 11, 12] # Path for the non Hydrogen charged samples
#list_test = [13, 14] # Path for the Hydrogen charged samples
test_name = ["sample", "_m120C.csv"]
crack_name = ["EU97C", "_crack_length.xlsx"]
report_name = ["report/test", ".txt"]
figure_name = ["fig/test", ".svg"]

file_names = [] # LD data, crack data, report
for i in list_test:
    file_names.append([os.path.join(path, test_name[0]) + str(i) + test_name[1],  # load-disp data
                       os.path.join(path, crack_name[0]) + str(i) + crack_name[1],      # crack data
                       report_name[0] + str(i) + report_name[1],                        # report path and name
                       figure_name[0] + str(i) + figure_name[1]                         # figure path and name
    ])

path_abaqus = "C:\\Users\\rotunn_n\\Documents\\Abaqus\\Charpy Models\\"
list_abaqus = ['01']
list_abaqus = []
ld_abaqus = "load_disp.rpt"
abaqus_name = ["charpy_", "micron_L=1500micron_R=2mm"]
abaqus_crack_name = ["EU97C", "_crack_length.xlsx"]
abaqus_report_name = ["report/abaqus", ".txt"]
abaqus_figure_name = ["fig/abaqus", ".svg"]
abaqus_files = []
for i in list_abaqus:
    abaqus_files.append([os.path.join(path_abaqus, abaqus_name[0] + i + abaqus_name[1], ld_abaqus),   # load-disp data
                       1.5,                                                                         # crack data
                       abaqus_report_name[0] + i + abaqus_report_name[1],                           # report path and name
                       abaqus_figure_name[0] + i + abaqus_figure_name[1]                            # figure path and name
    ])

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
# Compute load-displacement
# -------------------------------
fractures = []
fractures_mc = []

# -------------------------------
# Experimental
# -------------------------------
i = 0
for test in file_names:
    print("="*60)
    print(" Test " + str(list_test[i]) + " analysis ")
    print("="*60)

    # -------------------------------
    # Load data, treat and compute SIF and J-integral
    # -------------------------------
    print("Analysis of fracture")
    ld = experiment_LD_reader(test[0])
    crack_profile = crack_profile_reader(test[1])
    specimen = Specimen(W, S, B, B_N, crack_profile.initial_crack_length(), nu, E, eta_pl, sigma_YS)

    ld = experimental_LD_treatment(ld, 5, False)
    elastic_region = elastic_region_determination_r2_max(ld, 10, False)
    ld, elastic_region = offset_LD_according_to_stiffness(ld, elastic_region)

    fracture = Fracture(specimen, elastic_region, ld, id_computation, list_test[i])

    # -------------------------------
    # Monte-Carlo uncertainties evaluations
    # -------------------------------
    rng = np.random.default_rng()
    specimen_u.crack_profile_dist = crack_profile_distribution(crack_profile, 0.001, 0.001)
    specimen_mc = specimen_u.sample(nbr_sample, rng)
    elastic_mc = elastic_region_distribution(ld, elastic_region).sample(nbr_sample, rng)
    mc = Fracture(specimen_mc, elastic_mc, ld, id_computation)

    fractures.append(fracture)
    fractures_mc.append(mc)

    i += 1

# -------------------------------
# Abaqus
# -------------------------------
if len(list_abaqus) > 0:
    print("="*60)
    print(" Abaqus " + str(list_abaqus[0]) + " analysis ")
    print("="*60)

    # -------------------------------
    # Load data, treat and compute SIF and J-integral
    # -------------------------------
    print("Analysis of fracture")
    ld = abaqus_LD_reader(path_abaqus + "charpy_" + list_abaqus[0] + "micron_L=1500micron_R=2mm\\load_disp.rpt")
    #ld = abaqus_LD_reader("data_test\\load_disp.rpt")
    elastic_region = elastic_region_determination_r2_method(ld, 3, 0.999, False)

    fracture_abaqus = Fracture(specimen, elastic_region, ld, id_computation)

# -------------------------------
# Analysis and plot Master Curve
# -------------------------------
mc = MasterCurve(fractures, T, percentile, fractures_mc)
mc.plot_master_curve(True)
if len(list_abaqus) > 0:
    mc.plot_list_K(fracture_abaqus)
else:
    mc.plot_list_K()
fig, ax = mc.plot_ld_curves()
if len(list_abaqus) > 0:
    ax.plot(fracture_abaqus.ld.disp, fracture_abaqus.ld.load, color="black", linestyle="-.", label="Abaqus")
    ax.legend()

logger = Logger("txt", "report/MasterCurve.txt")
log_master_curve_uncertainties(mc, logger, True)


print("="*60)
print(" Analysis completed")
print("="*60)
plt.show()