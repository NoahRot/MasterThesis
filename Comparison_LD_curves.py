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

def single_T_master_curve_analysis(fractures : list[Fracture], T : float):
    K_min = 20
    Bx = 25.4
    nbr_uncencored_data = 0

    # Load data
    K_Jci = []
    K_Jc_lim = []
    for f in fractures:

        # Check limits
        if f.K_Jc < f.specimen.K_Jc_lim:
            nbr_uncencored_data += 1
            K_Jci.append(f.K_Jc*10**-1.5)
        else:
            K_Jci.append(f.specimen.K_Jc_lim*10**-1.5)

        K_Jc_lim.append(f.specimen.K_Jc_lim*10**-1.5)

    # Check if it is possible to compute the master curve
    if nbr_uncencored_data == 0:
        print("ERROR: No K_Jc in acceptable limit. Impossible to compute master curve.")
        raise ValueError("ERROR: No K_Jc in acceptable limit. Impossible to compute master curve.")

    K_Jci = np.array(K_Jci)
    K_Jc_lim = np.array(K_Jc_lim)

    # size-adjusted to 1T thickness
    K_Jc1T = K_min + (K_Jci - K_min)*(specimen_u.B/Bx)**0.25

    # Compute T0 of the master curve temperature
    K0 = (np.sum((K_Jc1T - K_min)**4/nbr_uncencored_data))**0.25 + K_min
    K_Jc_med =  K_min + 0.91*(K0 - K_min)
    T_0Q = T - (1/0.019)*np.log((K_Jc_med - 30)/70)

    # Check T0 validity
    valid_T0Q = T-T_0Q > -50 and  T-T_0Q < 50

    return K_Jc1T, K0, K_Jc_med, T_0Q, valid_T0Q, nbr_uncencored_data

def master_curve_tolerance_bounds(T : float, T0 : float, percentile : float = 0.05):
    K_Jc_percentile = 20.0 + (np.log(1.0/(1.0 - percentile)))**0.25 * (11.0 + 77.0*np.exp(0.019*(T-T0)))
    T0_percentile = T - (1/0.019)*np.log((K_Jc_percentile - 30)/70)
    return K_Jc_percentile, T0_percentile

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

T = -120 # Temperature of the tests [°C]

# -------------------------------
# Path to files
# -------------------------------
path = "C:\\Users\\rotunn_n\\Documents\\PDM\\data\\3_points_bending"

list_test = [1, 3, 4, 6, 7, 8, 9, 10, 11, 12]
#list_test = [13]
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
ld_list = []
legend_ld_list = []
K_Jc_list = []
K_Jc_lim_list = []
K_Jc_err_list = []
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
    #logger = Logger("txt", test[2])
    specimen = Specimen(W, S, B, B_N, crack_profile.initial_crack_length(), nu, E, eta_pl, sigma_YS)

    ld = experimental_LD_treatment(ld, 5, False)
    elastic_region = elastic_region_determination_r2_max(ld, 10, False)
    ld, elastic_region = offset_LD_according_to_stiffness(ld, elastic_region)

    fracture = Fracture(specimen, elastic_region, ld, id_computation, list_test[i])

    rng = np.random.default_rng()
    specimen_u.crack_profile_dist = crack_profile_distribution(crack_profile, 0.001, 0.001)
    specimen_mc = specimen_u.sample(nbr_sample, rng)
    elastic_mc = elastic_region_distribution(ld, elastic_region).sample(nbr_sample, rng)
    mc = Fracture(specimen_mc, elastic_mc, ld, id_computation)

    #fracture.plot_details()

    ld_list.append(ld)
    legend_ld_list.append("Test " + str(list_test[i]) + " $K_{Jc}$ = " + "{:.0f}".format(fracture.K_Jc*10**-1.5))
    K_Jc_list.append(fracture.K_Jc*10**-1.5)
    K_Jc_lim_list.append(fracture.specimen.K_Jc_lim*10**-1.5)
    K_Jc_err = compute_uncertainties(mc.K_Jc, 2.5)
    K_Jc_err_list.append(K_Jc_err[2]*10**-1.5)

    fractures.append(fracture)
    fractures_mc.append(mc)

    i += 1

# -------------------------------
# Abaqus
# -------------------------------
print("="*60)
print(" Abaqus " + str(list_test[0]) + " analysis ")
print("="*60)

# -------------------------------
# Load data, treat and compute SIF and J-integral
# -------------------------------
print("Analysis of fracture")
ld = abaqus_LD_reader(path_abaqus + "charpy_" + list_abaqus[0] + "micron_L=1500micron_R=2mm\\load_disp.rpt")
elastic_region = elastic_region_determination_r2_method(ld, 3, 0.999, False)

fracture = Fracture(specimen, elastic_region, ld, id_computation)
#fracture.plot_details()

# -------------------------------
# Analysis and plot of global results
# -------------------------------
mc = MasterCurve(fractures, T, 5, fractures_mc)
mc.plot_master_curve(True)
mc.plot_list_K(fracture)
fig, ax = mc.plot_ld_curves()
ax.plot(fracture.ld.disp, fracture.ld.load, color="black", linestyle="-.", label="Abaqus")
ax.legend()

logger = Logger("txt", "report/MasterCurve.txt")
#log_master_curve(mc, logger, True)
log_specimen(specimen, logger)
log_specimen_uncertainties(specimen, specimen_u.sample(nbr_sample, rng), logger)


print("="*60)
print(" Analysis completed")
print("="*60)
plt.show()