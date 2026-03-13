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

# -------------------------------
# Path to files
# -------------------------------
path = "C:\\Users\\rotunn_n\\Documents\\PDM\\data\\3_points_bending"

list_test = [1, 2, 3, 4, 6, 7, 8]
#list_test = [3, 4, 6, 7]
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

# -------------------------------
# Specimen
# -------------------------------
specimen_u = SpecimenDistribution(
    W = W,       # width [mm]
    W_u = 0.001,    # uncertainty in width [mm]
    S = S,      # span [mm] (4*W)
    S_u = 0.005,    # uncertainty in span [mm]
    B = B,       # thickness [mm]
    B_u = 0.001,    # uncertainty in thickness [mm]
    B_N = B_N,     # net thickness [mm]
    B_N_u = 0.001,  # uncertainty in net thickness [mm]
    nu = nu,      # Poisson ratio [-]
    E = E,    # Young modulus [MPa]
    E_u = 10,    # uncertainty in Young modulus [MPa]
    eta_pl = eta_pl,  # eta_pl [-],
    sigma_YS = sigma_YS,
    sigma_YS_u = 1,
    crack_profile_dist = None
)

# -------------------------------
# Compute load-displacement
# -------------------------------
ld_list = []
legend_ld_list = []
K_Jc_list = []

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
    logger = Logger("txt", test[2])
    specimen = Specimen(W, S, B, B_N, crack_profile.initial_crack_length(), nu, E, eta_pl, sigma_YS)

    ld = experimental_LD_treatment(ld, 5, False)
    elastic_region = elastic_region_determination_r2_max(ld, 10, False)
    ld, elastic_region = offset_LD_according_to_stiffness(ld, elastic_region)

    fracture = Fracture(specimen, elastic_region, ld, id_computation)

    fracture.plot_details(True, test[3])
    #fracture.log(logger)

    # -------------------------------
    # Compute uncertainties
    # -------------------------------
    print("Compute uncertainties")
    crack_profile_u = crack_profile_distribution(crack_profile, 0.01, 0.01)
    specimen_u.crack_profile_dist = crack_profile_u
    elastic_u = elastic_region_distribution(ld, elastic_region)

    rng = np.random.default_rng()

    mc = FractureMC(specimen_u.sample(nbr_sample, rng), elastic_u.sample(nbr_sample, rng), ld, id_computation)
    #mc.plot_mc_results(100)

    log_fracture_with_uncertainties(logger, fracture, mc, specimen_u, elastic_u)
    print("Report written")

    #plt.show()
    #plt.close('all')

    ld_list.append(ld)
    legend_ld_list.append("Test " + str(list_test[i]) + " $K_{Jc}$ = " + "{:.0f}".format(fracture.K_Jc*10**-1.5))
    K_Jc_list.append(fracture.K_Jc*10**-1.5)

    i += 1

plot_comparison_LD(ld_list, legend_ld_list)

K_Jc = np.array(K_Jc_list)
K_Jc_mean = np.mean(K_Jc)
K_Jc_std = np.std(K_Jc)
bar_label = []
for i in list_test:
    bar_label.append("Test " + str(i))
fig = plt.figure()
ax = fig.subplots()
p = ax.bar(bar_label, np.array(K_Jc_list), edgecolor = "black", color="skyblue", label="$K_{Jc}$ Tests")
ax.bar_label(p, label_type='center')
ax.axhline(K_Jc_mean, label="$K_{Jc}$ mean = "+"{:.0f}".format(K_Jc_mean), color="red", linestyle="--")
ax.axhline(K_Jc_mean + K_Jc_std, color="black", linestyle="-.", label="$K_{Jc}$ std = "+"{:.0f}".format(K_Jc_std))
ax.axhline(K_Jc_mean - K_Jc_std, color="black", linestyle="-.")
ax.set_ylabel("$K_{Jc}$ MPa mm$^{0.5}$")
ax.legend()

print("="*60)
print(" Analysis completed")
print("="*60)

plt.show()