from tools.reader import *
from tools.LoadDisplacement import *
from tools.plt_spec import *
from tools.Specimen import *
from tools.ElasticRegion import *
from tools.Fracture import *
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
eta_pl = 1.9 # ??? parameters to compute the J-elastic. It is 1.9 because we use the load-displacement curve

id_computation = -1

path = "C:\\Users\\rotunn_n\\Documents\\PDM\\data\\3_points_bending"
test1 = "sample1_m120C.csv"
test2 = "sample2_m120C.csv"
test3 = "sample3_m120C.csv"
test4 = "sample4_m120C.csv"

specimen = Specimen(W, S, B, B_N, a0, nu, E, eta_pl)

full_path = os.path.join(path, test1)
ld = experiment_LD_reader(full_path)
plot_LD(ld)
ld = experimental_LD_treatment(ld, 5, False)
elastic_region = elastic_region_determination_r2_max(ld, 10, False)
ld, elastic_region = offset_LD_according_to_stiffness(ld, elastic_region)
fracture = Fracture(specimen, elastic_region, ld, id_computation)
fracture.print_all()
fracture.plot_details(True, "fig/test1.svg")
fracture.report("report/test1.txt")

#ld2 = abaqus_LD_reader("data_test/load_disp.rpt")
#elastic_region = elastic_region_determination_r2_method(ld2, 3, 0.999, False)
#fracture = Fracture(specimen, elastic_region, ld2, id_computation)
#fracture.print_all()
#fracture.plot_details()
#fracture.report("simulation.txt")

#plot_comparison_LD([ld1, ld2], ["Experiment", "Abaqus"])

plt.show()