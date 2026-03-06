from tools.reader import *
from tools.LoadDisplacement import *
from tools.plt_spec import *
from tools.Specimen import *
from tools.ElasticRegion import *
from tools.Fracture import *

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

specimen = Specimen(W, S, B, B_N, a0, nu, E, eta_pl)

ld = experiment_LD_reader("data_test/3pointbending.csv")
ld = experimental_LD_treatment(ld, 5, False)
elastic_region = elastic_region_determination_r2_max(ld, 10, False)
fracture = Fracture(specimen, elastic_region, ld, id_computation)
fracture.print_all()
fracture.plot_details()
#fracture.report("experiment.txt")

ld = abaqus_LD_reader("data_test/load_disp.rpt")
elastic_region = elastic_region_determination_r2_method(ld, 3, 0.999, False)
fracture = Fracture(specimen, elastic_region, ld, id_computation)
fracture.print_all()
fracture.plot_details()
#fracture.report("simulation.txt")

plt.show()