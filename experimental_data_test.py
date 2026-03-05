from tools.load_disp_tool import *
from tools.plt_spec import *

init_plt(latex=False)
create_sns_palette()

t, RF2, U2 = load_load_disp_data("data_test/3pointbending.csv", "experiment")
treat_experimental_data(t, RF2, U2)

plt.show()