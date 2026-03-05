from tools.load_disp_tool import *
from tools.plt_spec import *

init_plt(latex=False)
create_sns_palette()

t, RF2, U2 = load_load_disp_data("data_test/3pointbending.csv", "experiment")
t, RF2, U2 = treat_experimental_data(t, RF2, U2, 5, True)
#elastic_end, stiffness, intercept_1 = elastic_region_determination_r2_method(RF2, U2, 60, 0.98, True)
elastic_end, stiffness, intercept_1 = elastic_region_determination_r2_max(RF2, U2, 10, True)

load = RF2
disp = U2
id_computation = -1
intercept_2 = -stiffness*disp[id_computation] + load[id_computation]
id_elastic_end = elastic_end

# Define the stiffness curves
x1 = np.linspace(0, (np.max(load)-intercept_1)/stiffness, 2)
x2 = np.linspace(-intercept_2/stiffness, disp[id_computation], 2)
y1 = stiffness*x1 + intercept_1
y2 = stiffness*x2 + intercept_2

# Define area curves
x = np.copy(disp)
y_up = np.copy(load)
x_c = -intercept_2/stiffness
idx = np.searchsorted(disp, x_c)
y_c = y_up[idx-1] + (y_up[idx]-y_up[idx-1])/(x[idx]-x[idx-1])*(x_c-x[idx-1])
x = np.insert(x, idx, x_c)
y_up = np.insert(y_up, idx, y_c)
y_down = stiffness*x + intercept_2
y_down = np.maximum(y_down, np.zeros_like(y_down))

# Plot the results
fig = plt.figure()
ax = fig.subplots()
ax.plot(disp, load, "bx", label="$L-\Delta$ curve")
ax.plot(disp[id_computation], load[id_computation], "rx", label="Computation point")
ax.vlines(disp[id_elastic_end], np.min(load), np.max(load), color="k", linestyles=":", label="End of elastic region")
ax.plot(x1, y1, "k--", label="Stiffness")
ax.plot(x2, y2, "k--")
ax.fill_between(x, y_down, y_up, alpha=0.2, label="$A_{pl}$")

ax.set_xlabel("$\Delta$ [μm]")
ax.set_ylabel("$L$ [μN]")
ax.set_title("$L-\Delta$ curve")
ax.legend()

plt.show()