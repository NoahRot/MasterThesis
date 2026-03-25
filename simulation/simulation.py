import numpy as np
import matplotlib.pyplot as plt

from plt_spec import *

# ============================================================
# Constant values
# ============================================================
F = 96485.3321 # [s·A/mol] Farraday constant
kB = 8.617333262e-5 # [eV/K] Boltzman constant
T_abs = -273.15 # [°C] Absolute zero
D0 = 2.52e-7 # [m^2/s] Constant diffusion value
K0 = 1.76e-1 # [mol/m^3/Pa^0.5] Constant solubility value
ED = 0.16 # [eV] Diffusion energy
EK = 0.27 # [eV] Solubility energy

# ============================================================
# Inpute for the model
# ============================================================
Lx = 3e-3   # [m] dimension x
Ly = 4e-3   # [m] dimension y
Lz = 27e-3  # [m] dimension z
dL = 0.1e-3 # [m] Delta length

I = 0.04 # [A] Current
T1 = 21 # [°C] Temperature of the sample during charging
T2 = -196 # [°C] Temperature of the sample during transport (in liquid nitrogen)

t_final = 6*3600 # [s] Charging time
dt = 1 # [s] Delta time

efficiency = 1 # [-] Efficiency of adsorption
k_r = 1e-4  # [m^4/mol/s] recombination coefficient (tune this)

print_freq = 100 #Frequency at which the program show the progression

# ============================================================
# Computed values
# ============================================================
D1 = D0*np.exp(-ED/(kB*(T1 - T_abs))) # [m^2/s] Diffusion coefficient for charging
D2 = D0*np.exp(-ED/(kB*(T2 - T_abs))) # [m^2/s] Diffusion coefficient for transport

K1 = K0*np.exp(-EK/(kB*(T1 - T_abs))) # [mol/m^3/Pa^0.5] Solubility for charging
K2 = K0*np.exp(-EK/(kB*(T2 - T_abs))) # [mol/m^3/Pa^0.5] Solubility for transport

Phi1 = D1*K1 # [mol/m/Pa^0.5/s] Permeability for charging
Phi2 = D2*K2 # [mol/m/Pa^0.5/s] Permeability for transport

surf = 2*Lx*Ly + 2*Lx*Lz + 2*Ly*Lz
J_dens = I/surf
JB = efficiency*J_dens/F/8 # Devide by 8 because we are simulating 1/8 of the sample
t = 0

# ============================================================
# Create specimen model
# ============================================================
C = np.zeros((int(0.5*Lx/dL) + 2, int(0.5*Ly/dL) + 2, int(0.5*Lz/dL) + 2)) # WARNING: Add cells at center boundary for ghost cell to use Neumann B.C. 
C_old = C.copy() # Previous concentration
dim = np.shape(C)

# Characteristic saturation concentration
C_sat = JB * Lz / D1

# Surface concentrations (ghost cells)
Cs_x = np.zeros((dim[1], dim[2]))  # surface at x = Lx/2
Cs_y = np.zeros((dim[0], dim[2]))  # surface at y = Ly/2
Cs_z = np.zeros((dim[0], dim[1]))  # surface at z = Lz/2

# ============================================================
# Print pre-processing values
# ============================================================
print("="*60)
print("Pre-processing values")
print("="*60)

print(f"Stability condition D dt/dL^2 < 1/6: D dt/dL^2={D1*dt/dL**2:.2f}")
if D1*dt/dL**2 < 1/6:
    print("STABLE")
else:
    print("UNSTABLE -> ABORT")
    quit()

# ============================================================
# Initialize matplotlib
# ============================================================
init_plt(latex=False)
create_sns_palette()

# ============================================================
# Record values
# ============================================================
times = []
H_total = []

# ============================================================
# Run simulation
# ============================================================
print("="*60)
print("Simulation")
print("="*60)

iter = 0
while t < t_final:

    # Print progress
    if iter%print_freq == 0:
        print(f"t={t}, t_final={t_final}")

    # Compute values
    D_dt_dL2 = D1*dt*dL**-2
    JB_dt_dL = JB*dt/dL

    # Swap
    C_old, C = C, C_old

    # Enforce symmetry. (Central B.C.)
    C_old[0,:,:] = C_old[1,:,:]
    C_old[:,0,:] = C_old[:,1,:]
    C_old[:,:,0] = C_old[:,:,1]

    # Start from old
    C[:] = C_old[:]

    # Bulk diffusion
    C[1:-1, 1:-1, 1:-1] += D_dt_dL2*(
        C_old[2:,   1:-1, 1:-1] + C_old[:-2,  1:-1, 1:-1] + # Diffusion along X
        C_old[1:-1, 2:,   1:-1] + C_old[1:-1, :-2,  1:-1] + # Diffusion along Y
        C_old[1:-1, 1:-1, 2:  ] + C_old[1:-1, 1:-1, :-2 ] - # Diffusion along Z
        6*C_old[1:-1, 1:-1, 1:-1]
    )

    # Surface dynamic
    # X surface 
    C1x = C_old[-2, 1:-1, 1:-1]  # first interior cells
    J_in_x = D1 * (Cs_x[1:-1, 1:-1] - C1x) / dL
    J_gen_x = JB * (1 - Cs_x[1:-1, 1:-1] / C_sat) # Smooth saturation (prevents divergence)

    Cs_x[1:-1, 1:-1] += dt * (J_gen_x - J_in_x - k_r * Cs_x[1:-1, 1:-1]**2) # Update surface concentration

    # Y surface
    C1y = C_old[1:-1, -2, 1:-1]
    J_in_y = D1 * (Cs_y[1:-1, 1:-1] - C1y) / dL
    J_gen_y = JB * (1 - Cs_y[1:-1, 1:-1] / C_sat)

    Cs_y[1:-1, 1:-1] += dt * (J_gen_y - J_in_y - k_r * Cs_y[1:-1, 1:-1]**2)

    # Z surface
    C1z = C_old[1:-1, 1:-1, -2]
    J_in_z = D1 * (Cs_z[1:-1, 1:-1] - C1z) / dL
    J_gen_z = JB * (1 - Cs_z[1:-1, 1:-1] / C_sat)

    Cs_z[1:-1, 1:-1] += dt * (J_gen_z - J_in_z - k_r * Cs_z[1:-1, 1:-1]**2)

    # --- Prevent negative values (important for stability)
    Cs_x = np.maximum(Cs_x, 0)
    Cs_y = np.maximum(Cs_y, 0)
    Cs_z = np.maximum(Cs_z, 0)

    C[-1, 1:-1, 1:-1] += D_dt_dL2 * (Cs_x[1:-1, 1:-1] - C_old[-1, 1:-1, 1:-1])
    C[1:-1, -1, 1:-1] += D_dt_dL2 * (Cs_y[1:-1, 1:-1] - C_old[1:-1, -1, 1:-1])
    C[1:-1, 1:-1, -1] += D_dt_dL2 * (Cs_z[1:-1, 1:-1] - C_old[1:-1, 1:-1, -1])

    # -----------------------

    # Record
    dV = dL**3
    times.append(t)
    H_total.append(np.sum(C[1:-1, 1:-1, 1:-1]) * dV)

    # Progression
    t += dt 
    iter += 1

# ============================================================
# Print results
# ============================================================

print(f"Dimension {dim}, total nbr cells: {dim[0]*dim[1]*dim[2]}")
print(f"Area: {surf*1e4} cm^2")
print(f"Current density j={J_dens*8:.6e} A/m^2")
print(f"Diffusion coefficient D={D1*1e4:.6e} cm^2/s")
print(f"Total Hydrogen content {H_total[-1]:.3e} mol")
print(f"Total concentration {np.sum(C[1:-1, 1:-1, 1:-1])*dV/(Lx*Ly*Lz):.3e} mol/m^3")

# ============================================================
# Plot results
# ============================================================

fig = plt.figure()
ax = fig.subplots()
ax.plot(times, H_total)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Total hydrogen [mol]")
ax.set_title("Hydrogen uptake vs time")

k_mid = dim[2] // 2

fig = plt.figure()
ax = fig.subplots()
im = ax.imshow(C[1:-1, 1:-1, k_mid], origin='lower', cmap='viridis',)
plt.colorbar(im, label="Concentration [mol/mm³]")
ax.set_title(f"Slice at z = {k_mid*dL*1000:.2f} mm")
ax.set_xlabel("x")
ax.set_ylabel("y")

i_mid = dim[0] // 2
j_mid = dim[1] // 2

z = np.arange(dim[2]) * dL

fig = plt.figure()
ax = fig.subplots()
ax.plot(z[1:-1], C[i_mid, j_mid, 1:-1])
ax.set_xlabel("z [mm]")
ax.set_ylabel("Concentration [mol/mm³]")
ax.set_title("Concentration profile along z")

plt.show()