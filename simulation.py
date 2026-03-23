import numpy as np

# Model
Lx = 3  # [mm] dimension x
Ly = 4  # [mm] dimension y
Lz = 27 # [mm] dimension z

dL = 0.1 # [mm] Delta length

I = 0.04 # [A] Current

t_final = 6*3600 # [s] Charging time
dt = 1 # [s] Delta time

efficiency = 0.1 # Efficiency of entering the bulk
F = 96485.3321 # [s·A/mol] Farraday constant
D = 0.1 # Diffusion coefficient [TODO]

# Compute some values
surf = 2*Lx*Ly + 2*Lx*Lz + 2*Ly*Lz
j = I/surf
JB = efficiency*j/F
t = 0

# Create cells
C = np.zeros((int(Lx/dL), int(Ly/dL), int(Lz/dL)))
C_old = C.copy()
dim = np.shape(C)

print(dim, "total nbr cells:", dim[0]*dim[1]*dim[2])
print(f"Current density j={j*1000:.6f} mA/mm^2")

# Run simulation
while t < t_final:
    C_old, C = C, C_old

    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                C[i,j,k] = C_old[i,j,k]
                if i == 0:
                    C[i,j,k] += D*dt/dL**2 * (C_old[1,j,k] - C_old[0,j,k]) + JB*dt/dL
                elif i == dim[0]-1:
                    # [TODO]


    t += dt 