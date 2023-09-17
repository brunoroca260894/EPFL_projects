import matplotlib.pyplot as plt
from MD import *

# Step 1.1 
# Create a crystalline fcc structure
#############################################################

Ncells = 6          # Number of unit cells along each axis
lat_par = 1.7048    # Lattice parameter
L = lat_par*Ncells  # Size of the simulation box
N = 4*Ncells**3     # Number of atoms in the simulation box

# Generate fcc structure
pos, vel = crystal(Ncells, lat_par)

# Write positions and velocities into a file
dump_pos_vel('sample10.dat', pos, vel, N, L)

# Step 1.2 
# Run a test simulation 
#############################################################

nsteps = 200        # Number of steps
dt = 0.003          # Integration step

# Read crystal shape, positions and velocities from a file
N, L, pos, vel = read_pos_vel('sample10.dat')

# Perform simulation and collect the output into a dictionary
output = run_NVE(pos, vel, L, nsteps, N, dt)

# Write positions and velocities into a file
dump_pos_vel('sample11.dat', output['pos'], output['vel'], N, L)

'''
# Step 1.3
# Compute velocities
#############################################################

nsteps = 200
dt = 0.0046

# Perform simulation starting from the output of a previous run
output = run_NVE(output['pos'], output['vel'], L, nsteps, N, dt)


# Step 1.4
# Change T
#############################################################

nsteps = 200
dt = 0.0046
T = 0.7867          # requested temperature

# Change T 
output = run_NVE(output['pos'], output['vel'], L, nsteps, N, dt, T)

# Plot temperature vs step
plt.plot(output['nsteps'],output['EnKin']*2/3)
plt.show()

# Equilibrate
#############################################################

nsteps = 800
dt = 0.0046

# Equilibrate
output = run_NVE(output['pos'], output['vel'], L, nsteps, N, dt)


# Write positions and velocities into a file
dump_pos_vel('sampleT94.4.dat', output['pos'], output['vel'], N, L)


# Plot total energy vs step
plt.plot(output['nsteps'],output['EnKin']+output['EnPot'])
plt.show()
'''
