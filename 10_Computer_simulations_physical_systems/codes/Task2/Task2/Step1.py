import matplotlib.pyplot as plt
from MD import *

plt.rc("legend", framealpha=None)
plt.rc("legend", edgecolor='black')
plt.rc("font", family="serif")

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

#############################################################

#%%
# From this point and on, we scale the dt variable by a factor given
# by the following value; this is apply for all dt from here on 
scale_variables = 1
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

# Step 1.3
# Compute velocities
#############################################################

nsteps = 200
dt = 0.0046

# Perform simulation starting from the output of a previous run
print('point 1.3 ')
output = run_NVE(output['pos'], output['vel'], L, nsteps, N, dt)

# Step 1.4
# Change T
#############################################################
nsteps = 200 * scale_variables
dt = 0.0046/scale_variables
T = 0.7867          # requested temperature

# Change T 
print('change T')
output = run_NVE(output['pos'], output['vel'], L, nsteps, N, dt, T)

# output.keys()

# Equilibrate
#############################################################
nsteps = 800
dt = 0.0046

# Equilibrate
print('equilibrate')
output = run_NVE(output['pos'], output['vel'], L, nsteps, N, dt)

# Write positions and velocities into a file
dump_pos_vel('sampleT94.4.dat', output['pos'], output['vel'], N, L)

# # Plot total energy vs step
# plt.plot(output['nsteps'],output['EnKin']+output['EnPot'])
# plt.title('total energy')
# plt.xlabel('step')
# plt.show()

#%%
### SI units
sigma = 3.4*10**-10
time_SI = (sigma)*(  (120* 1.38*10**-23)/ (6.69*10**-26) )**(-1/2)
kb = 1.38064*10**(-23)
eps = 120 * kb

# Plot energy vs step
#plt.figure(figsize=(10,7));
tim = np.array([output['nsteps']]).reshape((-1, 1))
plt.plot(tim*time_SI, eps*(output['EnKin']+output['EnPot']), lw = 1);
plt.title('energy');
plt.xlabel('time in s');
plt.ylabel('energy J');
plt.savefig('energy_step1' + '.eps', dpi = 300);
plt.show()

# Plot temperature vs step
#plt.figure(figsize=(10,7));
plt.plot(tim*time_SI, 120*output['EnKin']*2/3, lw  =1 );
plt.title('temperature');
plt.xlabel('time in s');
plt.ylabel('Temperature K');
plt.savefig('temperature_step1' + '.eps', dpi = 300);
plt.show()
