import matplotlib.pyplot as plt
from scipy.stats import linregress
from MD import *

nsteps = 450
dt = 0.002
Nruns = 12
cut = 200

# Step up an array for time axis
t = np.linspace(1,nsteps,nsteps)*dt

# Empty variable for output
msd_mean=np.zeros(nsteps)
vacf_mean=np.zeros(nsteps)
D_msd=[]
D_vacf=[]

# Read equilibrated structure from a file
N, L, pos, vel = read_pos_vel('sampleT94.4.dat')
output = {'pos':pos, 'vel': vel}

for i in range(Nruns):

    # Run MD simulation
    output = run_NVE(output['pos'], output['vel'], L, nsteps, N, dt)
    msd_mean += output['msd']/Nruns
    vacf_mean += output['vacf']/Nruns

    # MSD
    slope, intercept, r, p, se = linregress(t[cut:],output['msd'][cut:])
    D_msd.append(slope/6*3.4E-10**2*4.6286E+11*1E+4)

    # VACF
    D_vacf.append(sum(output['vacf']*dt*3.4**2*4.6286*10**-5))

    # Write MSD and VACF into a file
    np.savetxt('sample.dat%i.samp'%i, np.column_stack((t, output['msd'] ,output['vacf'])))


# Plot averaged MSD
plt.plot(t,msd_mean)
slope, intercept, r, p, se = linregress(t[cut:],msd_mean[cut:])
plt.plot(t,slope*t+intercept, color='red')
plt.show()

# Plot averaged VACF
plt.plot(t,vacf_mean)
plt.show()

# Write diffusion coefficients into a file
np.savetxt('Dcoeff.dat', np.column_stack((D_msd, D_vacf)))

