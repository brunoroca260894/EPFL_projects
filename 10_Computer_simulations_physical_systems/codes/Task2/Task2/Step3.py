import matplotlib.pyplot as plt
from scipy.stats import linregress
from MD import *
from scipy import integrate

nsteps = 450*2
dt = 0.0005
Nruns = 20
cut = 2*200

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
    print('running simulation: ', i+1)
    output = run_NVE(output['pos'], output['vel'], L, nsteps, N, dt)
    msd_mean += output['msd']/(Nruns)
    vacf_mean += output['vacf']/(Nruns)
    
    # MSD
    slope, intercept, r, p, se = linregress(t[cut:],output['msd'][cut:])
    D_msd.append(slope/6*3.4E-10**2*4.6286E+11*1E+4)

    # VACF
    D_vacf.append(sum(output['vacf']*dt*3.4**2*4.6286*10**-5))

    # Write MSD and VACF into a file
    np.savetxt('sample.dat%i.samp'%i, np.column_stack((t, output['msd'] ,output['vacf'])))

#%%
# Plot averaged MSD
sigma = 3.4*10**-10
time_SI = (sigma)*(  (120* 1.38*10**-23)/ (6.69*10**-26) )**(-1/2)
slope, intercept, r, p, se = linregress(t[cut:],msd_mean[cut:])
slope_SI = slope * sigma**2/ time_SI * (100**2)
D_SI = slope_SI/6
print('msd, D in SI cm2/S ', D_SI)

# plt.plot(t*time_SI, msd_mean* sigma**2, label='average over 20 runs')
# plt.plot(t*time_SI, slope*t*sigma**2 + intercept*sigma**2, color='red', label='linear fit')
plt.plot(t, msd_mean, label='average over 20 runs')
plt.plot(t, slope*t + intercept, color='red', label='linear fit')
plt.title(r'$MSD$')
plt.xlabel('time in '+r'$\Delta u$')
plt.ylabel('mean square displacement ' + r'$\sigma^{2}$')
plt.legend()
#plt.savefig('msd_linearFit_LJ'+'.eps', dpi = 300)
plt.show()

# Plot averaged VACF
plt.plot(t*time_SI,vacf_mean)
plt.title(r'$VACF$')
plt.xlabel('time in ' + r'$s$')
plt.ylabel('Velocity auto-correlation')
#plt.savefig('VACF_SI'+'.eps', dpi = 300)
plt.show()

# Write diffusion coefficients into a file
np.savetxt('Dcoeff.dat', np.column_stack((D_msd, D_vacf)))