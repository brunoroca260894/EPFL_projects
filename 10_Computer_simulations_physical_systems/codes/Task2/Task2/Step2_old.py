import matplotlib.pyplot as plt
from MD import *
from scipy.signal import find_peaks

plt.rc("legend", framealpha=None)
plt.rc("legend", edgecolor='black')
plt.rc("font", family="serif")

nsteps = 2000
dt = 0.0046
N, L, pos, vel = read_pos_vel('sampleT94.4.dat')

# Run MD simulation
output = run_NVE(pos, vel, L, nsteps, N, dt)

#%%

# Plot g(r)
# to measure the peaks of g(r)
r_index = np.argsort(-output['gofr']['g'])
gr_values = output['gofr']['r'][r_index]

peaks, _ = find_peaks(output['gofr']['g'], distance=65)
values_max_gr = output['gofr']['g'][ peaks[:3] ]
values_max_r = output['gofr']['r'][ peaks[:3] ]
 
plt.plot(output['gofr']['r'],output['gofr']['g'], lw = 0.8, c = 'b')
plt.plot(output['gofr']['r'][  peaks[:3] ], output['gofr']['g'][  peaks[:3] ] , 
         ls ='none', marker ='x', c = 'r')
plt.title(r'$g(r)$')
plt.xlabel(r'$r/\sigma$')
plt.ylabel(r'$g(r)$')
plt.savefig('gr_rahman'+'.eps', dpi = 300)
plt.show()

# Plot S(k)
# to measure the peaks of S(k)
k_index = np.argsort(-output['sofk']['s'])
Sk_values = output['sofk']['k'][k_index]

peaks_sk, _ = find_peaks(output['sofk']['s'], distance=85)
values_max_sk = output['sofk']['s'][ peaks_sk[:4] ]
values_max_k = output['sofk']['k'][ peaks_sk[:4] ]

plt.plot(output['sofk']['k'],output['sofk']['s'], lw = 0.8, c = 'b')
plt.plot(output['sofk']['k'][  peaks_sk[:4] ], output['sofk']['s'][  peaks_sk[:4] ] , 
         ls ='none', marker ='x', c = 'r')
plt.title(r'$S(k)$')
plt.xlabel(r'$k$')
plt.ylabel(r'$S(k)$')
plt.savefig('Sk_rahman_FT'+'.eps', dpi = 300)
plt.show()

# Write g(k) into a file
np.savetxt('gofr.dat',np.column_stack((output['gofr']['r'],output['gofr']['g'])))

# Write S(k) into a file
np.savetxt('sofk.dat',np.column_stack((output['sofk']['k'],output['sofk']['s'])))

#%%
# Sk using direct sampling
# Run MD simulation 