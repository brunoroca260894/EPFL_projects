import matplotlib.pyplot as plt
import numpy as np
from MD import *

nsteps = 2000
dt = 0.0046
N, L, pos, vel = read_pos_vel('sampleT94.4.dat')

# Run MD simulation
output = run_NVE(pos, vel, L, nsteps, N, dt)

# Plot g(r)
plt.plot(output['gofr']['r'],output['gofr']['g'])
plt.show()

# Plot S(k)
plt.plot(output['sofk']['k'],output['sofk']['s'])
plt.show()

# Write g(k) into a file
np.savetxt('gofr.dat',np.column_stack((output['gofr']['r'],output['gofr']['g'])))

# Write S(k) into a file
np.savetxt('sofk.dat',np.column_stack((output['sofk']['k'],output['sofk']['s'])))


###################
# Direct Sampling #
###################

N, L, pos, vel = read_pos_vel('sampleT94.4.dat')

# Run MD simulation with direct sampling on
output = run_NVE(pos, vel, L, nsteps, N, dt,direct_sofk=True)
# Plot S(k) direct
plt.plot(output['sofk_direct']['qvec'],output['sofk_direct']['sofk'],'o')
plt.show()
# Write S(k) direct into a file
np.savetxt('sofk-direct.dat',np.column_stack((output['sofk_direct']['qvec'
   ],output['sofk_direct']['sofk'])))