import lj
import ljmod
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation

# LJ system parameters
natom = 200       # number of atoms
rho = 0.5         # number density
T = 1.9           # temperature
rc = 2.5          # cutoff radius
eps = 1.0         # unity in LJ unit
sigma = 1.0       # unity in LJ unit

# Metropolis parameters
ns = 500           # sampling step 
nburn = 0          # initial burn-in step
nskip = 5         # sampling intervals
h = 0.5            # trial step size
nbins = 150        # pair distribution function resolution
gplt = True       # plot the pair distribution function
snap = False       # write the snapshots of the coordinates 

#####################

def run():
    global gpool
    sim = lj.lj(natom, rho, T, rc, eps, sigma)
    sim.run(ns, nburn, nskip, h, nbins, snap, gplt)
    gpool = sim.gpool    

def pressure():
    ng = 10 
    prho = np.zeros((ng,2))
    i = 0
    for rho in np.linspace(0.1, 0.8, ng):
        sim = lj.lj(natom, rho, T, rc, eps, sigma)
        sim.run(ns, nburn, nskip, h, nbins, snap, gplt)
        prho[i] = rho, sim.pavg
        i += 1
    np.savetxt('P_rho_'+str(natom)+'.dat', prho)

def ind_avg():
    Cl = np.zeros((10,2))
    i = 0
    nexpr = 5
    C = 0                  # temporary 
    sim = lj.lj(natom, rho, T, rc, eps, sigma)
    for nskip in np.arange(1,20,2):
        for k in xrange(nexpr):
            sim.run(ns, nburn, nskip, h, nbins, snap, gplt)
            C += sim.Cl
        C /= nexpr
        Cl[i] = nskip, C
        i += 1
    np.savetxt('ind_avg_'+str(natom)+'.dat', Cl)

def animate():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.autoscale(tight=True)
    ims = []
    for i in xrange(1,ns+2):
        im = plt.plot(gpool[:,0], gpool[:,i], 'b')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=40, blit = True)
    plt.show()

if __name__ == "__main__":
    run()
    #animate()
