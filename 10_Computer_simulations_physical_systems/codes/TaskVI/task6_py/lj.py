"""Computational Physics I EPFL
Monte Carlo Metropolis simulation of a Lennard-Jones Liquid
Some Fortran routines in 'ljmod' are implied
Wei Chen 11.2011"""

import numpy as np
import ljmod
from random import random, randint
from math import pi, hypot, exp, fabs

class lj(object):
    def __init__(self, natom=100, rho=0.5, T=1, rc=2.5, eps=1.0, sigma=1.0):
        self.n = int(natom**0.333333)+1 # number of atoms per direction
        self.natom = self.n**3          # renormalized number of atoms
        self.L = (natom/rho)**0.333333  # box length
        self.T = T                      # temperature
        self.rho = rho
        self.d = self.L/self.n          # interatomic distance
        self.rc = rc
        self.eps = eps
        self.sigma = sigma
        self.V = self.L**3
        rinv3 = (sigma/rc)**3
        self.utail = 8.0*pi*eps*rho*(rinv3**3/9-rinv3/3)
        self.ptail = 4.0*pi*eps*(rho**2)*(sigma**3)*(rinv3**3/4.5-rinv3/3)
        self.atoms = self.lattice()

    def lattice(self):
        """ create a cubic lattice with the atoms displaced randomly 
        off from the ideal structure"""
        d = self.d
        n = self.n
        natom = self.natom
        atoms = np.zeros((natom, 3))
        for i in xrange(n):
            atoms[n**2*i:n**2*(i+1),0] = d/2 + d*i
        for i in xrange(n**2):
            for j in xrange(n):
                atoms[n*j+n**2*i:n*(j+1)+n**2*i,1] = d/2 + d*j
        for i in xrange(n**2):
            for j in xrange(n):
                atoms[j+n*i,2] = d/2 + d*j       
        atoms += np.random.rand(natom,3)*(d/6)
        return atoms

    def distmat(self, atoms):
        """setup a distance matrix"""
        D = np.zeros((self.natom, self.natom))
        for i in xrange(self.natom):
            r = np.abs(atoms[i]-atoms)                   # dx, dy, dz
            r = np.where(r < 0.5*self.L, r, r-self.L)    # PBC
            D[i] = r[:,0]**2+r[:,1]**2+r[:,2]**2
        return np.sqrt(D)

    def calcU(self, atoms):
        """calculate the LJ potential for the lattice 'atoms'"""
        eps = self.eps
        sigma = self.sigma
        rc = self.rc
        self.D = ljmod.distmat(atoms, self.L) 
        u = 0    # potential
        v = 0    # virial coeff.
        for i in xrange(self.natom-1):
            ind = np.where(self.D[i] <= rc)[0] 
            ind = np.extract(ind > i, ind)
            for j in ind:
                ri = sigma/self.D[i,j]
                u += ri**12 - ri**6
                v += ri**12 - 0.5*ri**6
        u *= 4.0*eps
        v *= 48.0*eps
        return u,v

    def dU(self, atoms, iat):
        """calculate the energy difference due to the displacement
           of one particle"""
        D = ljmod.distmat(atoms, self.L)
        du, dv = ljmod.delta(D, self.D0, self.rc, iat, self.eps, self.sigma)
        return du, dv, D

    def metropolis(self, atoms, h):
        """Metropolis algorithm"""

        atoms0 = np.copy(atoms)                        # keep old config.
        iat = randint(0,self.natom-1)                  # select an atom     
        atoms[iat] += (np.random.rand(3)-0.5)*h        # move the atom      
        atoms[iat][atoms[iat] > self.L] -= self.L                # PBC
        atoms[iat][atoms[iat] < 0] += self.L

        du, dv, D = self.dU(atoms, iat) 
        u1 = self.u0 + du
        v1 = self.v0 + dv

        if du < 0 or exp(-du/self.T) > random():
            self.u0 = u1
            self.v0 = v1
            self.D0 = D
            self.iaccept += 1
        else:
            self.atoms = atoms0

    def adapt(self, h):
        """adaptive h ==> acceptance rate = 50%"""
        ratio = 2.0*(self.iaccept+1)/(self.ii*self.nskip)
        h1 = h*ratio
        if h1 > 2.0:
            h1 = h
        return h1
    
    def press(self, v):
        """calculate pressure from second order virial coeff."""
        return self.natom*self.T/self.V + v/(3*self.V)

    def writepdb(self, atoms, f, head, prefix):
        """write the snapshots in pdb"""
        f.write(''.join((head,'\n')))
        pdb = np.hstack((prefix, atoms))
        np.savetxt(f, pdb, fmt='%-5s %5s %4s %9s %11s %7s %7s')
        f.write(''.join(('ENDMDL', '\n')))

    def run(self, ns=100, nburn=0, nskip=10, h=0.2, nbins=100, \
            snap=False, gplt=False):
        """main coordinator"""
        ntot = ns*nskip
        self.nskip = nskip
        self.ii = 0 
        self.iaccept = 0
        self.u0, self.v0 = self.calcU(self.atoms)          # old energy terms
        self.D0 = self.D                               # old D matrix
        self.nbins = nbins
        h0 = h
        self.uavg = 0                                  # <U>
        self.u2avg = 0
        self.pavg = 0                                  # <P>
        self.Cl = 0                                    # autocorrelation function

        if snap == True:        # write the coordinates at each sample step?
            fn = 'snapshot.pdb'
            prefix = np.empty((self.natom,4),dtype='a5')
            prefix[:,0] = 'ATOM'
            prefix[:,1] = '0'
            prefix[:,2] = 'Ar'
            prefix[:,3] = '0'
            self.prefix = prefix
            f = open(fn, 'w')
        
        if gplt == True:        # calculate g(r) at each sample step?
            self.gpool = np.zeros((nbins, ns+2))
            #self.gr(self.atoms)
            self.g = ljmod.gr(self.atoms, nbins, self.L, self.rho)
            self.gpool[:,0:2] = self.g

        if nburn > 0:           # burn-in process
            print "Burn-in in process..."
            for i in xrange(nburn*nskip):
                self.metropolis(self.atoms, h)
            self.iaccept = 0    # reset 
        
        u00 = (self.u0+self.utail)/self.natom    # for autocorrelation calculation
        uu = 0                                   # <U_n*U_{n+l}>

        for i in xrange(ntot-1):
            self.metropolis(self.atoms, h)
            if i%nskip == 0:
                self.ii += 1
                self.U = (self.u0+self.utail)/self.natom
                self.P = self.press(self.v0)+self.ptail
                self.uavg += self.U
                self.u2avg += self.U**2
                self.pavg += self.P
                uu += self.U*u00
                u00 = self.U

                print "{0:10} {1:12.4f} {2:12.4f}".format(self.ii, self.U, self.P)
                h = self.adapt(h)                
                head = ''.join(('MODEL'.ljust(10), str(self.ii)))
                self.writepdb(self.atoms, f, head, prefix) if snap == True else ''
                if gplt == True:
                    #self.gr(self.atoms)
                    self.g = ljmod.gr(self.atoms, nbins, self.L, self.rho)
                    self.gpool[:,self.ii+1] = self.g[:,1]
                    
        self.uavg /= self.ii
        self.pavg /= self.ii
        self.varu = fabs(self.u2avg - self.uavg**2)/self.ii
        self.Cl = (uu/self.ii - self.uavg**2) / self.varu

        print "# acceptance rate: {0:2}%".format(100.0*self.iaccept/ntot)
        print "# initial h: {0:5.2f} ==> final h: {1:5.2f}".format(h0, h)

    def gr(self, atoms):
        """pair distribution function"""
        nbins = self.nbins
        dr = 0.5*self.L/nbins           # L divided by nbins grids
        g = np.zeros((nbins,2))         # pair distribution function
        hist = np.zeros(nbins)
        self.D = ljmod.distmat(atoms, self.L)

        b = np.array(np.ceil(self.D/dr), dtype=np.int)
        bind = np.unique(b)[1::]   # skip self-interaction
        bind = np.extract(bind <= nbins, bind)
        for i in bind-1: 
            hist[i] = np.count_nonzero(b == i)
        #hist /= 2                       # get rid of double counting

        for i in xrange(nbins):
            r = (i+0.5)*dr
            p = hist[i]/(4*pi*(r**2)*dr)/(self.rho*self.natom)
            g[i] = r, p

        self.g = g
        self.hist = hist
