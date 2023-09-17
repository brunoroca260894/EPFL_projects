#!/usr/bin/python3

# to place 'npart' particles on a simple cubic 
# lattice with density 'rho'

from numba import jit

#parameters section begin
import time
start_time = time.time()

npart=200
rho=0.5
temp=2.0
sig=1
rcut=5.0
pi=3.1415926
nsamp = 25
eps=1
mass=1
beta=1/temp
ndispl=50 # number of displacement attempts
dmax=0.1   # maximal displacement
nequil=10000 # number of equilibration cycles
lmax=100000  # number of production cycle
nbins=30
#parameters section end

import math
import random
import numpy as np
nl=int(npart**(1.0/3))+1
if nl==0:
   nl=1
box=(npart/rho)**(1.0/3) #size of the cubic lattice
delta=box*1.0/nl
rcut=min(box/2.0,rcut)
vol=npart*1.0/rho

lx=[]
ly=[]
lz=[]
ipart=0

for i in range (nl) :
  for j in range (nl):
    for k in range (nl):
      if ipart < npart :
        X=i*delta
        Y=j*delta
        Z=k*delta
        lx.append(X)  
        ly.append(Y)
        lz.append(Z)
        ipart += 1

lx=np.array(lx)
ly=np.array(ly)
lz=np.array(lz)

# add random numbers
delta=delta/10000.0
for ipart in range (npart):
   lx[ipart] += delta*(random.uniform(0,1)-0.5)
   ly[ipart] += delta*(random.uniform(0,1)-0.5)
   lz[ipart] += delta*(random.uniform(0,1)-0.5)

print ("Initialization on a lattice: particles placed on a lattice")

# tail correction for energy
sig3=sig**3
ri3=sig3*1.0/(rcut**3)
coru = 2*pi*eps*4*(rho*sig3)*(ri3*ri3*ri3/9-ri3/3)

# tail correction for pressure
sig3=sig**3
ri3=sig3*1.0/(rcut**3)
corp = 4*pi*eps*4*(rho**2)*sig3*(2*ri3*ri3*ri3/9-ri3/3)

@jit(nopython=True)
def energy (lx,ly,lz) : #energy function, L-J potential
  etot=0
  vir=0
  for i in range (npart) :
    for j in range (npart) :
      if j > i :
        # periodic boundary condition
        dis2=min ((lx[i]-lx[j])**2, (abs(lx[i]-lx[j])-box)**2 )
        dis2 += min ((ly[i]-ly[j])**2, (abs(ly[i]-ly[j])-box)**2 )
        dis2 += min ((lz[i]-lz[j])**2, (abs(lz[i]-lz[j])-box)**2 )
        rij=math.sqrt(dis2)
        if rij <=  rcut :
          r6i=(sig/rij)**6
          etot += 4*eps*(r6i**2-r6i)
          vir += 48*eps*(r6i**2-0.5*r6i)

  etail=npart* coru 
  etot += etail # tail correction
  press = rho/beta + vir/(3*vol)
  press += corp
  return etot,press

@jit (nopython=True)
def esingle (ipick, lx1, ly1, lz1, lx,ly,lz) :
  e1=0
  for j in range (npart) :
      if j != ipick :
        # periodic boundary condition
        dis2=   min ((lx1[ipick]-lx[j])**2, (abs(lx1[ipick]-lx[j])-box)**2 )
        dis2 += min ((ly1[ipick]-ly[j])**2, (abs(ly1[ipick]-ly[j])-box)**2 )
        dis2 += min ((lz1[ipick]-lz[j])**2, (abs(lz1[ipick]-lz[j])-box)**2 )
        rij=math.sqrt(dis2)
        if rij <=  rcut :
          e1 += 4*eps*((sig/rij)**12-(sig/rij)**6)
  return e1

@jit (nopython=True)
def gofr (lx,ly,lz) :  # to caulate g(r)
  lgofr=[]
  lgofr.append(0)
  dl=box*1.0/nbins/4
  for il in range (1,nbins) :
   neighb=0
   di=il*2*dl
   for i in range (npart) :
    for j in range (npart) :
      if j !=i :
        # periodic boundary condition
        dis2=   min ((lx[i]-lx[j])**2, (abs(lx[i]-lx[j])-box)**2 )
        dis2 += min ((ly[i]-ly[j])**2, (abs(ly[i]-ly[j])-box)**2 )
        dis2 += min ((lz[i]-lz[j])**2, (abs(lz[i]-lz[j])-box)**2 )
        rij=math.sqrt(dis2)
        if rij < di+dl and rij > di-dl :
            neighb +=1
   lgofr.append(neighb/npart/di**2/2/dl/4/pi/rho)
  return lgofr

name_energy = 'energy_'
name_pressure = 'pressure_'
g3=open(name_energy+str(nsamp),'w')
g4=open(name_pressure +str(nsamp),'w')

## Monte Carlo move

logprod=0
lx1=[]
ly1=[]
lz1=[]
for i in range (npart):
  lx1.append(lx[i])
  ly1.append(ly[i])
  lz1.append(lz[i])

lx1=np.array(lx1)
ly1=np.array(ly1)
lz1=np.array(lz1)

#@jit (nopython=True)
def move (dmax, frac) :
 naccpt=0
 global lx, ly, lz
 for i in range (ndispl) :
  # -- select a particle at random
  ipick=int(random.uniform(0,npart))
  # -- calculate energy of old configuration.
  # -- give the particle a random displacement 
  ax=lx1[ipick]
  ay=ly1[ipick]
  az=lz1[ipick]
  lx1[ipick] += dmax*(random.uniform(0,1)-0.5)
  ly1[ipick] += dmax*(random.uniform(0,1)-0.5)
  lz1[ipick] += dmax*(random.uniform(0,1)-0.5)
 # keep the particle in the box
  if lx1[ipick] < 0 :
     lx1[ipick] += box
  if ly1[ipick] < 0 :
     ly1[ipick] += box
  if lz1[ipick] < 0 :
     lz1[ipick] += box
  
  if lx1[ipick] > box :
     lx1[ipick] -= box
  if ly1[ipick] > box :
     ly1[ipick] -= box
  if lz1[ipick] > box :
     lz1[ipick] -= box

# -- calculate energy difference between new configuration
#     and old configuration.
  e1=esingle (ipick, lx, ly, lz, lx, ly, lz)
  e2=esingle (ipick, lx1, ly1, lz1, lx, ly, lz)
  ediff=e2-e1

  if random.uniform(0,1) < math.exp(-beta*ediff) :
    lx[ipick]=lx1[ipick]
    ly[ipick]=ly1[ipick]
    lz[ipick]=lz1[ipick]
    naccpt += 1
  else :
    lx1[ipick]=ax
    ly1[ipick]=ay
    lz1[ipick]=az
         # acceptance rate
 raccp=naccpt*1.0/ndispl # accpetance rate
 frac=raccp
 if logprod == 0 :
  return frac
 else :
  lx2=np.array(lx)
  ly2=np.array(ly)
  lz2=np.array(lz)
  ep=energy(lx2,ly2,lz2)
  g3.write (str(ep[0]/(npart*1.0))+'\n')
  g4.write (str(ep[1])+'\n')
  return frac, ep

# adjust maximum displacement to get ~50% acceptance rate
print ('equilibration cycles')

frac=0.5
lfrac5=[]
for i in range (nequil) :
  naccpt=0
  if math.fmod (i+1,5)==0 :
    # update dmax after every 5 cycles
    dold=dmax
    dmax=dold*(frac/0.5)
    # limit the change in dmax
    if dmax/dold > 1.5 :
      dmax=dold*1.5
    if dmax/dold < 0.5 :
      dmax=dold*0.5
    if dmax > box/2.0 :
      dmax=box/2.0

  frac5= move (dmax, frac) 
  lfrac5.append(frac5)
  frac=sum(lfrac5)/len(lfrac5) 
  if math.fmod (i+1,200)==0 :
    print ('cycle', i+1, 'frac',round(frac,4),'dmax',round(dmax,4) )
  # update dmax for every 5 cycles 

print ('production cycles')
logprod=1
lener=[]
lpre=[]

for i in range (lmax) :
 logprod=0
 a1=move (dmax, frac)
 if math.fmod(i+1,nsamp)==0 :
  logprod=1
  a1=move (dmax, frac)

g3.close()
g4.close()
print("--- %s seconds ---" % (time.time() - start_time))

#%%
import matplotlib.pyplot as plt
sigma = 3.4*10**-10
time_SI = (sigma)*(  (120* 1.38*10**-23)/ (6.69*10**-26) )**(-1/2)
kb = 1.38064*10**(-23)
eps = 120 * kb

file_data= np.loadtxt('energy_10')

size = file_data.shape[0]
tim = np.arange(0, size) * time_SI
plt.plot(tim[::20], eps*(file_data)[::20], lw = 1);
plt.title('energy');
plt.xlabel('time in s');
plt.ylabel('energy J');
plt.savefig('energy_step_blocking' + '.eps', dpi = 300);
plt.show()