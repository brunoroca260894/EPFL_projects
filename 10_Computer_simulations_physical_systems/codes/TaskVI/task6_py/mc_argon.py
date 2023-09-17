#!/usr/bin/python3

# to place 'npart' particles on a simple cubic 
# lattice with density 'rho'

from numba import jit
from scipy.signal import find_peaks
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#matplotlib.use ('Agg')

#parameters section begins
import time
start_time = time.time()

npart=864
rho=0.8072
temp=0.7833
sig=1
rcut=5.0
nsamp=10
pi=3.1415926
eps=1
beta=1/temp
ndispl=50 # number of displacement attempts
dmax=0.1   # maximal displacement
nequil=10000 # number of equilibration cycles
lmax=2000 # number of production cycle
nbins=100 # samping for g (r), use even number
#parameters section ends

import math
import random
import numpy as np

nl=int((npart/4.0)**(1.0/3))+1

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
                lx.append(X+0.5*delta)  
                ly.append(Y+0.5*delta)
                lz.append(Z)
                lx.append(X+0.5*delta) 
                ly.append(Y)
                lz.append(Z+0.5*delta)
                lx.append(X)            
                ly.append(Y+0.5*delta)
                lz.append(Z+0.5*delta)
    
                ipart += 4

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

def coru(r,rho) : # tail correction for energy
    sig3=sig**3
    ri3=sig3*1.0/(r**3)
    coru = 2*pi*eps*4*(rho*sig3)*(ri3*ri3*ri3/9-ri3/3)
    return coru

def energy(): #energy function, L-J potential
    etot=0
    vir=0
    global lx, ly, lz
    for i in range(npart) :
        for j in range(npart) :
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
    etail=npart* coru (rcut,rho)
    etot += etail # tail correction
    return etot

@jit (nopython=True)
def esingle(ipick, lx1, ly1, lz1, lx,ly,lz) :
    e1=0
    for j in range (npart):        
        if j != ipick :
            # periodic boundary condition
            dis2=   min ((lx1[ipick]-lx[j])**2, (abs(lx1[ipick]-lx[j])-box)**2 )
            dis2 += min ((ly1[ipick]-ly[j])**2, (abs(ly1[ipick]-ly[j])-box)**2 )
            dis2 += min ((lz1[ipick]-lz[j])**2, (abs(lz1[ipick]-lz[j])-box)**2 )
            rij=math.sqrt(dis2)
            if rij <=  rcut :
                e1 += 4*eps*((sig/rij)**12-(sig/rij)**6)
    return e1        

a=energy()
print ('Initialization: total energy', a)
 
@jit (nopython=True)
def gofr(lx,ly,lz) :  # to caulate g(r)
    lgofr=[]
    lgofr.append(0)
    dl=box*1.0/nbins/4
    for il in range (0+1,nbins+1) :
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

# file to store energy data
energy_file = open('energy_argon','w')

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
def move(dmax, frac) :
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
        if ly1[ipick] < 0:
            ly1[ipick] += box
        if lz1[ipick] < 0:
            lz1[ipick] += box
      
        if lx1[ipick] > box: 
            lx1[ipick] -= box
        if ly1[ipick] > box:            
            ly1[ipick] -= box
        if lz1[ipick] > box:
            lz1[ipick] -= box
    
    # -- calculate energy difference between new configuration
    #     and old configuration.
        e1=esingle(ipick, lx, ly, lz, lx, ly, lz)
        e2=esingle(ipick, lx1, ly1, lz1, lx, ly, lz)
        ediff=e2-e1
    
        if random.uniform(0,1) < math.exp(-beta*ediff) :
            lx[ipick]=lx1[ipick]
            ly[ipick]=ly1[ipick]
            lz[ipick]=lz1[ipick]
            naccpt += 1
        else:      
            lx1[ipick]=ax
            ly1[ipick]=ay
            lz1[ipick]=az

    raccp=naccpt*1.0/ndispl # accpetance rate
    frac=raccp
    if logprod == 0 :
        return frac
    else :   
        lx2=np.array(lx)
        ly2=np.array(ly)
        lz2=np.array(lz)
        ep = energy()
        energy_file.write(str(ep/(npart*1.0))+'\n')
        return frac, gofr(lx2,ly2,lz2)

# adjust maximum displacement to get ~50% acceptance rate
print ('equilibration cycles')

frac=0.5
lfrac5=[]

for i in range (nequil) :
    naccpt=0
    if math.fmod (i+1,5)==0:
        # update dmax after every 5 cycles
        dold=dmax
        dmax=dold*(frac/0.5)
        # limit the change in dmax
        if dmax/dold > 1.5:
            dmax=dold*1.5
        if dmax/dold < 0.5:
            dmax=dold*0.5
        if dmax > box/2.0:
            dmax=box/2.0
    
    frac5= move (dmax, frac) 
    lfrac5.append(frac5)
    frac=sum(lfrac5)/len(lfrac5)
    if math.fmod (i+1,500)==0 :        
        print ('cycle', i+1, 'frac. accp.',round(frac,4),'dmax',round(dmax,4) )
    # update dmax for every 5 cycles 

print ('production cycles')
logprod=1
l2gofr=[]

for i in range (lmax) :    
    logprod=0
    a1=move (dmax, frac)
    if math.fmod(i+1,nsamp)==0 :
        logprod=1
        a1=move (dmax, frac)
        b1=[]
        for j in range (len(a1[1])):
            b1.append(a1[1][j])
       
        l2gofr.append(b1)
  
l2gofr_avg=[]

#energy_file.close()

for i in range(len(l2gofr[0])):
    s1=0
    for j in range (len(l2gofr)):
        s1 += l2gofr[j][i]
        
    a1=s1/(len(l2gofr))
    l2gofr_avg.append(a1)

g1=open('gr.dat','w')
g1.write('# r/sigma '+'g (r)'+'\n')
dt=box/nbins/2
lr=[]

for i in range (len(l2gofr_avg)):        
    g1.write( str(i*dt)+' '+str(l2gofr_avg[i])+'\n')
    lr.append(i*dt)
    
g1.close()

#%%
fig1=plt.figure()
ax=plt.subplots()
x=lr
y=l2gofr_avg
plt.plot(x,y)
plt.xlabel('r/$\sigma$')
plt.ylabel('g (r)')
plt.show()
#plt.savefig ('gofr.pdf')

l2gofr_avg = np.array(l2gofr_avg)
lr = np.array(lr)
 
peaks, _ = find_peaks(l2gofr_avg, distance=15);
values_max_gr = l2gofr_avg[ peaks[:3] ];
values_max_r = lr[ peaks[:3] ];
 
plt.plot(lr, l2gofr_avg,  lw = 0.8, c = 'b');
plt.plot(values_max_r, values_max_gr , 
          ls ='none', marker ='x', c = 'r');
plt.plot(values_max_r, values_max_gr, 
          ls ='none', marker ='x', c = 'r');
plt.title(r'$g(r)$');
plt.xlabel(r'$r/\sigma$');
plt.ylabel(r'$g(r)$');
#plt.savefig('gr_rahman_MC'+'.eps', dpi = 300);
plt.show();

g2=open('sk.dat','w')
g2.write('# k*sigma '+'S (k)'+'\n')
lq=[]
lsk=[]
for i in range(1,144): # to calculate S (k)    
    dk=1.0/4.5
    q=i*dk
    j0=0
    
    for j in range (nbins+1) :
        gr=l2gofr_avg[j]
        if j+1 == 1 or j+1==nbins +1: #Simpson's rule
            f1=1
        if math.fmod (j+1, 2) ==0:
            f1=4
        if math.fmod (j+1, 2) ==1:
            f1=2
    
        r=j*dt
        gr=l2gofr_avg[j]  #Simpson's rule
        if q*r < 0.01 :
            #for small x use series expansion to avoid computing 0/0
            x2=(q*r)**2
            taylor=1.0-x2/6.0*(1.0 -x2/20.0*(1.0 -x2/42.0*(1.0 -x2/72.0))) 
            j0 += taylor*r**2*(gr-1)*f1
        else:
            j0 += r**2*(gr-1)*math.sin(q*r)/(q*r)*f1
    
    sk=1+(4*pi*rho*j0)*dt/3.0
    g2.write (str(q)+' '+str(sk)+'\n')
    lq.append (q)
    lsk.append (sk)

g2.close()

lsk = np.array(lsk); #y
lq = np.array(lq); #x

peaks_sk, _ = find_peaks(lsk, distance=25);
values_max_sk = lsk[ peaks_sk[:4] ];
values_max_k = lq[ peaks_sk[:4] ];

plt.plot(lq, lsk, lw = 0.8, c = 'b')
plt.plot(values_max_k, values_max_sk, 
         ls ='none', marker ='x', c = 'r')
plt.title(r'$S(k)$')
plt.xlabel(r'$k \sigma$')
plt.ylabel(r'$S(k)$')
#plt.savefig('Sk_rahman_FT_MC'+'.eps', dpi = 300);
plt.show()


print("--- %s seconds ---" % (time.time() - start_time))
