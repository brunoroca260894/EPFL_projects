#!/usr/bin/python3

# The code performs a blocking analysis on the input data
# Output: sigma(I) and tau as a function of the block transformation 
#         step.
# Author: Sai Lyu, EPFL, Mar 2021

import numpy as np
import matplotlib.pyplot as plt
import math
# plt.rc("legend", framealpha=None)
# plt.rc("legend", edgecolor='black')
# plt.rc("font", family="serif")

## paramters
name_common = 'energy_'
#files_number= np.array([1, 5, 10, 15, 20, 25])
#ntrans_vector = np.array([17, 12, 12, 14, 13, 12])

files_number= np.array([1, 5, 10, 15, 20, 25])
ntrans_vector = np.array([17, 14, 13, 12, 12, 11])

ntrans = 17
fn = 'energy_1' 

## paramters
f1=open(fn,'r')
ff1=f1.readlines()
f1.close()

f1=open(fn,'r')
nline=0
for line in f1:
   nline+=1
f1.close()

g1=open(str(fn)+'_BLOCKING.dat','w')
g1.write('# step, Sigma_I(M), SD{Sigma_I}\n')

A0list=[]
for i in range (nline):
  aa=ff1[i].split()[0]
  A0list.append(float(aa))

Alist=[]
Blist=[]
def blocking (Alist):
  Blist=[]
  for i in range (  int( len(Alist)/2)) :
    b=0.5*(Alist[2*i]+Alist[2*i+1])
    Blist.append(b)
  return Blist  

def var2 (Alist) :
  var2=0
  suma=sum(Alist)
  for i in range (len(Alist))  :
     var2 += (Alist[i]-suma/len(Alist))**2/len(Alist)
  return var2  

g1.write(str(0)+' '+str(math.sqrt(var2(A0list)/len(A0list))) \
          +' '+str(1.0/math.sqrt(2*len(A0list))*math.sqrt(var2(A0list)/len(A0list))) \
          +' '+'\n')
Alist=A0list
for iblock in range (ntrans) : 
  Blist=blocking(Alist)
  taum=var2(Blist)*2**(iblock+1-1)/var2(A0list)
  sdtaum=math.sqrt(2.0/len(Blist))*taum
  sigmaI=math.sqrt(2*taum*var2(A0list)/len(A0list))
  sdsigmaI=math.sqrt(0.5/len(Blist))*sigmaI
  Alist=Blist
  g1.write( str(iblock+1)+' '+str(sigmaI)+' '+str(sdsigmaI)+' '+'\n')  
g1.close() 

#############################################################
# plot results
#%%
files_number= np.array([1, 5, 10, 20, 25])
ntrans_vector = np.array([17,14, 13, 12, 11])

for i, ntrans in zip(files_number, ntrans_vector):
    fn = 'energy_' +str(i)    
    data_energy = np.loadtxt(str(fn)+'_BLOCKING.dat',unpack=True).transpose()
    M = data_energy[:, 0]
    sigma = data_energy[:, 1]
    std_sigma = data_energy[:, 2]
    
    plt.errorbar(M,sigma,yerr=std_sigma, marker = '^', ls = '-', lw = 0.8, 
              label ='ntrans '+ str(i))
    plt.figure(figsize=(20,10))
    plt.xlabel('M')
    plt.ylabel(r'$\sigma(M)$')
    plt.legend()
    #plt.grid(True, which='both')
    plt.savefig('blockEnergy' + '.png', dpi = 300)
plt.title('energy, blocking analysis')
plt.show()