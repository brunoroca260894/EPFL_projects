#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 2021

@author: bruno
"""

import numpy as np
import scipy.constants as sp
import matplotlib.pyplot as plt
plt.style.use(['seaborn-ticks', 'seaborn-talk'])

n_energy_levels= 10
reducedTemperatures=[0.5, 1, 2, 3]

def energy_rotor(i):
    return i*(i+1)/2

def calculateStateOccupancy_rotor(T, N):
    # Linear Rotor
    beta = 1/T
    z = 2/beta      
    e = np.zeros(shape = N)
    counter = 0
    
    for i in range(N):    
        if i < 3:
            e[i] = np.exp(-beta*energy_rotor(i))
            print('level: ', i)
        else:           
            if i%2 ==0:
                e[i] = np.exp(-beta*energy_rotor(i))
                print('level: ', i)
            else:
                e[i] = np.exp(-beta*energy_rotor(i-2-counter))
                print('level: '+ str(i) + ' degeneracy: '+ str (i-2-counter)) 
                counter += 1    
    return e, z

fig, ax =plt.subplots(2)
ax[0].set_title('linear rotor')
ax[0].set_ylabel("Occupancy")

ax[1].set_xlabel("Energy level")
ax[1].set_ylabel("CDF")

for T in reducedTemperatures:        
    e, z = calculateStateOccupancy_rotor(T, n_energy_levels)
    print('Z: ', e.sum())
    print('z: ', z)
    print('-----------------')
    
    ax[0].plot(e/e.sum(), label=T, marker = 'o')           
    ax[1].plot(np.cumsum(e/e.sum()), label=T, marker = 'o')
    
    #ax[0].ticklabel_format(useOffset=False)
    ax[0].legend()
    ax[0].grid(True, which='both')
    
    #ax[1].ticklabel_format(useOffset=False)
    ax[1].legend()
    ax[1].grid(True, which='both')
    
# beta = 1/0.50
# t =np.array([np.exp(-beta*energy_rotor(i)) for i in range(N)]) 

