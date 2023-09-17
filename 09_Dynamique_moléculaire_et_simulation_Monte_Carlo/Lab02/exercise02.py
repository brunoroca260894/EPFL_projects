#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 12:05:14 2021

@author: bruno
"""
import numpy as np
import scipy.constants as sp
import matplotlib.pyplot as plt
plt.style.use(['seaborn-ticks', 'seaborn-talk'])

n_energy_levels= 10
reducedTemperatures=[0.5, 1, 2, 3]

def energy(i):
    w =1 # omega equals 1    
    return (i + 0.50)*w

def energy_rotor(i):
    return i*(i+1)/2
    
def Z(T, i):    
    beta = 1/(T)
    return np.exp(-beta*energy(i))   
    
def calculateStateOccupancy(T, i):
    # No degeneracy
    beta = 1/T
    z = np.exp(-beta*energy(i))
    return np.exp(-beta*energy(i)), z

def calculateStateOccupancy_s1(T, i):
    # Degeneracy s+1
    beta = 1/T
    if i == 0:
        e = np.exp(-beta*energy(i))
        z = np.exp(-beta*energy(i))
    elif i == 1:
        e = np.exp(-beta*energy(i))
        z = np.exp(-beta*energy(i))
    else:
        e = np.exp(-beta*energy(i-1))
        z = np.exp(-beta*energy(i-1))
    return e, z

def calculateStateOccupancy_s2(T, i):
    # Degeneracy s+2
    beta = 1/T
    if i == 0:
        e = np.exp(-beta*energy(i))
        z = np.exp(-beta*energy(i))
    elif i == 1:
        e = np.exp(-beta*energy(i))
        z = np.exp(-beta*energy(i))
    elif i == 2:
        e = np.exp(-beta*energy(i))
        z = np.exp(-beta*energy(i))
    else:
        e = np.exp(-beta*energy(i-2))
        z = np.exp(-beta*energy(i-2))
    return e, z

functions ={
    "no_degeneracy": calculateStateOccupancy,
    "s+1": calculateStateOccupancy_s1,
    "s+2": calculateStateOccupancy_s2
}

for f in functions.keys():
    fig, ax =plt.subplots(2)
    ax[0].set_title(f)
    ax[0].set_ylabel("Occupancy")

    ax[1].set_xlabel("Energy level")
    ax[1].set_ylabel("CDF")
    
    calculateOccupancy=functions[f]

    print('function: ', f) 

    for T in reducedTemperatures:
        distribution = np.array([]) # For each state there is one entry
        partition_function= 0.0

        for i in range(n_energy_levels):
            stateOccupancy, z = calculateOccupancy(T, i) 
            distribution = np.append(distribution, stateOccupancy)# MODIFY HERE
            partition_function += z   # MODIFY HERE
        
        print('--------------------')
        ax[0].plot(distribution/partition_function, label=T, marker = 'o')           
        ax[1].plot(np.cumsum(distribution/partition_function), label=T, marker = 'o')
    
    ax[0].ticklabel_format(useOffset=False)
    ax[0].legend()
    ax[0].grid(which='both')
    
    ax[1].ticklabel_format(useOffset=False)
    ax[1].legend()
    ax[1].grid(which='both')
    
plt.show()