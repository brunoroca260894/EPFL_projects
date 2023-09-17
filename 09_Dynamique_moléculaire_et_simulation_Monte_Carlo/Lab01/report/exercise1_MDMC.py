#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:48:48 2021
Molecular dynamics and Monte Carlo methods
    This is the exercise session 1 
@author: bruno rc
"""

import random as r
import math as m
import matplotlib.pyplot as plt
import numpy as np

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

# Random Seed
r.seed(42)

def estimatePI(squareLength, circleRadius, n_trials=100):    
    inside_x, inside_y = np.array([]), np.array([])
    outside_x,outside_y=np.array([]), np.array([])
    drift_radius = circleRadius*0.50 # to change the center of the circle
    
    # Iterate for the number of darts.
    for i in range(0, n_trials):        
        x = squareLength*r.uniform(-1,1)
        y = squareLength*r.uniform(-1,1)
        if (x+drift_radius)**2 + (y+drift_radius)**2 < circleRadius**2:
            inside_x = np.append(inside_x, x)
            inside_y = np.append(inside_y, y)
        else:
            outside_x = np.append(outside_x, x)
            outside_y = np.append(outside_y, y)        
            
    p_success =  inside_x.size/(inside_x.size + outside_x.size)       
    pi = 4*(squareLength/circleRadius)**2*p_success # Estimate Pi
    
    fig, ax=plt.subplots(1, figsize=(5,5))

    ax.scatter(inside_x,inside_y, marker = '.', c = 'b', lw = 0.1)
    ax.scatter(outside_x,outside_y,marker = '.', c= 'gray', lw = 0.1)
    ax.grid(True, which = 'both', ls = '--')
    #plt.savefig('estimation4_2_'+ str(n_trials)+ '.png')
    plt.show()    
    return p_success, pi

#different number of samples
N  = 10000*np.arange(1, 4)
# case 1:
#   d = l 
l = 1
d = l/2
pi_estimation = np.zeros(shape=(N.size))
p_success = np.zeros(shape=(N.size))
i = 0

for n in N:
    p_success[i], pi_estimation[i]= estimatePI(l, d, int(n))
    #print(f"{pi_estimation[i]:.5f}")
    #print('probability of success: ', round(p_success[i],4))
    #print('---------------')
    i = i +1

estimator_var = np.var(pi_estimation)

#--error plot
plt.loglog(N, np.sqrt(estimator_var/N), marker = 'x', lw =1.5 , c = 'b',
           label = 'Monte Carlo estimation')
plt.xlabel('number of samples')
plt.loglog(N, N**(-1/2), marker = 'x', ls = '--', lw = 1, c = 'k',
           label ='Monte Carlo theoretical error')
plt.grid(True, which='both', ls = '--')
plt.legend(fontsize = 8)
plt.show()

pi_true = np.pi* np.ones(shape =(N.size))

plt.plot(pi_estimation,marker = 'x', lw =1.5 , c = 'b',
         label = 'estimated value of pi')        
plt.plot(pi_true, marker = 'x', ls = '--', lw = 1, c = 'k',
         label = 'True value of pi') 
plt.grid(True, which='both', ls = '--')
plt.legend(fontsize = 8)
plt.show()
