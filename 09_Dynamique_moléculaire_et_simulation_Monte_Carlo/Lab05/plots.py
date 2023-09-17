#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:36:53 2021

@author: bruno
"""


import matplotlib.pyplot as plt
import numpy as np

DataIn = np.loadtxt('co.dat', usecols=[0,1])

DataInOO = np.loadtxt('oo.dat', usecols=[0,1])

radial = np.loadtxt('radialDistribution.dat', usecols=[0,1])

plt.plot(DataIn[:, 0],DataIn[:, 1], ls = '-', c= 'k', lw =0.8)
plt.ylabel('bounds')
plt.xlabel('time step')
plt.title('CO bond length')
plt.show()


# plt.plot(DataInOO[:, 0],DataInOO[:, 1], ls = '-', c= 'k', lw =0.8)
# plt.ylabel('bounds')
# plt.xlabel('time step')
# plt.title('OO bond length')
# plt.show()

# plt.plot(radial[:, 0],radial[:, 1], ls = '-', c= 'k', lw =0.8)
# plt.ylabel('g(r)')
# plt.xlabel('r/sigma')
# plt.title('radial distribution function length')
# plt.show()