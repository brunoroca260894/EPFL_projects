#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:18:12 2021
    Final code standard MLMC
    Lorenz equaion in 3D
@author: brunorc
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits import mplot3d
import time
import math 
from numba import jit, njit, float32
import timeit
import itertools

plt.rc("legend", framealpha=None)
plt.rc("legend", edgecolor='black')
plt.rc("font", family="serif")

########################################
class WeakConvergenceFailure(Exception):
    pass

#Euler-Maruyama method for one level  
@njit
def SDE_one_level_EM(T, l, X0, h0): 
    hf = h0/(2**l)
    nf = int(T/hf) #number of points for each path   

    Xf = X0 #X0 must have d columns  
    for i in range(0, nf-1):
        Xf = Xf + f(Xf)*hf + np.sqrt(hf) * np.random.randn(3) 
    
    Pf = np.sqrt((Xf**2).sum()) 
   
    return Pf

#Euler-Maruyama method for two levels
@njit
def eulerMaruyama(T, l, h0, X0): # hf is the finest level
    hf = h0/(2**l)
    nf = int(T/hf) #number of points for each path   
    d = X0.shape[0] #number of dimensions of solution vector
    
    # Crude monte carlo method 
    if l == 0:        
        Xf = X0*np.ones(shape = (nf, d) )        
        for i in range(0, nf-1):
            Xf[i+1, :] = Xf[i, :] + f(Xf[i, :])*hf + np.sqrt(hf) * np.random.randn(3)         
            
        Pf = np.sqrt((Xf[-1, :]**2).sum()) 
        Pc = 0.
        
    # std MLMC method 
    else:        
        Xf = X0*np.ones(shape = (nf, d) )
        Xc = X0*np.ones(shape = (int(nf/2), d) )
        
        for i in range(0, nf-1):          
            if (i+1)%2 != 0: # this is the first one to be computed                
                dW_odd = np.sqrt(hf) * np.random.randn(3)                
                Xf[i+1, :] = Xf[i, :] + f(Xf[i, :])*hf + dW_odd
            else:
                dW_even = np.sqrt(hf) * np.random.randn(3)
                j = int(0.50*(i+1)) # index for Xc
                k = int(0.50*(i))# index for Xc, past
                
                Xf[i+1, :] = Xf[i, :] + f(Xf[i, :])*hf + dW_even             
                Xc[j , :] = Xc[k, :] + f(Xc[k, :])*2*hf + dW_even + dW_odd
                
        # t = np.linspace(0, T, Xf.shape[0])
        
        # plt.figure(1) 
        # plt.plot(t, Xf[:, 0], c = 'k', label =r'$X^{f}$', lw = 0.6)
        # plt.plot(t[::2], Xc[:, 0], c ='b', label =r'$X^{c}$', lw = 0.6)
        # plt.xlabel('time T')
        # plt.ylabel(r'$x_{1}$')
        # #plt.title('Trajectory standard MLMC')
        # plt.legend(loc='upper right', fontsize='medium')
        # plt.show()
        
        # plt.figure(2) 
        # plt.plot(t, Xf[:, 1], c = 'k',label =r'$X^{f}$', lw = 0.6)
        # plt.plot(t[::2], Xc[:, 1], c ='b', label =r'$X^{c}$', lw = 0.6)
        # plt.xlabel('time T')
        # plt.ylabel(r'$x_{2}$')
        # #plt.title('Trajectory standard MLMC')
        # plt.legend(loc='upper right', fontsize='medium')
        # plt.show()
        
        # plt.figure(3) 
        # plt.plot(t, Xf[:, 2], c = 'k', label =r'$X^{f}$', lw = 0.6)
        # plt.plot(t[::2], Xc[:, 2], c ='b', label =r'$X^{c}$', lw = 0.6)
        # plt.xlabel('time T')
        # plt.ylabel(r'$x_{3}$')
        # #plt.title('Trajectory standard MLMC')
        # plt.legend(loc='upper right',fontsize='medium')
        # plt.show()

        Pf = np.sqrt((Xf[-1, : ]**2).sum()) 
        Pc = np.sqrt((Xc[-1, : ]**2).sum()) 
    
    return Pf, Pc

########################################
# Lorenz system
@njit
def B(x):
    return 65*x/max([65, abs(x)])

@njit
def f(x): #Lorenz equation
    '''
    if np.abs(x[0]) > 65 and np.abs(x[1]) > 65: # |x1| > 65 and |x2| > 65 
        a = 650*np.sign(x[1]) - 10*x[0]
        b = 65*np.sign(x[0])*(28 - x[2]) - x[1]
        c = 65*np.sign(x[0])*x[1] - 8*x[2]/3        
    elif np.abs(x[0]) < 65 and np.abs(x[1]) > 65:
        a = 10*( np.sign(x[1]) - x[0] )
        b = (28 - x[2])*x[0] - x[1]
        c = x[0]*x[1] - 8*x[2]/3
    elif np.abs(x[0]) > 65 and np.abs(x[1]) < 65:
        a = 10*( x[1] - x[0] )
        b = (28 - x[2])*np.sign(x[0]) - x[1]
        c = np.sign(x[0])*x[1] - 8*x[2]/3
    else:
        a = 10*( (x[1]) - x[0] )
        b = (28 - x[2])*(x[0]) - x[1]
        c = (x[0])*x[1] - 8*x[2]/3

    '''
    a = 10*( B(x[1]) - x[0] )
    b = (28 - x[2])*B(x[0]) - x[1]
    c = B(x[0])*x[1] - 8*x[2]/3
    return np.array([a, b, c])       
      
########################################
####### Multilevel approach #######
def mlmc(Lmin, Lmax, N0, eps, alpha_0, beta_0, gamma_0, h0):
    # Check arguments
    if Lmin < 2:
        raise ValueError("Need Lmin >= 2")
    if Lmax < Lmin:
        raise ValueError("Need Lmax >= Lmin")
    if N0 <= 0 or eps <= 0:
        raise ValueError("Need N0 > 0, eps > 0")

    # Initialisation
    alpha = max(0, alpha_0)
    beta  = max(0, beta_0)
    gamma = max(0, gamma_0)

    L = Lmin

    Nl   = np.zeros(L+1)
    suml = np.zeros((2, L+1))
    costl = np.zeros(L+1)
    dNl  = N0*np.ones(L+1)
    
    while sum(dNl) > 0:
        # update sample sums
        for l in range(0, L+1):
            print('level: ', l)
            if dNl[l] > 0:                               
                Pf = np.zeros(shape = int(dNl[l])) 
                Pc = np.zeros(shape = int(dNl[l]))
                start =time.perf_counter()
                for i in range(int(dNl[l])):        
                    Pf[i], Pc[i] = eulerMaruyama(T, l, h0, X0)                
                end = time.perf_counter()
                
                cost = end - start
                sums = np.array([np.sum(Pf - Pc), np.sum((Pf - Pc)**2)])
                
                Nl[l]        = Nl[l] + dNl[l]
                suml[0, l]   = suml[0, l] + sums[0]
                suml[1, l]   = suml[1, l] + sums[1]
                costl[l]     = costl[l] + cost

        # compute absolute average, variance and cost
        ml = np.abs( suml[0, :]/Nl)
        Vl = np.maximum(0, suml[1, :]/Nl - ml**2)
        #Vl[0] = np.maximum(0, suml[1, 0]/Nl[0])
        Cl = costl/Nl 
        Cl[0] = Cl[1]*0.50;

        # fix to cope with possible zero values for ml and Vl
        # (can happen in some applications when there are few samples)
        for l in range(3, L+2):
            ml[l-1] = max(ml[l-1], 0.5*ml[l-2]/2**alpha)
            Vl[l-1] = max(Vl[l-1], 0.5*Vl[l-2]/2**beta)
            
        # use linear regression to estimate alpha, beta, gamma if not given
        if alpha_0 <= 0:
            A = np.ones((L, 2)); 
            A[:, 0] = range(1, L+1)
            x = np.linalg.lstsq(A, np.log2(ml[1:]),  rcond=None)[0]
            alpha = max(0.5, -x[0])

        if beta_0 <= 0:
            A = np.ones((L, 2)); 
            A[:, 0] = range(1, L+1)
            x = np.linalg.lstsq(A, np.log2(Vl[1:]),  rcond=None)[0]
            beta = max(0.5, -x[0])

        if gamma_0 <= 0:
            A = np.ones((L, 2)); 
            A[:, 0] = range(1, L+1)
            x = np.linalg.lstsq(A, np.log2(Cl[1:]),  rcond=None)[0]
            gamma = max(0.5, x[0])
            
        # set optimal number of additional samples
        Ns = np.ceil(2*np.sqrt(Vl/Cl) * sum(np.sqrt(Vl*Cl)) / (eps**2) )
        dNl = np.maximum(0, Ns-Nl)           

        # if (almost) converged, estimate remaining error and decide
        # whether a new level is required
        
        print('bias error - ',eps**2/2, ml[-1]**2)
        print('stat error - ', eps**2/2, sum(Vl/Nl) )

        if sum(Vl/Nl) < eps**2/2: #check variance convergence
            rang = list(range(min(3, L)))
            rem = ( np.amax(ml[[L-x for x in rang]] / 2.0**(np.array(rang)*alpha))
                    / (2.0**alpha - 1.0) )
            
            if rem > eps/math.sqrt(2):
                if L == Lmax:
                    raise WeakConvergenceFailure("Failed to achieve weak convergence")
                else:
                    print('extra level added')
                    L = L + 1
                    Vl = np.append(Vl, Vl[-1] / 2.0**beta)
                    Nl = np.append(Nl, 0.0)
                    suml = np.column_stack([suml, [0, 0]])
                    Cl = np.append(Cl, Cl[-1]*2**gamma)
                    costl = np.append(costl, 0.0)

                    Ns = np.ceil( 2*np.sqrt(Vl/Cl) * sum(np.sqrt(Vl*Cl)) / (eps**2) )
                    dNl = np.maximum(0, Ns-Nl)
        print('Vl: ', Vl)
        print('cost: ', Cl)
        print('extra samples required:')
        print(dNl)
        print('additional run')
        print('---------')
    # evaluate the multilevel estimator
    P = sum(suml[0,:]/Nl)
    print('******value: ',  P)
    
    return (P, Nl, Cl, Vl, ml, alpha, beta, gamma)

########################################
## input data
T= 5
X0 = np.random.randn(3)
Lmin = 2
Lmax = 20
N0 = 500 #initial number of samples
eps = 0.4 # epsilon error
h0 = 2**(-11)
alpha_0 = 0
beta_0 = 0
gamma_0 = 0

########################################
print('Standard Multilevel Monte Carlo')
epsilon =np.array( [0.2, 0.1, 0.05, 0.025])

a = np.zeros(shape = len(epsilon) )
b = np.zeros(shape = len(epsilon) )
g = np.zeros(shape = len(epsilon) )

var = np.zeros(shape = len(epsilon) )
mu = np.zeros(shape = len(epsilon) )
cost = np.zeros(shape = len(epsilon) )

marker = itertools.cycle(('d', 's', 'v', 'o', '*', '+')) 

i = 0
for eps in epsilon:    
    print('epsilon ', eps)
    t1 = time.time()
    P, Nl, Cl, Vl, ml, alpha, beta, gamma = mlmc(Lmin, Lmax, N0, eps, alpha_0, beta_0, gamma_0, h0)
    t2 = time.time()
    
    f= open("standardMLMCT_5.txt","a");
    f.write('*******Data for epsilon******' + str(eps) +'\n')
    f.write('time T: ' + str(T) + '\n')
    f.write('P: ' + str(P) + '\n');
    f.write('Nl: ' + str(Nl)+ '\n');
    f.write('Cl =' + str(Cl) + '\n');
    f.write('Vl =' + str(Vl) + '\n');
    f.write('ml =' + str(ml) + '\n');
    f.write('alpha =' + str(alpha) + '\n');
    f.write('beta =' + str(beta) + '\n');
    f.write('gamma =' + str(gamma) + '\n');
    f.write('total time: ' + str(t2-t1) + '\n');
    f.write('----------------------'+ '\n');
    f.write('\n');
    f.close()
    
    l = np.arange(len(Vl), dtype = np.int)    

    mkr = next(marker)
    
    # variance per eps
    # plt.figure(1)
    # plt.plot(l, np.log2(Vl), label=r'$\epsilon = $' + str(eps), ls = '--', 
    #           marker = mkr, clip_on=False, c = 'k', lw = 0.80 )
    # plt.xlabel('level $\ell$')
    # plt.ylabel(r'$\mathrm{log}_2 $' + ' $\mathrm{variance}$')
    # #plt.title('T = ' +str(T) +', Variance')
    # plt.legend(fontsize='medium')
    # plt.show()

    # number of samples 
    # plt.figure(2)
    # plt.plot(l, np.log2(Nl), label=r'$\epsilon = $' + str(eps), ls = '--', 
    #           marker = mkr, clip_on=False, c = 'k', lw =0.8 )
    # plt.xlabel('level $\ell$')
    # plt.ylabel(r'$N_{\ell}$')
    # #plt.title('T = ' +str(T) + ' Number of samples Nl - standard MLMC')
    # plt.legend(fontsize='medium')
    # plt.show()
    
    a[i] = alpha
    b[i] = beta
    g[i] = gamma
    
    var[i] = sum(Vl/Nl)
    mu[i] = P    
    cost[i] = sum(Nl*Cl)
    
    i = i + 1
    print('----------**** New epsilon ****----------------')

# variance for smallest eps
plt.figure(3)
plt.plot(l, np.log2(Vl), label=r'$\epsilon = $' + str(epsilon[-1]), ls = '--', marker = '*', clip_on=False,
              c = 'k' );
plt.xlabel('level $\ell$');
plt.ylabel(r'$\mathrm{log}_2 $'+'$\mathrm{variance}$');
#plt.title('T = ' + str(T) + ', Variance - $\epsilon$= ' + str(epsilon[-1]));
#plt.legend(loc='upper right', fontsize='medium');
plt.show()

# cost vs accuracy epsilon
plt.figure(4)
plt.plot(epsilon, np.log2(cost), label='computed cost', ls = '--', marker = '*', clip_on=False,
              c = 'k' )
plt.plot(epsilon, np.log2(2**8*epsilon**-2), label='theory cost', ls = '--', marker = 'x', clip_on=False,
              c = 'b' )
plt.xlabel('accuracy $\epsilon$')
plt.ylabel('$\log_{2}$' + ' cost(CPU time)')
plt.xticks(epsilon)
plt.legend(loc='upper right', fontsize='medium')      
plt.show()

#######################################
# # create fig 1
# l = 8, N for all T is 200
# tm = np.arange(1, 16)
# tm = np.append([0.25, 0.5, 0.75], tm)
# L = np.arange(0, 4)
# ml = np.zeros(shape = (len(tm), N0))

# N0 = 200
# X0 = np.random.randn(3)
# eps = 0.4 # epsilon error
# h0 = 2**(-11)

# for l in L:    
#     print('level: ', l)
#     i = 0
#     for T in tm:
#         Vl = 0.
#         Pf = 0.
#         Pc = 0.
#         print('time: ', T)
#         for j in range(N0):
#             Pf, Pc = eulerMaruyama(T, l, h0, X0)
#             ml[i, j] = Pf-Pc
#         i = i+1        

#     if l > 0:
#         plt.plot(tm, np.log(np.var(ml, axis =1)), label= r'$\ell = $' + str(l));
#     print('----------')    

# #plt.grid(True, which ='both');
# plt.xlabel('time $T$');
# plt.ylabel('$\log_{2}$ variance');
# #plt.title( 'computed variance for different ' + '$\ell$');
# plt.legend(loc='lower right', fontsize='medium');
# plt.show()

# # file writing
# f= open("standardMLMC.txt","a");
# f.write('Data for figure 1' + '\n')
# f.write('L =' + str(L) + '\n');
# f.write('time: ' + str(tm) + '\n');
# f.write('ml: ' + '\n');
# f.write(str(ml));
# f.write('\n');
# f.write('Vl: ' + '\n');
# f.write(str(  np.var(ml, axis =1) ) );
# f.write('\n');
# f.close()

#######################################
## PLOT Vl vs hl, Nl = 200, l = 0,.., 4
# X0 = np.random.randn(3)
# eps = 0.4 # epsilon error
# h0 = 2**(-11)
# N0 = 200

# L = np.arange(0, 4)
# tm =np.array([5, 10,20])
# vl = np.zeros(shape = (len(tm),len(L)))
# ml = np.zeros(shape = (N0))
# hl = h0/2**L

# j = 0
# for T in tm:      
#     print('time: ', T)
#     for l in L:
#         print('level ', l)
#         Pf = 0.
#         Pc = 0.
        
#         for i in range(N0):
#             Pf, Pc = eulerMaruyama(T, l, h0, X0)
#             ml[i] = Pf-Pc
            
#         vl[j, l] = np.var(ml)
#     j = j +1    
#     print('----------')    
    
# plt.plot(hl, np.log2(vl[0, :]), label= r'$T = $' + str(tm[0]), marker='x' );
# plt.plot(hl, np.log2(vl[1, :]), label= r'$T = $' + str(tm[1]), marker='x' );
# plt.plot(hl, np.log2(vl[2, :]), label= r'$T = $' + str(tm[2]), marker='x' );
# plt.grid(True, which ='both');
# plt.xlabel('$\ell$');
# plt.ylabel('$\log_{2}$ variance');
# plt.xticks(l);
# #plt.title( 'computed variance for different ' + '$T$');
# plt.legend(loc ='lower right',fontsize='medium');
# plt.show()


#######################################
# # create Fig 2 in Giles'
# l = 8, N for all T is 500
# tm = np.arange(1, 11)
# tm = np.append(0.5, tm)

# l = 3
# N0= 200
# ml = np.zeros(shape = (len(tm), N0))
# i = 0

# for T in tm:
#     Vl = 0.
#     Pf = 0.
#     Pc = 0.
#     print('time: ', T)
#     for j in range(N0):
#         Pf, Pc = eulerMaruyama(T, l, h0, X0)
#         ml[i, j] = Pf-Pc
#     i = i+1
#     print('----------')
        
# #theoretical rate for variance: 
# kappa = 1.36
# hl = h0/(2**l)
# var_theory = (hl**2)*np.exp(kappa*tm)

# # linear fit
# x = tm
# y = np.var(ml, axis =1 )
# z = np.polyfit(x, np.log(y), 1)
# p = np.poly1d(z)

# plt.plot(tm, np.log(np.var(ml, axis =1 )), c = 'blue', label='computed variance');
# plt.plot(tm,np.log(var_theory),  ls = '--', c = 'k', label = 'theory rate');
# plt.plot(tm, p(tm), c = 'g', ls= '--', label='linear fit');
# plt.grid(True, which ='both');
# plt.xlabel('time $T$');
# plt.ylabel('log variance');
# plt.title( 'variance for $\ell = 3$, $T = 1,.., 10$');
# plt.legend(loc='upper left', fontsize='medium');
# plt.show()

# # file writing
# f= open("standardMLMC.txt","a");
# f.write('Data for figure 2 a' + '\n')
# f.write('N0: ' + str(N0));
# f.write('\n');
# f.write('L =' + str(l) + '\n');
# f.write('time: ' + str(tm) + '\n');
# f.write('ml: ' + '\n');
# f.write(str(ml));
# f.write('\n');
# f.write('Vl: ' + '\n');
# f.write(str(  np.var(ml, axis =1) ) );
# f.write('\n');
# f.close()
