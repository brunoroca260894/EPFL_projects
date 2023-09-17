#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:18:12 2021
    Final code MLMC with change of measure
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
    d = X0.shape[0] #number of columns (vector of three components)

    Xf = X0*np.ones(shape = (nf, d) )        
    for i in range(0, nf-1):
        Xf[i+1, :] = Xf[i, :] + f(Xf[i, :])*hf + math.sqrt(hf)*np.random.randn(3)             
    Pf = math.sqrt((Xf[-1, :]**2).sum())  
    return Pf

########################################
# Lorenz system
@njit
def B(x):
    return 65*x/max([65, abs(x)])

@njit
def f(x):
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
## Monte Carlo method with change of maesure
@njit
def solSDE(T, l, X0, h0, S):
    h = h0/(2**l) # h on fine level
    nf = int(T/h) #number of points per sample path    
    
    d = X0.shape[0] #number of columns (vector of three components)
    Xf = X0*np.ones(shape = (nf, d)) #X0 must have d columns
    Xc = X0*np.ones(shape = (nf, d)) #X0 must have d columns  
    Rf = 1.
    Rc = 1.
    
    # uncomment to plot Radon-Nikodym derivatives; 
    # if plot Radon derivatives, comment @njit at the top of this function
    # Rc_plot = np.zeros(shape = int(nf/2))
    # Rf_plot = np.zeros(shape = nf)
    
    for i in range(0, nf-1):      
        if (i+1)%2 != 0: # ODD CASE. This is the first to be computed
            dW_odd = np.sqrt(h) * np.random.randn(3)
                    
            # # COARSE LEVEL           
            sy = S*(Xf[i, :] - Xc[i, :]) # S(Yf - Yc)
            fy= f(Xc[i, : ]) # f(Yc)            
            Xc[i+1, :] = Xc[i, :] + (sy + fy)*h + dW_odd                                              
            
            # FINE LEVEL           
            Xf[i+1, :] = Xf[i, :] + (-sy + f(Xf[i, : ]) )*h + dW_odd 
            
            Rf = Rf*RadonNikodym(dW_odd, -sy , h)                            
            
            # Rf_plot[i] = Rf
            
        else: # EVEN CASE            
            dW_even = np.sqrt(h) * np.random.randn(3)
            
            # COARSE LEVEL                   
            Xc[i+1, :] = Xc[i, :] + (sy + fy)*h + dW_even
            
            # FINE LEVEL        
            syf = S*(Xc[i, :] - Xf[i, :])
            fyf = f(Xf[i, : ]) 
            
            Xf[i+1, :] = Xf[i, :] + h*(syf + fyf ) + dW_even   
                                                       
            Rf = Rf*RadonNikodym(dW_even, syf , h)        
            Rc = Rc*RadonNikodym(dW_even + dW_odd, sy , 2*h)  

            # Rf_plot[i] = Rf
            # Rc_plot[int((i-1)/2)] = Rc
    
    # t = np.linspace(0, T, Xf.shape[0])
    # plt.figure(1)        
    # plt.plot(t[:-1],Rf_plot[:-1], label = r'$R^{f}$', lw = 0.6, c ='k')
    # plt.plot(t[:-2:2],Rc_plot[:-1], label = r'$R^{c}$', lw = 0.6, c ='b')
    # plt.xlabel('time T')
    # plt.legend(fontsize='medium')
    # plt.show()
    # print('Rf mean: ', Rf_plot.mean())
    # print('Rc mean: ', Rc_plot.mean())
    
    # plt.figure(2)    
    # plt.plot(t, Xf[:, 0], c = 'k', label = r'$X^{f}$', lw = 0.6)
    # plt.plot(t, Xc[:, 0], c ='b', label = r'$X^{c}$', lw = 0.6)
    # plt.xlabel('time T')
    # plt.ylabel(r'$x_{1}$')
    # plt.title('Trajectory change of measure MLMC')
    # plt.legend(fontsize='medium')
    # plt.show()
    
    # plt.figure(3)
    # plt.plot(t, Xf[:, 1], c = 'k', label = r'$X^{f}$', lw = 0.6)
    # plt.plot(t, Xc[:, 1], c ='b', label = r'$X^{c}$', lw = 0.6)
    # plt.xlabel('time T')
    # plt.ylabel(r'$x_{2}$')
    # plt.title('Trajectory change of measure MLMC')
    # plt.legend(fontsize='medium')
    # plt.show()
    
    # plt.figure(4)
    # plt.plot(t, Xf[:, 2], c = 'k', label = r'$X^{f}$', lw = 0.6)
    # plt.plot(t, Xc[:, 2], c ='b', label = r'$X^{c}$', lw = 0.6)
    # plt.xlabel('time T')
    # plt.ylabel(r'$x_{3}$')
    # plt.title('Trajectory change of measure MLMC')
    # plt.legend(fontsize='medium')
    # plt.show()        
                
    Pc = Rc*math.sqrt((Xc[-1, :]**2).sum())        
    Pf = Rf*math.sqrt((Xf[-1, :]**2).sum()) 
    
    return Pf, Pc

@njit
def RadonNikodym(dW,S,h):
    return np.exp(-(dW*S).sum() - 0.5*h*(S*S).sum() )

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
                # Standard Monte Carlo
                if l == 0 :
                    Pf = np.zeros(shape = int(dNl[l])) 
                    Pc = 0 
                    start =time.time()
                    for i in range(int(dNl[l])):
                        Pf[i] = SDE_one_level_EM(T, l, X0, h0)
                    end = time.time()
                    
                # multilevel Monte Carlo with change of measure
                else:
                    Pf = np.zeros(shape = int(dNl[l]))         
                    Pc = np.zeros(shape = int(dNl[l]))
                    start =time.time()
                    for i in range( int(dNl[l]) ):                    
                        Pf[i], Pc[i] = solSDE(T, l, X0, h0, S)
                    end = time.time()        
                
                cost = end - start
                
                sums = np.array([np.sum(Pf - Pc), np.sum((Pf - Pc)**2)])            
                Nl[l]        = Nl[l] + dNl[l]
                suml[0, l]   = suml[0, l] + sums[0]
                suml[1, l]   = suml[1, l] + sums[1]
                costl[l]     = costl[l] + cost

        # compute absolute average, variance and cost
        ml = np.abs( suml[0, :]/Nl)
        Vl = np.maximum(0, suml[1, :]/Nl - ml**2)
        Vl[0] = np.maximum(0, suml[1, 0]/Nl[0])
        Cl = costl/Nl # cost per sample    
        """ cost on level 0 = 0.50(cost level 1)"""  
        Cl[0] = Cl[1]*0.50;

        # use linear regression to estimate alpha, beta, gamma if not given
        # alpha, beta and gamma are computed with those those value where the MLMC
        # approach is used
        
        if alpha_0 <= 0:
            A = np.ones((L, 2)); 
            A[:, 0] = range(1, L+1)
            x = np.linalg.lstsq(A, np.log2(ml[1:]),  rcond=None)[0]
            alpha = max(0.5, -x[0])
            #print('alpha: ',alpha)

        if beta_0 <= 0:
            A = np.ones((L, 2)); 
            A[:, 0] = range(1, L+1)
            x = np.linalg.lstsq(A, np.log2(Vl[1:]),  rcond=None)[0]        
            beta = max(0.5, -x[0])
            #print('beta: ', beta)
            
        if gamma_0 <= 0:
            A = np.ones((L, 2)); 
            A[:, 0] = range(1, L+1)
            x = np.linalg.lstsq(A, np.log2(Cl[1:]),  rcond=None)[0]
            gamma = max(0.5, x[0])            
            #print('gamma: ', gamma)
        
        # set optimal number of additional samples
        Ns = np.ceil( 2*np.sqrt(Vl/Cl) * sum(np.sqrt(Vl*Cl)) / (eps**2) )
        dNl = np.maximum(0, Ns-Nl)
        
        #print('bias error - ',eps**2/2, ml[-1]**2)                        
        #print('stat error - ', eps**2/2, sum(Vl/Nl) )

        if sum(Vl/Nl) < eps**2/2: #check variance convergence
            rang = list(range(min(3, L)))
            rem = ( np.amax(ml[[L-x for x in rang]] / 2.0**(np.array(rang)*alpha))
                    / (2.0**alpha - 1.0) )
            print('rem - ', rem)
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
        print('**********************')
        
    #evaluate the multilevel estimator
    P = sum(suml[0,:]/Nl)    
    
    return (P, Nl, Cl, Vl, ml, alpha, beta, gamma)

########################################
## input data
S = 5
T = 12
X0 = np.random.randn(3) # initial position
Lmin = 2
Lmax = 12
N0 = 500 #initial number of samples
eps = 0.8 # epsilon error
h0 = 2**(-11)
alpha_0 = 0
beta_0 = 0
gamma_0 = 0

########################################
# print('Multilevel Monte Carlo chnage of measure')
epsilon =np.array( [0.6, 0.5, 0.4, 0.3, 0.2])

a = np.zeros(shape = len(epsilon) )
b = np.zeros(shape = len(epsilon) )
g = np.zeros(shape = len(epsilon) )

var = np.zeros(shape = len(epsilon) )
mu = np.zeros(shape = len(epsilon) )
cost = np.zeros(shape = len(epsilon) )

marker = itertools.cycle(('d', 's', 'v', 'o', '*', '+')) 

i = 0
for eps in epsilon:    
    print('epsilon: ', eps)
    t1 = time.time()
    P, Nl, Cl, Vl, ml, alpha, beta, gamma = mlmc(Lmin, Lmax, N0, eps, alpha_0, beta_0, gamma_0, h0)
    t2 = time.time()
    
    #file writing
    f= open("changeOfMeasureMLMC.txt","a");
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

    a[i] = alpha
    b[i] = beta
    g[i] = gamma
    
    var[i] = sum(Vl/Nl) #total variance at time T
    mu[i] = P    # expected at time T
    cost[i] = sum(Nl*Cl) # total cost
    
    i = i + 1
    print('-----------******-----------------')
    
# variance for smallest eps
l = np.arange(len(Vl), dtype = np.int)      
plt.figure(1)  
plt.plot(l, np.log2(Vl), label=r'$\epsilon = $' + str(epsilon[-1]), ls = '--', marker = '+', clip_on=False,
              c = 'k' )
plt.xlabel('level $\ell$')
plt.ylabel('$\log_{2}$ variance')
plt.legend(loc='upper right', fontsize='medium')
plt.xticks(l)
plt.grid(True, which ='both');
plt.show()

# total cost vs accuracy epsilon
plt.figure(2)
plt.plot(epsilon, np.log2(cost), label='computed cost', ls = '--', marker = '*', clip_on=False,
              c = 'k' );
plt.plot(epsilon, np.log2(2**8*epsilon**-2), label='theory cost, $O(\epsilon^{-2})$', ls = '--', marker = 'x', clip_on=False,
              c = 'b' );
plt.xlabel('accuracy $\epsilon$');
plt.ylabel('$\log_{2}$' + ' cost(CPU time)');
plt.xticks(epsilon);
plt.legend(loc='upper right', fontsize='medium');
plt.show();

#######################################
# create fig 3
# tm = np.arange(1, 11)
# tm = np.append([0.25, 0.5, 0.75], tm)
# L = np.arange(0, 4)
# ml = np.zeros(shape = N0 ) 
# vl = np.zeros(shape = (len(L), len(tm)) )

# X0 = np.random.randn(3)
# eps = 0.4 # epsilon error
# h0 = 2**(-11)
# N0 = 200

# for l in L:
#     print('level: ', l)
#     i = 0
#     for T in tm:
#         Vl = 0.
#         Pf = 0.
#         Pc = 0.
#         print('time: ', T)
#         if l == 0:
#             Pc = 0 
#             for j in range(N0):
#                 Pf = SDE_one_level_EM(T, l, X0, h0)
#                 ml[j] = Pf - Pc
#             vl[l, i ] = sum(ml**2)/N0
        
#         else:
#             for j in range(N0):
#                 Pf, Pc = solSDE(T, l, X0, h0, S)
#                 ml[j] = Pf - Pc               
#             vl[l, i ] = np.var(ml)
#             #vl[l, i ] = sum(ml**2)/N0
#         i = i+1
#     print('vl')
#     print(vl[l, : ])
#     print('----------')
    
#     plt.plot(tm, np.log2(vl[l, : ]), label= r'$\ell = $' + str(l) );
       
# plt.grid(True, which ='both');
# plt.xlabel('time $T$');
# plt.ylabel('$\log_{2}$ variance');
# plt.xticks(tm)
# plt.title( 'variance for different ' + '$\ell$');
# plt.legend(loc='lower right', fontsize='medium');
# plt.show()

#######################################
# # create Fig 2 in Giles'
# # l = 8, N for all T is 500
# tm = np.arange(0, 21)
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
#         Pf, Pc = solSDE(T, l, X0, h0, S)
#         ml[i, j] = Pf-Pc
#     i = i+1
#     print('----------')
        
# #theoretical rate for variance: 
# hl = h0/(2**l)
# var_theory = 200000000*(hl**2)*tm

# ml[0, :] = 0

# # linear fit
# x = tm
# y = np.var(ml, axis =1 )
# z = np.polyfit(x, y, 1)
# p = np.poly1d(z)

# plt.plot(tm, np.var(ml, axis =1 ), c = 'blue', label='computed variance');
# plt.plot(tm, p(tm), c = 'g', ls= '--', label='linear fit');
# plt.plot(tm,var_theory,  ls = '--', c = 'k', label = 'theory rate');
# plt.grid(True, which ='both');
# plt.xlabel('time $T$');
# plt.ylabel('variance');
# plt.title( 'variance for $\ell = 3$, $T = 1,.., 20$');
# plt.legend(loc='upper left', fontsize='medium');
# plt.show()

#######################################
## PLOT Vl vs hl, Nl = 200, l = 0,.., 4
# X0 = np.random.randn(3)
# h0 = 2**(-11)
# N0 = 100

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
        
#         if l == 0:
#             for i in range(N0):                 
#                 Pf = SDE_one_level_EM(T, l, X0, h0)
#                 Pc = 0
#                 ml[i] = Pf-Pc
#             vl[j, l] = sum(ml**2)/N0
#         else:
#             for i in range(N0):            
#                 Pf, Pc = solSDE(T, l, X0, h0, S)
#                 ml[i] = Pf-Pc                
#             vl[j, l] = np.var(ml)
#             #vl[j, l] = sum(ml**2)/N0
#     j = j +1    
#     print('----------')    

# l = np.array([1,2,3])
# plt.plot(l, np.log2(vl[0, 1:]), label= r'$T = $' + str(tm[0]), marker='x' );
# plt.plot(l, np.log2(vl[1, 1:]), label= r'$T = $' + str(tm[1]), marker='x' );
# plt.plot(l, np.log2(vl[2, 1:]), label= r'$T = $' + str(tm[2]), marker='x' );
# #plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0));
# #plt.grid(True, which ='both');
# plt.xlabel('$h_{\ell}$');
# plt.ylabel('$\log_{2}$ variance');
# plt.xticks(l);
# #plt.title('variance, ' + '$h_{0} = 2^{-10}$');
# plt.legend(fontsize='medium');
# plt.show()
