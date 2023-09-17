# -*- coding: utf-8 -*-
"""
EPFL Spring 2022. 
Numerical integration of stochastic differential equations
Mini project. 
author: bruno
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib import rc
import math 
from numba import jit, njit, float32

plt.rc("legend", framealpha=None)
plt.rc("legend", edgecolor='black')
plt.rc("font", family="serif")

np.random.seed(41)

#Euler-Maruyama method
@njit
def SDE_EM(alpha, sigma, T, X0, h): 
    N = int(T/h) #number of points for each path   
    X = X0*np.ones(shape =N)
    for i in range(0, N-1):        
        X[i+1] =X[i] - alpha*X[i]*h + np.sqrt(2*sigma)*np.sqrt(h)*np.random.randn()
    return X

############################################
##############    Q3    ####################
T=10**3 
h=10**-2 
M=10**4 #total runs
alpha=1
sigma=1
X0=0
data = np.zeros(shape = (M)) # store value of X at time T

for i in range(M):
    X = SDE_EM(alpha, sigma, T, X0, h)
    data[i] = X[-1];

mu = 0
variance = sigma/alpha
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

plt.hist(data, bins=40, density=True, alpha=0.3,
         histtype='stepfilled',color='steelblue',
         edgecolor='none', label = r'$\{ X^{(m)}(T) \}^{M}_{m=1}$');    
plt.plot(x, stats.norm.pdf(x, mu, sigma), label=r'density $\rho_{\infty}$',c='k', ls='--') 
plt.legend(loc ='upper left', fontsize='medium')
plt.savefig('Q3_histogram.pdf', dpi=300)
plt.show()  

# %%
############################################
##############    Q7    ####################
# @njit
# def sigma_numerical(X, delta, N):
#     num = np.sum((np.diff(X))**2)
#     sigma_num = num/(2*delta*N)    
#     return sigma_num


def sigma_numerical(X, delta, T):
    #how many samples we take for given numerical solution
    N = int(T/delta)
    X_tilde = np.zeros(N)
    step_equi = int(np.round(len(X)/N))
    mask = np.arange(0, len(X), step_equi)
    
    X_tilde = X[mask]
    
    sums = np.sum((np.diff(X_tilde))**2)
    
    sigma_num = sums/(2*T)    
    return sigma_num

def alpha_numerical(X, delta, T):
    N = int(T/delta)
    X_tilde = np.zeros(N)
    step_equi = int(np.round(len(X)/N))
    mask = np.arange(0, len(X), step_equi)
    
    X_tilde = X[mask]
        
    num = -np.sum( np.diff(X_tilde)*X_tilde[:-1] )
    den = np.sum(X_tilde[:-1]**2)
    alpha_num = num/(delta*den)
    return alpha_num

T=10**3 
h=10**-3 
alpha=1
sigma=1
X0=0
delta=np.array([ 1./2**i for i in range(0, 8) ])
N = T/delta
N = N.astype(int)

sol_alpha = np.zeros(shape = len(delta))
sol_sigma = np.zeros(shape = len(delta))

multiple_alpha = np.zeros(shape = len(delta))
multiple_sigma = np.zeros(shape = len(delta))

runs = 20
for k in range(runs):    
    j = 0
    sol_EM = SDE_EM(alpha, sigma, T, X0, h) # one solution only
    for n,dlt in zip(N, delta):    
        print(n)
        sol_sigma[j] = sigma_numerical(sol_EM, dlt, T)
        sol_alpha[j] = alpha_numerical(sol_EM, dlt, T)
        j=j+1
         
    multiple_alpha += sol_alpha
    multiple_sigma += sol_sigma
    
    plt.plot(delta, sol_alpha,c='b', lw =0.1) 
    plt.plot(delta, sol_sigma,c='k', lw =0.1)
    
        
plt.plot(delta, sigma*delta/delta, label=r'true $\sigma$',c='k', ls='--') 
plt.plot(delta, multiple_sigma/20, label=r'mean $\sigma_{N}^{\Delta}$',c='k', marker='x') 
plt.plot(delta, alpha*delta/delta, label=r'true $\alpha$',c='k', ls='--') 
plt.plot(delta, multiple_alpha/20, label=r'mean $\alpha_{N}^{\Delta}$',c='b', marker='x') 
plt.xscale("log")
plt.xlabel(r'$\Delta$')
plt.legend(loc ='upper right', fontsize='medium')
plt.savefig('Q7_histogram02.pdf', dpi=300)
plt.show()  

# %%
############################################
##############    Q12    ###################
T=10**3 
h=10**-3 
alpha=1
sigma=1
X0=0
delta=np.array([ 1./2**i for i in range(0, 8) ])
N = T/delta
N = N.astype(int)

sol_alpha = np.zeros(shape = len(delta))

def alpha_numerical_2(X, delta, T):
    N = int(T/delta)
    X_tilde = np.zeros(N)
    step_equi = int(np.round(len(X)/N))
    mask = np.arange(0, len(X), step_equi)
    #print(step_equi)
    X_tilde = X[mask]
        
    num =np.sum( X_tilde[:-1]*X_tilde[1:])
                    
    den = np.sum(X_tilde[:-1]**2)
    alpha_num = -np.log( num/den )/delta
    return alpha_num

#np.random.seed(41)

## multiple runs 
runs = 20
alpha_multiple=  np.zeros(shape = len(delta))

for k in range(runs):    
    j = 0
    sol_EM = SDE_EM(alpha, sigma, T, X0, h) # one solution only
    for n,dlt in zip(N, delta):    
        sol_alpha[j] = alpha_numerical_2(sol_EM, dlt, T)
        j=j+1
    alpha_multiple += sol_alpha
    plt.plot(delta, sol_alpha, marker='', lw=0.6) 

plt.plot(delta, alpha_multiple/20, label=r'$\alpha_{N}^{\Delta}$, mean over 20 runs', marker='x',c='k', lw=2) 
plt.plot(delta, alpha*delta/delta, label=r'true $\alpha$',c='k', ls='--') 
plt.xscale("log")
plt.xlabel(r'$\Delta$')
plt.legend(loc ='upper right', fontsize='medium')
plt.savefig('Q12_histogram02.pdf', dpi=300)
plt.show()  

# %%
############################################
##############    Q13    ###################
T=10**3 
h=10**-2
alpha=1
sigma=1
M=10**4
X0=0
delta=1
N = int(T/delta)
data = np.zeros(shape = (M)) # store value of X at time T

sol_alpha = np.zeros(shape = (M))

for i in range(M):
    sol_EM = SDE_EM(alpha, sigma, T, X0, h)
    sol_alpha[i] = alpha_numerical_2(sol_EM, delta, T)    

alpha_distribution = np.sqrt(N)*(sol_alpha - alpha)
## theoretical distribution
S = (np.exp(2*alpha*delta)-1)/delta**2

mu = 0
variance = S
sigma = math.sqrt(variance)
x = np.linspace(mu - 3.5*sigma, mu + 3.5*sigma, 500)
true_distribution = stats.norm.pdf(x, mu, sigma)

plt.hist(alpha_distribution, bins=50, density=True, alpha=0.3,
         histtype='stepfilled',color='steelblue',
         edgecolor='none', label =r'$\sqrt{N}\left(  \widetilde{\alpha}_{N}^{\Delta, (m)}-\alpha \right)$');    
plt.plot(x, true_distribution, label=r'$\widetilde{\mu}$',c='k', ls='--', lw=2) 
plt.legend(loc ='upper left', fontsize='medium')
plt.savefig('Q13_histogram.pdf', dpi=300)
plt.show()  

# %%
##############    Q15    ###################
T=5*10**3 
h=10**-2
alpha=1
sigma=1
X0=0
delta=1
N = int(T/delta)

def alpha_sigma_numerical(sol_SDE, delta, T):
    N = int(T/delta)
    sol_alpha = np.zeros(N)
    sol_sigma = np.zeros(N)
    X_tilde = np.zeros(N)
    step_equi = int(np.round(len(sol_SDE)/N))
    mask = np.arange(0, len(sol_SDE), step_equi)    
    X_tilde = sol_SDE[mask]
    
    initial =[0.5, 0.5]
    
    for i in range(1, len(X_tilde)):
        sol = fsolve( linear_system, initial, args=(X_tilde[:i], N)  )
        #print('after fsolve')
        sol_alpha[i] = sol[0]
        sol_sigma[i] = sol[1]
        
    return sol_alpha,sol_sigma
     
def linear_system(x, sol_SDE, N, delta=1):    
    # x[0] corresponds to a
    # x[1] corresponds to s
    t1 = sol_SDE[1:]-np.exp(-x[0]*delta)*sol_SDE[:-1] + sol_SDE[1:]**2 -(x[1]/x[0]) \
        -np.exp(-2*x[0]*delta)*( sol_SDE[:-1]**2 - (x[1]/x[0]) )
    e1= np.sum( t1*(sol_SDE[:-1])**2 )/N
    e2= np.sum( t1*(sol_SDE[:-1]) )/N
    return [e1, e2]
 
alpha_multiple = np.zeros(5000)
sigma_multiple = np.zeros(5000)
runs = 5 # independent runs

for k in range(runs):    
    sol_EM = SDE_EM(alpha, sigma, T, X0, h)
    al, sig =  alpha_sigma_numerical(sol_EM, delta, T)
    alpha_multiple += al
    sigma_multiple += sig

    plt.plot(al, c='k', ls='-', lw=0.05) 
    plt.plot(sig, c='b', ls='-', lw=0.05)
    
plt.plot(alpha_multiple/runs, label=r'mean $\widetilde{\alpha}_{N}^{\Delta}$',c='g', ls='-', lw=0.8) 
plt.plot(sigma_multiple/runs, label=r'mean $\widetilde{\sigma}_{N}^{\Delta}$',c='r', ls='-', lw=0.8)        

plt.plot(alpha*al/al, label=r'true $\alpha$', c='g', ls='--', lw=0.8) 
plt.plot(sigma*sig/sig, label=r'$true \sigma$', c='r', ls='--', lw=0.8) 
plt.ylim([0,2]) 
plt.xlabel(r'number of equispaced observations')
plt.legend(loc ='lower right', fontsize='medium')
plt.savefig('Q15_plot02.pdf', dpi=300)
plt.show()  