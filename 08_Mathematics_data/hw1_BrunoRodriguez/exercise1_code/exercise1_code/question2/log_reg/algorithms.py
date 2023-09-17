"""
Modified by bruno rodriguez
EPFL. Fall 2021. 
All the codes provided here were tested with Python 3.7.9
The modified sections were the ones labeled #### YOUR CODE GOES HERE
For all required methods, no modifications in # Compute error and save data to be plotted later on.
"""

import time
import numpy as np
import scipy.sparse.linalg as spla
from numpy.random import randint
from scipy.sparse.linalg.dsolve import linsolve

from log_reg.utils import print_end_message, print_start_message, print_progress
from log_reg.operators import l1_prox

##########################################################################
# Unconstrained methods
##########################################################################

def GD(fx, gradf, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Gradient Descent'
    print_start_message(method_name)
    tic_start = time.time()
	
    # Initialize x and alpha.
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    alpha = 1./parameter['Lips']
    d = len(x)
    maxit = parameter['maxit']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit,'x': np.zeros([maxit, x.shape[0]])}

    # Main loop.
    for iter in range(parameter['maxit']):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE        
        x_next = x - alpha * np.matmul( np.identity(d),  gradf(x) )        
        
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)
        info['x'][iter] = x

        # Print the information.
        if (iter %  100 ==0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        
    info['iter'] = maxit
    print_end_message(method_name, time.time() - tic_start)
    return x, info


# gradient with strong convexity
def GDstr(fx, gradf, parameter) :
    """
    Function:  GDstr(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
                strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """

    method_name = 'Gradient Descent with strong convexity'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    L = parameter['Lips']
    mu = parameter['strcnvx']
    alpha = 2./( L + mu )
    d = len(x)
    maxit = parameter['maxit']
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit,'x': np.zeros([maxit, x.shape[0]])}

    # Main loop.
    for iter in range( parameter['maxit'] ):
        # Start timer
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        x_next = x - np.matmul( alpha * np.identity(d),  gradf(x) )

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)
        info['x'][iter] = x

        # Print the information.
        if (iter % 100 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info

# accelerated gradient
def AGD(fx, gradf, parameter):
    """
    Function:  AGD (fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx	  - strong convexity parameter
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
    #### YOUR CODE GOES HERE
    x = parameter['x0'] # x0 
    y = parameter['x0'] # y0
    t = 1 #t0
    L = parameter['Lips']
    alpha = 1./L
    d = len(x)
    maxit = parameter['maxit']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(parameter['maxit']):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        x_next = y - alpha * np.matmul( np.identity(d),  gradf(y) ) 
        t_next = 0.50 * ( 1. + np.sqrt(1. + 4*t**2 ) )
        y_next = x_next + (t - 1. )*(x_next - x)/t_next

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 100 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next
        y = y_next
        t = t_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info

# accelerated gradient with strong convexity
def AGDstr(fx, gradf, parameter):
    """
    Function:  AGDstr(fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx	  - strong convexity parameter
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    method_name = 'Accelerated Gradient with strong convexity'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    y = parameter['x0']
    L = parameter['Lips']
    mu = parameter['strcnvx']
    alpha = ( np.sqrt(L) - np.sqrt(mu) ) / ( np.sqrt(L) + np.sqrt(mu) )    
    maxit = parameter['maxit']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        x_next = y - gradf(y) / L
        y_next = x_next + alpha* (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 100 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        y = y_next
        x = x_next        

    print_end_message(method_name, time.time() - tic_start)
    return x, info

# point C
# LSGD 
def LSGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSGD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent with line-search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    method_name = 'Gradient Descent with line search'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
    #### YOUR CODE GOES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()
        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 100 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info

# point C
# LSAGD
def LSAGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGD (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient with line search'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
    #### YOUR CODE GOES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 100 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# AGDR
def AGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = AGDR (fx, gradf, parameter)
    Purpose:   Implementation of the AGD with adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient with restart'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y, t and find the initial function value (fval).
    #### YOUR CODE GOES HERE
    
    x = parameter['x0'] # x0 
    y = parameter['x0'] # y0
    t = 1 #t0
    L = parameter['Lips']
    alpha = 1./L
    d = len(x)
    maxit = parameter['maxit']
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        x_next = y - alpha *   gradf(y)
        f_previous = fx( x )
        f_next = fx( x_next )
        
        if f_previous < f_next:
            y = x
            t = 1
            x_next = y - alpha * gradf(y)
        
        t_next = 0.50 * ( 1. + np.sqrt(1. + 4*t**2 ) )
        y_next = x_next + (t - 1. )*(x_next - x)/t_next
        
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 100 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration        
        x = x_next
        y = y_next
        t = t_next
        
    print_end_message(method_name, time.time() - tic_start)
    return x, info

# point e
# LSAGDR
def LSAGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGDR (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search and adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient with line search + restart'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y, t and find the initial function value (fval).
    #### YOUR CODE GOES HERE

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 100 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info

def AdaGrad(fx, gradf, parameter):
    """
    Function:  [x, info] = AdaGrad (fx, gradf, hessf, parameter)
    Purpose:   Implementation of the adaptive gradient method with scalar step-size.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Adaptive Gradient method'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, B0, alpha, grad (and any other)
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    alpha = 1.
    delta = 10**-5
    Q = 0
    d = len(parameter['x0'])
    maxit = parameter['maxit']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        Q_next = Q + np.linalg.norm( gradf(x) )**2 # norm is squared
        H = ( np.sqrt(Q_next) + delta ) * np.identity(d)
        x_next = x - alpha * np.matmul( np.linalg.inv(H), gradf(x) )

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 100 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        Q = Q_next
        
    print_end_message(method_name, time.time() - tic_start)
    return x, info
# point g
# Newton
def ADAM(fx, gradf, parameter):
    """
    Function:  [x, info] = ADAM (fx, gradf, parameter)
    Purpose:   Implementation of ADAM.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """

    method_name = 'ADAM'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, beta1, beta2, alphs, epsilon (and any other)
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    alpha = 0.1
    beta1 = 0.90
    beta2 = 0.999
    epsilon = 10**-8    
    d = len(parameter['x0'])
    maxit = parameter['maxit']
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        g = gradf(x)
        m = beta1 * 1
        
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 100 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info

def SGD(fx, gradfsto, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradfsto:
    :param parameter:
    :return:
    """
    method_name = 'Stochastic Gradient Descent'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    #### YOUR CODE GOES HERE
    x = parameter['x0']    
    alpha = 1.
    n = parameter['no0functions']
    maxit = parameter['maxit']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        alpha = 1/(iter + 1) #iter starts at 0 
        i = np.random.randint(0 , high=n)
        x_next = x - alpha*gradfsto(x, i)
        
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 100 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info

def SAG(fx, gradfsto, parameter):
    """
    Function:  [x, info] = SAG(fx, gradfsto, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
    :param fx:
    :param gradfsto:
    :param parameter:
    :return:
    """
    method_name = 'Stochastic Gradient Descent with averaging'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    L_max = parameter['Lmax']
    alpha = 1./(16*L_max)
    p = len(parameter['x0']) #dimension of x vector (number of features)
    n = parameter['no0functions']  # number of samples
    maxit = parameter['maxit'] 
    #print('value of n ', n) # basic debugging 
    #print('value of p ', p) # basic debugging 
    #print('shape of gradient: ', gradfsto( x, 1).shape)  # basic debugging 
    
    v = np.zeros( (n,p) )
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}
    # Main loop.

    for iter in range(maxit):
        tic = time.time()
        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        ik = np.random.randint(0 , n)        
        
        if iter == 0:
            x_next = x - alpha*np.sum(v, axis = 0 )/n
            
        else:
            for i in range(n):
                if i == ik:
                    v[i, :] = gradfsto( x, i)            
                    
            x_next = x - alpha*np.sum(v, axis = 0 )/n
                
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 100 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next    

    print_end_message(method_name, time.time() - tic_start)
    return x, info

def SVR(fx, gradf, gradfsto, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param gradfsto:
    :param parameter:
    :return:
    """
    method_name = 'Stochastic Gradient Descent with variance reduction'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    L_max = parameter['Lmax']
    p = len(parameter['x0'])
    n = parameter['no0functions'] 
    maxit = parameter['maxit']    
    
    gamma = 0.01/L_max 
    q = int( 1000 * L_max )    

    v = np.zeros( p )        
    sum_xl = np.zeros( p )
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        x_tilde = x
        vk = gradf(x_tilde)     
        x_tilde_l = x_tilde
        
        for l in range(0, int(q-1) ):
            il = np.random.randint(1 , n)        
            vl = gradfsto( x_tilde_l , il) - gradfsto( x_tilde , il ) + vk
            x_tilde_l = x_tilde_l - gamma*vl
            sum_xl += x_tilde_l
        x_next = sum_xl/q
                
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 100 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        sum_xl = 0 * sum_xl

    print_end_message(method_name, time.time() - tic_start)
    return x, info

##########################################################################
# Prox
##########################################################################

def SubG(fx, gx, gradfx, parameter):
    """
    Function:  [x, info] = subgrad(fx, gx, gradfx, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
    :param fx:
    :param gx:
    :param gradfx:
    :param parameter:
    :return:
    """
    method_name = 'Subgradient'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize lambda, maxit, x0, alpha and subgrad function
    G = 54  # pre computed with np.linalg.norm(A)
    R = 0.41529129 # pre computed with np.linalg.norm(x0 - xstar)
    #### YOUR CODE GOES HERE
    x0 = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    lmbd = parameter['lambda']    

    print('no0functions ', parameter['no0functions'])

    subgrad = lambda x: gradfx(x) + lmbd * np.sign(x)

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop
    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the next iteration (x and alpha)
        #### YOUR CODE GOES HERE
        alpha = R/(np.sqrt(k+1) * G )
        x_k = x_k - alpha * subgrad(x_k)

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % parameter['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

    print_end_message(method_name, time.time() - tic_start)
    return x_k, info

def ista(fx, gx, gradf, proxg, params):
    """
    Function:  [x, info] = ista(fx, gx, gradf, proxg, parameter)
    Purpose:   Implementation of ISTA.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               prox_Lips  - Lipschitz constant for gradient.
               lambda     - regularization factor in F(x)=f(x)+lambda*g(x).
    :param fx:
    :param gx:
    :param gradf:
    :param proxg:
    :param parameter:
    :return:
    """

    method_name = 'ISTA'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize parameters.
    #### YOUR CODE GOES HERE
    x0 = params['x0']
    L = params['prox_Lips']
    alpha = 1./L
    maxit = params['maxit']    
    lmbd = params['lambda']
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the iterate
        #### YOUR CODEGOES HERE
        value = x_k - alpha * gradf(x_k)
        #print( np.linalg.norm(value) )
        x_k = proxg(value, alpha*lmbd)
                
        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

    print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def fista(fx, gx, gradf, proxg, params):
    """
    Function:  [x, info] = fista(fx, gx, gradf, proxg, parameter)
    Purpose:   Implementation of FISTA (with optional restart).
    Parameter: x0            - Initial estimate.
               maxit         - Maximum number of iterations.
               prox_Lips     - Lipschitz constant for gradient.
               lambda        - regularization factor in F(x)=f(x)+lambda*g(x).
               restart_fista - enable restart.
    :param fx:
    :param gx:
    :param gradf:
    :param proxg:
    :param parameter:
    :return:
    """
    
    if params['restart_fista']:
        method_name = 'FISTAR'
    else:
        method_name = 'FISTA'
    print_start_message(method_name)
        
    tic_start = time.time()
    # Initialize parameters
    #### YOUR CODE GOES HERE
    x0 = params['x0']
    L = params['prox_Lips']
    alpha = 1/L
    t = 1
    maxit = params['maxit']    
    lmbd = params['lambda']
    restart_parameter = params['restart_fista']
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    y_k = x0
    for k in range(maxit):
        tic = time.time()
        
        argument = y_k - alpha * gradf(y_k)    
        x_next = proxg(argument, alpha*lmbd)
        if method_name =='FISTAR':
            if gradient_scheme_restart_condition(x_k, x_next, y_k):
                print('enter momentum loop')
                t_next = 1
                y = x_next
            else:
                t_next = 0.50 * ( 1. + np.sqrt(1. + 4*t**2 ) )            
                y_next = x_next + (t - 1. )*(x_next - x_k)/t_next
        else:
            t_next = 0.50 * ( 1. + np.sqrt(1. + 4*t**2 ) )            
            y_next = x_next + (t - 1. )*(x_next - x_k)/t_next
        # Update iterate
        #### YOUR CODE GOES HERE                
                
        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))
            
        x_k = x_next
        y_k = y_next
        t = t_next

    print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def gradient_scheme_restart_condition(x_k, x_k_next, y_k):
    """
    Whether to restart
    """
    #### YOUR CODE GOES HERE
    condition = np.dot(y_k - x_k_next, x_k_next - x_k ) > 0
    return condition


def prox_sg(fx, gx, gradfsto, proxg, params):
    """
    Function:  [x, info] = prox_sg(fx, gx, gradfsto, proxg, parameter)
    Purpose:   Implementation of ISTA.
    Parameter: x0                - Initial estimate.
               maxit             - Maximum number of iterations.
               prox_Lips         - Lipschitz constant for gradient.
               lambda            - regularization factor in F(x)=f(x)+lambda*g(x).
               no0functions      - number of elements in the finite sum in the objective.
               stoch_rate_regime - step size as a function of the iterate k.
    :param fx:
    :param gx:
    :param gradfsto:
    :param proxg:
    :param parameter:
    :return:
    """
    
    method_name = 'PROXSG'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize parameters
    #### YOUR CODE GOES HERE    
    x0 = params['x0'] 
    maxit = params['maxit']    
    lmbd = params['lambda'] 
    C = params['stoch_rate_regime']  #this is a function of the iteration k
    n = params['no0functions'] # number of data points
    
    X_avg = 0
    gamma_k = 0    
    subgrad = lambda x, j: gradfsto(x, j) + lmbd * np.sign(x) #stochastic gradient
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()            
        ik = np.random.randint(0 , n) # ik is picked at random between 0 and n-1
        y = x_k -  C(k)* subgrad(x_k, ik)
        x_k = proxg(y, C(k)*lmbd)    
        X_avg += x_k * C(k) # where the average is computed, x_k is a vector that stores the sum
        
        gamma_k += C(k) # to store the values of gamma at each iteartion
        
        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(X_avg) + lmbd * gx(X_avg)
        if k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(X_avg), gx(X_avg))
            
    # cimputation  of final value according to the ergodic iterate
    X_avg = X_avg/gamma_k
            
    print_end_message(method_name, time.time() - tic_start)
    return X_avg, info
