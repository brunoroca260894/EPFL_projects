"""
Modified by bruno rodriguez
EPFL. Fall 2021. 
No major changes to the required ones were made on this code.
All the codes provided here were tested with Python 3.7.9
The reconstructed images were created with FISTA only for 
the three povided pictures and they were plotted using the 
optimal parameter lambda L-1 and lambda L-TV
Here, the FISTAR was not implemented nor tested
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.measure import compare_ssim as ssim

from common.utils import apply_random_mask, psnr, load_image
from common.operators import TV_norm, Representation_Operator, p_omega, p_omega_t, l1_prox, norm2sq,norm1
from common.utils import print_end_message, print_start_message, print_progress

def ISTA(fx, gx, gradf, proxg, params):
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

def FISTA(fx, gx, gradf, proxg, params, verbose=False):
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
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    y_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update iterate
        #### YOUR CODE GOES HERE    
        argument = y_k - alpha * gradf(y_k).reshape((-1, 1))
        #x_next = proxg(argument, alpha*lmbd)
        x_next = proxg(argument, alpha)
        t_next = 0.50 * ( 1. + np.sqrt(1. + 4*t**2 ) )            
        y_next = x_next + (t - 1. )*(x_next - x_k)/t_next
                
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

def reconstructL1(image, indices, optimizer, params):
    # Wavelet operator
    r = Representation_Operator(m=params["m"])

    # Define the overall operator
    m = params["m"]
    
    forward_operator = lambda x: p_omega( r.WT(x), indices) # P_Omega.W^T
    adjoint_operator = lambda x: r.W( p_omega_t(x, indices, m) ) # W. P_Omega^T

    # Generate measurements
    b = p_omega( image , indices)  ## TO BE FILLED ##

    fx = lambda x: norm2sq(b - forward_operator(x) )
    gx = lambda x: norm1(x) ## TO BE FILLED ##
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)
    gradf = lambda x: -adjoint_operator( b - forward_operator(x) ) ## TO BE FILLED ##

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return r.WT(x).reshape((params['m'], params['m'])), info

def reconstructTV(image, indices, optimizer, params):
    """
        image: undersampled image (mxm) to be reconstructed
        indices: indices of the undersampled locations
        optimizer: method of reconstruction (FISTA/ISTA function handle)
        params:
    """
    # Define the overall operator
    m = params["m"]    
    forward_operator = lambda x: p_omega( x, indices) ## P_Omega
    adjoint_operator = lambda x: p_omega_t(x, indices, m) ## P_Omega^T
    
    # Generate measurements
    b = forward_operator(image)

    fx = lambda x: norm2sq(b - forward_operator(x) ) # TO BE FILLED ##
    gx = lambda x:  TV_norm(x) ## TO BE FILLED ##
    proxg = lambda x, y: denoise_tv_chambolle(x.reshape((params['m'], params['m'])),
                                              weight=params["lambda"] * y, eps=1e-5,
                                              n_iter_max=50).reshape((params['N'], 1))
    gradf = lambda x: -adjoint_operator( b - forward_operator(x) )  # TO BE FILLED ##

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return x.reshape((params['m'], params['m'])), info

# %%
if __name__ == "__main__":
    ##############################
    # Load image and sample mask #
    ##############################
    shape = (256, 256)
    params = {
        'maxit': 200,
        'tol': 10e-15,
        'prox_Lips': 1 , ## TO BE FILLED ##,
        'lambda': 0.01 , ## TO BE FILLED ##,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'restart_fista':  False ,## TO BE FILLED ##, gradient_scheme,
        #'stopping_criterion':  1 , ## TO BE FILLED ##,
        'iter_print': 50,
        'shape': shape,
        'restart_param': 50,
        'verbose': True,
        'm': shape[0],
        'rate': 0.4,
        'N': shape[0] * shape[1]
    }
    
    images_data = ['gandalf.jpg', 'lauterbrunnen.jpg', 'lena.jpg']
    
    for ima in images_data:        
        PATH = 'data/'+ ima
        image = load_image(PATH, params['shape'])
        
        im_us, mask = apply_random_mask(image, params['rate'])
        indices = np.nonzero(mask.flatten())[0]
        params['indices'] = indices
        # Choose optimization parameters
        #######################################
        # Reconstruction with L1 and TV norms #
        #######################################
        # lambda parameters to test which one gives the highest 
        # psn
        lamda_vector = np.logspace(-4,-0.1, 20)
        psnr_l1_vector = np.zeros( (len(lamda_vector)) )
        psnr_tv_vector = np.zeros( (len(lamda_vector)) )
                
        # to compute the optimal value of l1 norm
        counter = 0
        for lb in lamda_vector:
            params['lambda'] = lb
            print('lambda changing ', lb)
            
            print('reconstruction_l1')
            t_start = time.time()
            reconstruction_l1 = reconstructL1(image, indices, FISTA, params)[0]
            t_l1 = time.time() - t_start
        
            psnr_l1 = psnr(image, reconstruction_l1)
            ssim_l1 = ssim(image, reconstruction_l1)
            psnr_l1_vector[counter] = psnr_l1
            counter +=1
            print('----------------------')        
            
        # to compute the optimal value of TV norm
        counter = 0
        for lb in lamda_vector:
            params['lambda'] = lb
            print('lambda changing ', lb)
        
            print('reconstructTV')
            t_start = time.time()
            reconstruction_tv = reconstructTV(image, indices, FISTA, params)[0]
            t_tv = time.time() - t_start
        
            psnr_tv = psnr(image, reconstruction_tv)
            ssim_tv = ssim(image, reconstruction_tv)
            psnr_tv_vector[counter] = psnr_tv
            counter +=1
            print('----------------------')
        
        # max psnr for l1 norm
        id_l1 = np.argmax(psnr_l1_vector)
        # max psnr for TV norm
        id_TV = np.argmax(psnr_tv_vector)
        
        #computing with optimal lambdas 
        # l1 norm
        print('l1, picture ' + ima + ' ' + str(lamda_vector[id_l1]))
        params['lambda'] = lamda_vector[id_l1]
        t_start = time.time()
        reconstruction_l1 = reconstructL1(image, indices, FISTA, params)[0]
        t_l1 = time.time() - t_start
        psnr_l1 = psnr(image, reconstruction_l1)
        ssim_l1 = ssim(image, reconstruction_l1)
        
        # TV norm
        print('TV, picture ' + ima + ' ' +str( lamda_vector[id_TV]) )
        params['lambda'] = lamda_vector[id_TV]
        t_start = time.time()
        reconstruction_tv = reconstructTV(image, indices, FISTA, params)[0]
        t_tv = time.time() - t_start
        psnr_tv = psnr(image, reconstruction_tv)
        ssim_tv = ssim(image, reconstruction_tv)
        
        # Plot the reconstructed image alongside the original image and PSNR
        fig, ax = plt.subplots(1, 4, figsize=(20, 8))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original')
        ax[1].imshow(im_us, cmap='gray')
        ax[1].set_title('Original with missing pixels')
        ax[2].imshow(reconstruction_l1, cmap="gray")
        ax[2].set_title('L1 - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_l1, ssim_l1, t_l1))
        ax[3].imshow(reconstruction_tv, cmap="gray")
        ax[3].set_title('TV - PSNR = {:.2f}\n SSIM  = {:.2f}  - Time: {:.2f}s'.format(psnr_tv, ssim_tv, t_tv))
        [axi.set_axis_off() for axi in ax.flatten()]
        plt.tight_layout()
        plt.savefig("reconstructedImages_" + ima + ".pdf")
        plt.show()
        
        # plot PSNR for optimal lambadas 
        # l1 norm
        plt.loglog(lamda_vector, psnr_l1_vector, c = 'b', marker = 'x');
        plt.title(r'$\ell$-1 norm')
        plt.xlabel(r'$\lambda$');
        plt.ylabel('PSNR');
        plt.grid(True, which ='both');
        plt.savefig("PSNRImages_" + ima + "l1" + ".pdf")
        plt.show()
        
        # TV norm
        plt.loglog(lamda_vector, psnr_tv_vector, c = 'b', marker = 'x');
        plt.title(r'TV norm')
        plt.xlabel(r'$\lambda$');
        plt.ylabel('PSNR');
        plt.grid(True, which ='both');
        plt.savefig("PSNRImages_" + ima + "TV" + ".pdf")
        plt.show()
        
        print('next image contruction ')
        print('------xxxXXXxxx------')
