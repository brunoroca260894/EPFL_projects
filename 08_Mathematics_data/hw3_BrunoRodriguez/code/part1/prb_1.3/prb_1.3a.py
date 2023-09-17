import scipy.io
import numpy as np
from scipy.sparse import csr_matrix
from proj_L1 import proj_L1
from time import time

def proj_nuc(Z, kappa):
    #proj_nuc: This function implements the projection onto nuclear norm ball.
    U, S, Vt = np.linalg.svd(Z, full_matrices=False) # SVD computation
    # S is already a vector
    # project S into the L-1 norm ball
    S_l = proj_L1(S, s=kappa)
    S_l = np.diag(S_l) # create matrix Sl1
    # print('dim of Z', Z.shape)
    # print('dim of U', U.shape)
    # print('dim of Sl1', S_l.shape)
    # print('dim of V^T', Vt.shape)
    proj_Z = U@ S_l @Vt # equation Q-1.1 c)
    return proj_Z

data = scipy.io.loadmat('./dataset/ml-100k/ub_base')  # load 100k dataset

Rating = data['Rating'].flatten()
UserID = data['UserID'].flatten() - 1  # Python indexing starts from 0 whereas Matlab from 1
MovID = data['MovID'].flatten() - 1    # Python indexing starts from 0 whereas Matlab from 1

nM = np.amax(data['MovID'])
nU = np.amax(data['UserID'])

Z = csr_matrix((Rating, (MovID, UserID)),shape=(nM, nU),dtype=float).toarray()
kappa = 5000

tstart = time()
Z_proj = proj_nuc(Z, kappa)
elapsed = time() - tstart
print('proj for 100k data takes {} sec'.format(elapsed))

# NOTE: This one can take few minutes!
data = scipy.io.loadmat('./dataset/ml-1m/ml1m_base')  # load 1M dataset

Rating = data['Rating'].flatten()
UserID = data['UserID'].flatten() - 1  # Python indexing starts from 0 whereas Matlab from 1
MovID = data['MovID'].flatten() - 1    # Python indexing starts from 0 whereas Matlab from 1

nM = np.amax(data['MovID'])
nU = np.amax(data['UserID'])

Z = csr_matrix((Rating, (MovID, UserID)),shape=(nM, nU),dtype=float).toarray()
kappa = 5000

tstart = time()
Z_proj = proj_nuc(Z, kappa)
elapsed = time() - tstart
print('proj for 1M data takes {} sec'.format(elapsed))
