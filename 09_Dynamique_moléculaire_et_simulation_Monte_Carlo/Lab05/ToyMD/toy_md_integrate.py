#!/usr/bin/env python3

import math

def put_in_box(box, resnr, coords):
    N      = len(coords)
    cgcm   = []
    old    = -1
    invres = []
    for i in range(N):
        if (resnr[i] != old):
            cgcm.append([ 0.0, 0.0, 0.0 ])
            invres.append([])
            old = resnr[i]
        for m in range(3):
            cgcm[len(cgcm)-1][m] += coords[i][m]
        invres[len(invres)-1].append(i)
    N = len(cgcm)
    for i in range(N):
        for m in range(3):
            cgcm[i][m] /= len(invres[i])
    for i in range(N):
        for m in range(3):
            if (cgcm[i][m] > box[m]):
                for k in invres[i]:
                    coords[k][m] -= box[m]
            if (cgcm[i][m] <= 0):
                for k in invres[i]:
                    coords[k][m] += box[m]
    
def compute_lambda_T(T_inst, T_0, time_step, tau_T):
    lam = 0
    if (T_inst == 0 or tau_T == 0):
        lam = 1
    else:
        lam = 1 + (time_step/tau_T)*( T_0/T_inst -1)
    return math.sqrt(lam) 

