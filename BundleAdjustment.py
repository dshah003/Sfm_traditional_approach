import numpy as np
import scipy.optimize as opt
import math
from scipy.spatial.transform import Rotation as Rscipy


def minimizeFunction(init, K, V_bundle, traj):
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    
    C = CQ[0:3]
    R = CQ[3:7]
    C = C.reshape(-1, 1)
    r_temp = Rscipy.from_quat([R[0],R[1],R[2],R[3]])
    R = r_temp.as_dcm()
    
    P = np.dot(np.dot(K, R), np.hstack((np.identity(3), -C)))
    
    u_rprj = (np.dot(P[0, :], X.T)).T / (np.dot(P[2, :], X.T)).T
    v_rprj = (np.dot(P[1, :], X.T)).T / (np.dot(P[2, :], X.T)).T
    e1 = x[:,0] - u_rprj
    e2 = x[:,1] - v_rprj
    e = e1+e2

    return sum(e)



def BundleAdjustment(Cset, Rset, X, K, traj, V_bundle):

    n = len(Cset)
    assert len(Cset)==len(Rset),"bundle length error"
    sh = 3*n + 4*n + X.shape[0]*3
    init = np.zeros((1,sh))
    i=0
    for C0,R0 in zip(Cset,Rset):
        q_temp  = Rscipy.from_dcm(R0)
        Q0 = q_temp.as_quat()
        init[0,i:i+8] = [C0[0],C0[1],C0[2], Q0[0], Q0[1], Q0[2], Q0[3]]
        i+=7

    assert i==3*n+4*n,"i different"
    init[0,i:] = X.flatten()

    optimized_param = opt.least_squares(fun=minimizeFunction, method="dogbox",x0=init, args=[K, V_bundle, traj])

    # Cnew = optimized_param.x[0:3]
    
    # R = optimized_param.x[3:7]
    # r_temp = Rscipy.from_quat([R[0],R[1],R[2],R[3]])
    # Rnew = r_temp.as_dcm()

    return R_final,C_final,X_final