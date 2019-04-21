import numpy as np
import LinearPnP as LPnP
import random


def proj3Dto2D(x3D, K, C, R):
    P = np.linalg.multi_dot(K, R, np.hstack(np.identity(3), -C))
    X3D = LPnP.convertHomogeneouos(x3D)

    u_rprj = (np.dot(P[1], X3D.H)).H / (np.dot(P[3], X3D.H)).H
    v_rprj = (np.dot(P[2], X3D.H)).H / (np.dot(P[3], X3D.H)).H
    X2D = np.hstack(u_rprj, v_rprj)
    return X2D


def PnPRANSAC(X, x, K):
    cnt = 0
    M = x.shape[0]
    # p = 0.99
    threshold = 6
    # N = 1
    # idx = 0
    # X_ = LPnP.convertHomogeneouos(X)
    x_ = LPnP.convertHomogeneouos(x)

    Cnew = np.zeros((3, 1))
    Rnew = np.identity(3)

    for trails in range(500):
        # random.randrange(0, len(corr_list))
        random_idx = random.sample(range(M), 6)
        C, R = LPnP.LinearPnP(X[random_idx][:], x[random_idx][:], K)
        S = []
        for j in range(M):
            reprojection = proj3Dto2D(x_[j][:], K, C, R)
            e = np.square(np.sqrt((x_[j][1]) - reprojection[1]) +
                          np.square((x_[j][2] - reprojection[2])))
            if e < threshold:
                S.append(j)
        countS = len(S)
        if(cnt < countS):
            cnt = countS
            Rnew = R
            Cnew = C

        if (countS == M):
            break
    print("Inliers = %d / %d \n", cnt, M)
    return Cnew, Rnew
