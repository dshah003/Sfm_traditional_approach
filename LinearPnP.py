"""Summary
"""
import numpy as np


def convertHomogeneouos(x):
    """Summary

    Args:
        x (TYPE): Description

    Returns:
        TYPE: Description
    """
    m, n = x.shape
    if (n == 3 or n == 2):
        x_new = np.hstack((x, np.ones((m, 1))))
    else:
        x_new = x
    return x_new


def LinearPnP(X, x, K):
    """Summary

    Args:
        X (TYPE): Description
        x (TYPE): Description
        K (TYPE): Description

    Returns:
        TYPE: Description
    """
    N = X.shape[0]
    # print("N = ", np.shape(X))
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    x = np.hstack((x, np.ones((x.shape[0], 1))))

    x = np.transpose(np.dot(np.linalg.inv(K), x.T))
    A = []
    for i in range(N):
        xt = X[i, :].reshape((1, 4))
        z = np.zeros((1, 4))
        p = x[i, :]  #.reshape((1, 3))
        # print("p[1]*xt", np.shape(p[1]*xt))
        # print("p", p)
        # print("xt", xt.shape)
        a1 = np.hstack((np.hstack((z, -xt)), p[1] * xt))
        a2 = np.hstack((np.hstack((xt, z)), -p[0] * xt))
        a3 = np.hstack((np.hstack((-p[1] * xt, p[0] * xt)), z))
        a = np.vstack((np.vstack((a1, a2)), a3))
        # a = np.array([[z, -xt, p[1] * xt], [xt, z, -p[0] * xt],
        #               [-p[1] * xt, p[0] * xt, z]])
        # print(a)
        # print(type(a))
        # a.astype(float)
        if (i == 0):
            A = a
        else:
            A = np.vstack((A, a))
    # print("A type", A)

    # print("shape of A = ", A.shape)

    _, _, v = np.linalg.svd(A)
    # v_t = v.H
    P = v[-1].reshape((3, 4))
    R = P[:, 0:3]
    t = P[:, 3]
    u, _, v = np.linalg.svd(R)
    # print("u", u.shape)
    # print("v", v.shape)
    R = np.matmul(u, v)
    d = np.identity(3)
    d[2][2] = np.linalg.det(np.matmul(u, v))
    R = np.dot(np.dot(u, d), v)
    C = -np.dot(np.linalg.inv(R), t)
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    return C, R
