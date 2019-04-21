import numpy as np


def convertHomogeneouos(x):
    m, n = x.shape
    if(n == 3 or n == 2):
        x_new = np.hstack(x, np.ones((m, 1)))
    else:
        x_new = x
    return x_new


def LinearPnP(X, x, K):
    N = X.shape[0]
    X = convertHomogeneouos(X)
    x = convertHomogeneouos(x)

    x = np.transpose(np.linalg.inv(K) * x.T)
    A = []
    for i in range(N):
        xt = X[i][:]
        z = np.zeros((1, 4))
        p = x[i][:]
        a = np.array([[z, -xt, p[2] * xt],
                      [xt, z, -p[1] * xt],
                      [-p[2] * xt, p[1] * xt, z]])
        A = np.vstack((A, a))

    _, b, v = np.linalg.svd(A)
    v_t = v.H
    P = v[:][12].reshape((4, 3))
    R = P[:][0:2]
    t = P[:][3]
    u, _, v = np.linalg.svd(R)
    R = np.matmul(u, v_t)
    d = np.identity(3)
    d[3][3] = np.linalg.det(np.matmul(u, v_t))
    R = np.linalg.multi_dot(u, d, v_t)
    C = - np.dot(np.linalg.inv(R), t)
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    return C, R
