import numpy as np


def EstimateCameraPose(E, K, U, V_T):
    W = np.zeros((3, 3))
    W[0] = [0, -1, 0]
    W[1] = [1, 0, 0]
    W[2] = [0, 0, 0]
    R1 = np.dot(U, np.dot(W, V_T))
    R2 = np.dot(U, np.dot(W, V_T))
    R3 = np.dot(U, np.dot(W.T, V_T))
    R4 = np.dot(U, np.dot(W.T, V_T))
    C1 = U[:, 3]
    C2 = - U[:, 3]
    C3 = U[:, 3]
    C4 = -U[:, 3]
    Ident = np.identity(3)

    P1 = np.dot(K, np.dot(R1, np.hstack(Ident, -C1)))
    P2 = np.dot(K, np.dot(R2, np.hstack(Ident, -C2)))
    P3 = np.dot(K, np.dot(R3, np.hstack(Ident, -C3)))
    P4 = np.dot(K, np.dot(R4, np.hstack(Ident, -C4)))
    return P1, P2, P3, P4
