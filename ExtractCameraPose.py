import numpy as np

def EstimateCameraPose(E, K):
    U,S,V_T = np.linalg.svd(E)
    W = np.zeros((3, 3))
    W[0] = [0, -1, 0]
    W[1] = [1, 0, 0]
    W[2] = [0, 0, 1]
    # print("W",W)
    R = []
    C = []
    R.append(np.dot(U, np.dot(W, V_T.T)))
    R.append(np.dot(U, np.dot(W, V_T.T)))
    R.append(np.dot(U, np.dot(W.T, V_T.T)))
    R.append(np.dot(U, np.dot(W.T, V_T.T)))
    C.append(U[:, 2])
    C.append(- U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    # print("-U",-U)
    # print("U",U)

    for i in range(4):
        if(np.linalg.det(R[i])<0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R,C
