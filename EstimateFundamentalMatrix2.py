import numpy as np


def EstimateFundamentalMatrix(points_list, n):
    points_list = np.transpose(points_list)
    # print("points_list = ", points_list)
    # print("points_list[:][0]", points_list[:][0])
    x1 = points_list[:][0]
    x1 = x1[1:n]
    x1 = np.array(x1).reshape((len(x1), -1))
    x2 = points_list[:][2]
    x2 = x2[1:n]
    x2 = np.array(x2).reshape((len(x2), -1))
    y1 = points_list[:][1]
    y1 = y1[1:n]
    y1 = np.array(y1).reshape((len(y1), -1))
    y2 = points_list[:][3]
    y2 = y2[1:n]
    y2 = np.array(y2).reshape((len(y2), -1))

    C1 = x1 * x2
    C2 = x1 * y2
    C3 = x1
    C4 = y1 * x2
    C5 = y1 * y2
    C6 = y1
    C7 = x2
    C8 = y2
    C9 = np.ones((len(x1), 1))

    # print("C1 shape: ", C1.shape)
    # print("C2 shape: ", C2.shape)
    # print("C3 shape: ", C3.shape)
    # print("C4 shape: ", C4.shape)
    # print("C5 shape: ", C5.shape)
    # print("C6 shape: ", C6.shape)
    # print("C7 shape: ", C7.shape)
    # print("C8 shape: ", C8.shape)
    # print("C9 shape: ", C9.shape)

    A = np.array([C1, C2, C3, C4, C5, C6, C7, C8, C9])
    A = A.reshape((n - 1, 9))
    # print("A shape : ", np.shape(A))
    # print("A = ",A)
    U, S, V_trans = np.linalg.svd(A)
    # print("U Shape: ", U.shape)
    # print("S shape: ", S.shape)
    # print("V shape: ", V_trans.shape)
    F = V_trans[-1].reshape(3, 3)
    # print("F = ", F)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    F / F[2, 2]
    # print("F = ", F)

    return F
