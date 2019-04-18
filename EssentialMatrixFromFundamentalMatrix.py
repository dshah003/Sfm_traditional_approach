import numpy as np


def get_camera_param():
    Cam = np.fromfile('Data/calibration_new.txt', sep=' ')
    Cam = Cam.reshape(3, 3)
    return(Cam)


def getEssentialMatrix(F):
    # Normalize the image coordinates
    K = get_camera_param()
    E = np.matmul(K.T, np.matmul(F, K))
    U, S, V_T = np.linalg.svd(E)
    # if np.linalg.det(np.dot(U, V_T)) < 0:
    #     V_T = -V_T
    E = np.dot(U, np.dot(np.diag([1, 1, 0]), V_T))
    return E, U, V_T
