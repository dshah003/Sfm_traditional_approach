import numpy as np
from scipy.optimize import leastsq
import math
import LinearPnP as LPnP


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]


def quaternionNormalize(q):
    Q = q / np.linalg.norm(q)
    return Q


def reprojError(CQ, K, X, x):
    X = LPnP.convertHomogeneouos(X)
    x = LPnP.convertHomogeneouos(x)
    C = CQ[0:2]
    q = CQ[3:6]
    R = quaternion_to_euler(q)
    reproj = LPnP.proj3Dto2D(X, K, C, R)
    e = x[:][:1] - reproj[:][:2]
    return e


def NonlinearPnP(X, x, K, C0, R0):
    Q0 = euler_to_quaternion(R0)
    Q0 = quaternionNormalize(Q0)
    # reprojE = reprojError(C0, K, X, x)
    init = np.hstack(Q0, C0)
    optimized_param = leastsq(fun=reprojError,
                              x0=init, args=[C0, K, X, x])
    Cnew = optimized_param[0:3]
    Rnew = optimized_param[4:7]
    Rnew = quaternion_to_euler(Rnew)
    return Cnew, Rnew
