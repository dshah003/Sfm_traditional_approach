import EstimateFundamentalMatrix2 as EFM
import random
import numpy as np


def GetInliersRANSAC(corr_list, thresh):
    num_points_F = 15
    maxInliers = []
    finalF = None
    for i in range(1000):
        corr = []
        # find n random points to calculate a homography
        for n in range(num_points_F):
            corr.append(corr_list[random.randrange(0, len(corr_list))])

        # Calculate Fundamental Matrix function on those points
        f = EFM.EstimateFundamentalMatrix(corr, 15)
        inliers = []

        for i in range(len(corr_list)):
            d = geometricDistance(corr_list[i], f)
            if d < 5:
                inliers.append(corr_list[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalF = f
        print "Corr size: ", len(corr_list), " NumInliers: ",
        len(inliers), "Max inliers: ", len(maxInliers)

        if len(maxInliers) > (len(corr) * thresh):
            break
    return finalF, maxInliers


def geometricDistance(correspondence, f):
    p1 = np.transpose(np.matrix([correspondence[0].item(0),
                                correspondence[0].item(1), 1]))
    estimatep2 = np.dot(f, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2),
                                correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)
