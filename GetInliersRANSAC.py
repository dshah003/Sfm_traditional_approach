import EstimateFundamentalMatrix2 as EFM
import random
import numpy as np


def GetInliersRANSAC(corr_list, thresh):
    print("Into GetInliersRANSAC now,")
    num_points_F = len(corr_list)
    corr_list = np.transpose(corr_list)
    print("Shape of corr_list = " + str(len(corr_list)))
    maxInliers = []
    finalF = None
    for i in range(1001):
        if(i % 40 == 0):
            print("Running Ransac for " + str(i) + "/1000")
        corr = []
        # find n random points to calculate a homography
        for n in range(num_points_F):
            corr.append(corr_list[random.randrange(0, len(corr_list))])

        # Calculate Fundamental Matrix function on those points
        # print("shape of corr for Fundamental Matrix = ", np.shape(corr))
        f = EFM.EstimateFundamentalMatrix(corr, num_points_F)
        inliers = []

        for j in range(len(corr_list)):
            # print("shape corr_list[", j, "]: ", np.shape(corr_list[j]))
            d = geometricDistance(corr_list[j], f)
            if d < thresh:
                inliers.append(corr_list[j])
                # print("--  This is an inliner --")
        # print("len(corr) = ", len(corr))
        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalF = f
        if (i == 1000):
            print("Corr size: " + str(len(corr_list)) + ", NumInliers: " +
                str(len(inliers)) + ", Max inliers: " + str(len(maxInliers)))

        # if len(maxInliers) > (len(corr) * thresh):
            # break

    print("shape of inliers = ", np.shape(maxInliers))
    return finalF, maxInliers


def geometricDistance(correspondence, f):
    # correspondence = np.transpose(correspondence)
    # print("type(correspondence) = ", correspondence[1].item(0))
    p1 = np.transpose(np.matrix([correspondence[0].item(0),
                                correspondence[1].item(0), 1]))
    p2 = np.transpose(np.matrix([correspondence[2].item(0),
                                correspondence[3].item(0), 1]))

    estimatep1 = np.dot(f, p1)
    estimatep2 = np.dot(f, p2)
    # estimatep1 = (1 / estimatep1.item(2)) * estimatep1

    denom = estimatep1[0]**2 + estimatep1[1]**2 + estimatep2[0]**2 + estimatep2[1]**2
    err = (np.diag(np.dot(p1.T, np.dot(f, p2))))**2 / denom

    # return error per point
    # return err

    # error = p2 - estimatep1
    # print("error = ", err)
    # return np.linalg.norm(error)
    return err
