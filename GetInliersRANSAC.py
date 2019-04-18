import EstimateFundamentalMatrix2 as EFM
import random
import numpy as np


def GetInliersRANSAC(corr_list, thresh):
    print("Into GetInliersRANSAC now,")
    num_points_F = 15
    corr_list = np.transpose(corr_list)
    print("Shape of corr_list = ", len(corr_list))
    maxInliers = []
    finalF = None
    for i in range(401):
        if(i % 40 == 0):
            print("Running Ransac for ", str(i) + "/400")
        corr = []
        # find n random points to calculate a homography
        for n in range(num_points_F):
            corr.append(corr_list[random.randrange(0, len(corr_list))])

        # Calculate Fundamental Matrix function on those points
        # print("shape of corr for Fundamental Matrix = ", np.shape(corr))
        f = EFM.EstimateFundamentalMatrix(corr, 15)
        inliers = []

        for i in range(len(corr_list)):
            # print("shape corr_list[", i, "]: ", np.shape(corr_list[i]))
            d = geometricDistance(corr_list[i], f)
            if d < thresh:
                inliers.append(corr_list[i])
        # print("len(corr) = ", len(corr))
        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalF = f
        if (i == 400):
            print("Corr size: ", len(corr_list), " NumInliers: ",
                len(inliers), "Max inliers: ", len(maxInliers))

        # if len(maxInliers) > (len(corr) * thresh):
            # break
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
