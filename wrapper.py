import numpy as np
import cv2
# import matplotlib.pyplot as plt
from findCorrespondance import findCorrespondance
from GetInliersRANSAC import GetInliersRANSAC
# from EstimateFundamentalMatrix import EstimateFundamentalMatrix
# from EstimateFundamentalMatrix3 import *
from DrawCorrespondence import DrawCorrespondence
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix


# elimination_threshold = 5

all_Fs = []
all_inliers = []

for i in range(1, 4):
    for j in range(i, 4):
        if (i != j):
            print("\n ---------- \n\n Finding Correspondance between image "
                + str(i)
                + " and " + str(j) + ":")
            matching_list = findCorrespondance(i, j)

            pts1 = []
            pts2 = []
            for row in matching_list:
                pt1 = (row[3], row[4])
                pt2 = (row[5], row[6])
                pts1.append(pt1)
                pts2.append(pt2)

            best_F, inliers_a, inliers_b = GetInliersRANSAC(np.int32(pts1), np.int32(pts2))
            # finalF, inliers = GetInliersRANSAC(correspondence_list, elimination_threshold)
            print("Final F = ", best_F)
            all_Fs.append(best_F)
            all_inliers.append(np.hstack((inliers_a, inliers_b)))
            print("Total Number of points found between image " + str(i) + " and "
                + str(j) + " is = " + str(len(matching_list)))

            print("Number of inliners between image " + str(i) + " and "
                + str(j) + " is = " + str(len(inliers_a)))

            out = DrawCorrespondence(i, j, inliers_a, inliers_b)

            cv2.imshow("img3", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

print("shape of all_inliers", np.shape(all_inliers))
print("shape of all_Fs", np.shape(all_Fs))
# A = np.array(correspondence_list)
# print(A.shape)
