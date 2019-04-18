import numpy as np
import cv2
import matplotlib.pyplot as plt
# from io import StringIO
from findCorrespondance import findCorrespondance
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *


elimination_threshold = 5

for i in range(1, 7):
    for j in range(i, 7):
        if (i != j):
            print("\n ---------- \n\n Finding Correspondance between image " + str(i)
                + " and " + str(j) + ":")
            matching_list = findCorrespondance(i, j)
            correspondence_list = [[row[3]for row in matching_list],
            [row[4]for row in matching_list],
            [row[5]for row in matching_list], [row[6]for row in matching_list]]
#             correspondence_list = np.transpose(correspondence_list)
            # print("shape(correspondence_list): ",
                # np.shape(correspondence_list))
            finalF, inliers = GetInliersRANSAC(correspondence_list, elimination_threshold)
            # print("Final F = ", finalF)
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   # matchesMask = matchesMask, # draw only inliers
                   flags = 2)

            img1 = cv2.imread('Data/' + str(i) + '.jpg')
            img2 = cv2.imread('Data/' + str(j) + '.jpg')
            kp1 = [[row[0] for row in inliers], [row[1] for row in inliers]]
            kp2 = [[row[2] for row in inliers], [row[3] for row in inliers]]
            print("Shape kp1 = ", np.shape(kp1))
            img3 = cv2.drawMatches(img1, kp1, img2, kp2)  # , good, None, **draw_params)
            cv2.imshow("img3",img3)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
A = np.array(correspondence_list)
print(A.shape)
