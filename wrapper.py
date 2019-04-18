import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from io import StringIO
from findCorrespondance import findCorrespondance
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *


elimination_threshold = 0.4

for i in range(1, 6):
    for j in range(i, 6):
        if (i != j):
            print("Finding Correspondance between image ", i, " and ", j, ":")
            matching_list = findCorrespondance(i, j)
            correspondence_list = [[row[3]for row in matching_list],
            [row[4]for row in matching_list],
            [row[5]for row in matching_list], [row[6]for row in matching_list]]
#             correspondence_list = np.transpose(correspondence_list)
            print("shape(correspondence_list): ",
                np.shape(correspondence_list))
            finalF, inliers = GetInliersRANSAC(correspondence_list, elimination_threshold)

A = np.array(correspondence_list)
print(A.shape)
