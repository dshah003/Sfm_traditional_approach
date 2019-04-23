import numpy as np
import cv2
from matplotlib import pyplot as plt

from GetInliersRANSAC import GetInliersRANSAC
from ExtractCameraPose import ExtractCameraPose
from DrawCorrespondence import DrawCorrespondence
from EssentialMatrixFromFundamentalMatrix import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonLinearTriangulation import *
from EstimateFundamentalMatrix import EstimateFundamentalMatrix


MIN_MATCH_COUNT = 10

img1 = cv2.imread('./Images/4.jpeg')  # queryImage
img2 = cv2.imread('./Images/6.jpeg')  # trainImage

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# # # ORB Detector
# orb = cv2.ORB_create()
# kp1, des1 = orb.detectAndCompute(gray1, None)
# kp2, des2 = orb.detectAndCompute(gray2, None)
# print("len(des1) = ", len(des1))
# print("len(des2) = ", len(des2))

# # # Brute Force Matching
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# print("Matches Found")

# # Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)

plt.imshow(img3),plt.show()

pts1 = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,2)
pts2 = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,2)

visualize = True

K = np.array([[3.79728180e+03,0.00000000e+00,2.08038461e+03],
 [0.00000000e+00,3.83458104e+03,1.76272469e+03],
 [0.00000000e+00,0.00000000e+00,1.00000000e+00]])

best_F = EstimateFundamentalMatrix(np.float32(pts1), np.float32(pts2))

if (visualize):
    out = DrawCorrespondence(img1, img2, pts1, pts2)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1000, 600)
    cv2.imshow('image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

E = EssentialMatrixFromFundamentalMatrix(best_F, K)
R_set, C_set = ExtractCameraPose(E, K)
# print(R_set)
# print(C_set)
X_set = []
color = ['r','g','b','k']

for n in range(0, 4):
    X1 = LinearTriangulation(K, np.zeros((3, 1)), np.identity(3),
                            C_set[n].T, R_set[n], np.float32(pts1),
                            np.float32(pts2))
    X_set.append(X1)
    

X, R, C = DisambiguateCameraPose(C_set, R_set, X_set)


# Plotting Linear Triangulation output
plt.scatter(X[:, 0], X[:, 2], c='g', s=4)
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('z')

# ax.set_xlim([-0.5, 0.5])
# ax.set_ylim([-0.1, 2])

X = NonLinearTriangulation(K,np.float32(pts1),np.float32(pts2),X,np.eye(3),np.zeros((3,1)),R,C)