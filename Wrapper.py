import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import argparse

from LoadData import *
from GetInliersRANSAC import GetInliersRANSAC
from ExtractCameraPose import ExtractCameraPose
from DrawCorrespondence import DrawCorrespondence
from EssentialMatrixFromFundamentalMatrix import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonLinearTriangulation import *

# Camera Intrinsic Matrix
K = np.array([[568.996140852,0,643.21055941],
         [0, 568.988362396, 477.982801038],
         [0, 0, 1]])
# img1 = 3
# img2 = 5
n_images = 6
limit = 10



def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="./Data/", help='Folder of Images')
    Parser.add_argument('--Visualize', default=False, help='Show correspondences')
    Args = Parser.parse_args()
    DataPath = Args.DataPath
    visualize = Args.Visualize
    

    Mx,My,M = loadData(DataPath)


    # for i in range(1, n_images):
    #     for j in range(i+1, n_images + 1): 

    img1 = 1
    img2 = 2
    _,_,_,rgb_list = findCorrespondance(img1, img2,DataPath)


    output = np.logical_and(M[:, img1-1], M[:, img2-1])
    indices, = np.where(output == True)




    pts1 = np.hstack((Mx[indices,img1-1].reshape((-1,1)),My[indices,img1-1].reshape((-1,1))))
    pts2 = np.hstack((Mx[indices,img2-1].reshape((-1,1)),My[indices,img2-1].reshape((-1,1))))
    best_F, inliers_a, inliers_b,inlier_index = GetInliersRANSAC(np.int32(pts1), np.int32(pts2))

    assert len(inliers_a)== len(inliers_b)==len(inlier_index),"Length not matched"

    # if(len(inlier_index)<50):
    #     continue

    if(visualize):
        out = DrawCorrespondence(img1, img2, inliers_a, inliers_b)
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1000,600)
        cv2.imshow('image', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    E = EssentialMatrixFromFundamentalMatrix(best_F,K)
    R_set,C_set = ExtractCameraPose(E,K)

    X_set = []
    for n in range(0,4):
        X_set.append(LinearTriangulation(K,np.zeros((3,1)),np.identity(3),C_set[n].T,R_set[n],np.int32(inliers_a),np.int32(inliers_b)))

    X, R, C = DisambiguateCameraPose(C_set, R_set, X_set)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection = '3d')

    # print(X.shape)
    # print(rgb_list.shape)

    # assert X.shape == rgb_list[inlier_index].shape, "Num points same"


    # ax.scatter(X[:,0], X[:,1], X[:,2], c=rgb_list[inlier_index]/255.0)
    plt.scatter(X[:,0], X[:,2],cmap='g',s=1)
    axes = plt.gca()
    # axes.set_xlim([-limit,limit])
    # axes.set_ylim([-limit,limit])
    plt.show()

    # X_final = NonLinearTriangulation(K,x1,x2,X,R1,C1,R2,C2)

    # plt.scatter(X_final[:,0], X_final[:,2], c=X_final[:,2], cmap='viridis',s = 1)
    # axes = plt.gca()
    # axes.set_xlim([-limit,limit])
    # axes.set_ylim([-limit,limit])
    # plt.show()




if __name__ == '__main__':
    main()



