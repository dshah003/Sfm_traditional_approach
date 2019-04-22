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
from EstimateFundamentalMatrix import EstimateFundamentalMatrix

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

    # Create matrices for All correspondences

    Mx,My,M,Color = loadData(DataPath)

    #  Filter M for inliers

    for i in range(1, n_images):
        for j in range(i+1, n_images + 1):
                img1 = i
                img2 = j
                output = np.logical_and(M[:, img1-1], M[:, img2-1])
                indices, = np.where(output == True)
                if(len(indices)<8):
                    continue
                rgb_list = Color[indices]
                pts1 = np.hstack((Mx[indices,img1-1].reshape((-1,1)),My[indices,img1-1].reshape((-1,1))))
                pts2 = np.hstack((Mx[indices,img2-1].reshape((-1,1)),My[indices,img2-1].reshape((-1,1))))


                _, inliers_a, inliers_b,inlier_index = GetInliersRANSAC(np.float32(pts1), np.float32(pts2),indices)
                assert len(inliers_a)== len(inliers_b)==len(inlier_index),"Length not matched"

                for k in indices:
                    if(k!=inlier_index):
                        M[k,i-1] = 0


    recon_bin = np.zeros((M.shape[0],1))
    X_3D = np.zeros((M.shape[0],3))
    #  We have all inliers at this point in M
    img1 = 1
    img2 = 2

    output = np.logical_and(M[:, img1-1], M[:, img2-1])
    indices, = np.where(output == True)
    rgb_list = Color[indices]

    pts1 = np.hstack((Mx[indices,img1-1].reshape((-1,1)),My[indices,img1-1].reshape((-1,1))))
    pts2 = np.hstack((Mx[indices,img2-1].reshape((-1,1)),My[indices,img2-1].reshape((-1,1))))
    best_F = EstimateFundamentalMatrix(np.float32(pts1), np.float32(pts2))


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



    # X = NonLinearTriangulation(K,np.float32(pts1),np.float32(pts32),X,np.eye(3),np.zeros((3,1)),R,C)

    recon_bin[indices] = 1
    X_3D[indices,:] = X

    # fig = plt.figure(1)
    # ax = plt.axes(projection = '3d')

    # print(X.shape)
    # print(rgb_list[inlier_index].shape)

    assert X.shape == rgb_list[inlier_index].shape, "Num points same"


    # ax.scatter3D(X[:,0], X[:,1], X[:,2], c=rgb_list[inlier_index]/255.0,s=1)
    # plt.scatter(X[:,0], X[:,2],c=rgb_list[inlier_index]/255.0,s=1)
    # axes = plt.gca()
    # axes.set_xlim([-limit,limit])
    # axes.set_ylim([-limit,limit])
    #         # axes.set_zlim([-limit,limit])
    # plt.show()


    Cset = []
    Rset = []

    Cset.append(C)
    Rset.append(R)

    r_indx = np.array([img1,img2])

    for i in range(1,n_images+1):

        if(np.isin(r_indx,i)[0]):
            print(i,"Skiiping this frame")
            continue

        output = np.logical_and(recon_bin, M[:, i-1])
        indices, = np.where(output == True)
        if(len(indices)<6):
            continue

        x = np.array([Mx[indices,i],My[indices,i]])
        X = X_3D[indices,:]

        C,R = PnPRANSAC(X,x,K)

        print
        # C,R = NonLinearPnP(X,x,K,C,R)

        Cset.append(C)
        Rset.append(R)
        r_indx.append(i)

        for j in range(1,len(r_indx)):
            output = np.logical_and(np.logical_and(!recon_bin,M[:,r_indx[j]]),M[:,i])
            indices, = np.where(output==True)
            if(len(indices)<8):
                continue

            x1 = np.array([Mx[indices,r_indx[j]], My[idices,r_indx[j]]])
            x2 = np.array([Mx[indices,i] ,My[indices,i]])
            X = LinearTriangulation(K, Cset[j], Rset[j], C, R, x1, x2)

            # X = NonlinearTriangulation(K, x1, x2, X, Rset[j],Cset[j],R,C);
            X_3D[indices,:] = X
            recon_bin[indices] = 1


if __name__ == '__main__':
    main()



