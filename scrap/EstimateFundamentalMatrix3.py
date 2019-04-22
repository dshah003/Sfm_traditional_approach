import numpy as np

def estimate_fundamental_matrix(points_a, points_b):
    points_num = points_a.shape[0]
    A = []
    B = np.ones((points_num, 1))

    cu_a = np.sum(points_a[:, 0])/points_num
    cv_a = np.sum(points_a[:, 1])/points_num

    s = points_num/np.sum(((points_a[:, 0]-cu_a)**2 + (points_a[:, 1]-cv_a)**2)**(1/2))
    T_a =np.dot(np.array([[s, 0, 0], [0,s,0], [0,0,1]]), np.array([[1,0,-cu_a],[0,1,-cv_a],[0,0,1]]))

    points_a = np.array(points_a.T)
    points_a = np.append(points_a,B)

    points_a = np.reshape(points_a, (3,points_num))
    points_a = np.dot(T_a, points_a)
    points_a = points_a.T

    cu_b = np.sum(points_b[:, 0])/points_num
    cv_b = np.sum(points_b[:, 1])/points_num

    s = points_num/np.sum(((points_b[:,0]-cu_b)**2 + (points_b[:,1]-cv_b)**2)**(1/2))
    T_b =np.dot(np.array([[s,0,0], [0,s,0], [0,0,1]]), np.array([[1,0,-cu_b],[0,1,-cv_b],[0,0,1]]))

    points_b = np.array(points_b.T)
    points_b = np.append(points_b,B)

    points_b = np.reshape(points_b, (3,points_num))
    points_b = np.dot(T_b, points_b)
    points_b = points_b.T

    for i in range(points_num):
        u_a = points_a[i,0]
        v_a = points_a[i,1]
        u_b = points_b[i,0]
        v_b = points_b[i,1]
        A.append([u_a*u_b, v_a*u_b, u_b, u_a*v_b, v_a*v_b, v_b, u_a, v_a,1])

#     A = np.array(A)
#     F = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, -B))
#     F = np.append(F,[1])

    _, _, v = np.linalg.svd(A)
    F = v[-1]

    F = np.reshape(F, (3, 3)).T
    F = np.dot(T_a.T, F)
    F = np.dot(F, T_b)

    F = F.T
    U, S, V = np.linalg.svd(F)
    S = np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]])
    F = np.dot(U, S)
    F = np.dot(F, V)

    F = F/F[2, 2]

    return F


def ransac_fundamental_matrix(matches_a, matches_b):
    matches_num = matches_a.shape[0]
    Best_count = 0

    for iter in range(500):
        sampled_idx = np.random.randint(0, matches_num, size = 8)
        F = estimate_fundamental_matrix(matches_a[sampled_idx, :], matches_b[sampled_idx, :])
        in_a = []
        in_b = []
        update = 0
        for i in range(matches_num):
            matches_aa = np.append(matches_a[i,:],1)
            matches_bb = np.append(matches_b[i,:],1)
            error = np.dot(matches_aa, F.T)
            error = np.dot(error, matches_bb.T)
            if abs(error) < 0.05:
                in_a.append(matches_a[i,:])
                in_b.append(matches_b[i,:])
                update +=1

        if update > Best_count:
            Best_count = update
            best_F = F
            inliers_a = in_a
            inliers_b = in_b

    inliers_a = np.array(inliers_a)
    inliers_b = np.array(inliers_b)

    return best_F, inliers_a, inliers_b
