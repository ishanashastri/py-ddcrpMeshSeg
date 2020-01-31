import numpy as np 

def measure_nonrigidity(test_set,labels):
    test_pose = test_set.Y
    reference_pose = test_set.X
    error = np.inf*np.ones((1,1))
    per_trierr = np.zeros((len(labels),1))
    [error[0],per_trierr[:,0]] = evaluate_segmentation_perpose(test_pose[:,:,0],reference_pose,labels)   
    tri_err = np.sum(per_trierr,axis=1) 
    return (error,tri_err)


def evaluate_segmentation_perpose(test_pose,reference_pose,labels):
# test_pose, reference_pose are N*3 matrices, where N is the number of
# triangles constituiting the mesh. labels is N*1.
  
    num_parts = len(np.unique(labels))
    part_score = np.zeros((1,num_parts))
    pertri_err = np.full((1,len(labels)),np.nan)
    design = np.asarray([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] ,
              [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0] ,
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
    design = np.reshape(design,(3,12))
    ctr = 0    
    for k in np.unique(labels):
        # select the triangles belonging to part k
        ind = labels==k
        zeropadx = np.zeros((3*len(np.nonzero(ind>0)[0]),12))  
        y = test_pose[ind,:]
        x = reference_pose[ind,:]
        # stack y row wise
        y = y.transpose()
        y= y.flatten(1)
        # construct zero padded x matrix
        for i in range(0,len(x)):
            x_row = np.append(x[i,:], 1)
            zeropadx[np.arange(3*i,3*i+3),:] = np.kron(np.ones((3,3)), x_row)*design            

        # least squares minimization
        a = np.linalg.lstsq(zeropadx,y)[0]
        # compute residuals
        residual = np.dot(zeropadx,a)-y
        part_score[0,ctr] = np.linalg.norm(residual,2)
        # compute per face residuals 
        tri_ctr = 0
        for tri in np.nonzero(ind)[0]:
            pertri_err[0,tri] = np.linalg.norm(residual[(tri_ctr)*3+np.arange(0,3)],2)
            tri_ctr += 1
        ctr += 1
    
    # sum over all parts.
    score = np.sum(part_score)

    return (score,pertri_err)