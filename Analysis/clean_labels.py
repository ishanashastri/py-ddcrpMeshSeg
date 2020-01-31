import numpy as np
from Inference.SamplerState import *

def clean_labels(A,mniw,param,data,left_tris):
### hill climb on the map sample to clean up boundary artifacts ###

    #find faces at boundaries
    [conflict_faces,conflict_segs] = find_conflict(param,A)

    # visualize faces at boundaries
    p.t = conflict_faces.transpose()
    #visualize_result(2,p,1,left_tris)


    # compute current table liks
    table_lik = np.zeros((1,param.T))
    for t in range(0,param.T):
        table_members = param.t==t
        table_lik[0,t] = compute_table_lik(mniw,data,table_members)

    # go through the conflict faces and check if moving them improves the
    # likelihood
    update = 1
    lapctr = 1
    while(update):
        ind = np.nonzero(conflict_faces)
        # randomly order conflict traingles.
        ind = ind[np.random.permutation(len(ind))]
        ctr = 0
        iter_update = 0
        
        for i in ind.tranpose():
            if(ctr==100):
                print('\nCleaning Labels iteration %d \n' % lapctr)
            
            ctr += 1
            # find the likelihood of the table current face is sitting at
            curr_lik = table_lik[param.t[0,i]]
            # potential tables to move the current face to are in conflict_segs
            potential_tabs = conflict_segs
            new_lik = np.zeros((1,len(potential_tabs)))
            for j in range(0,len(potential_tabs)):
                table_members = param.t==potential_tabs[j]
                table_members[0,i] = 1 # move the current face to table j
                new_lik[0,j] = 2*compute_table_lik(mniw,data,table_members)
            
            [v,max_ind] = np.maximum(new_lik)
            # If moving the face helps assign it to the new table
            if(v>curr_lik):
                param.t[0,i] = potential_tabs[max_ind]
                #update stored table_likelihood
                table_lik[param.t[0,i]] = v
                
                # visualize the change
                # visualize_result(3,param,1,left_tris)
                iter_update = 1
            
        
        lapctr += 1
        # if no updates happened reset update and exit
        #param = clean_surrounded(param,A)

        #break
        if(iter_update==0):
            update = 0
            break
    
    return param



def clean_surrounded(param,A):
# return set of faces surrounded by faces from other segments

    # go through all faces
    for i in range(0, param.num_data):
        #nbors = A(i,:)
        curr_seg = param.t[i]
        #nbor_seg = setdiff(param.t(logical(A(i,:))),curr_seg)
        nbor_segs = param.t(logical(A[i,:]))
        
        if(isempty(intersect(nbor_segs,curr_seg))):
            param.t(i) = nbor_segs(1)
        
    return param
    
    


def find_conflict(param,A):
# return set of faces that border faces from other segments
    conflict_faces = np.zeros((param.num_data,1))
    nbors = cell(1,param.num_data)
    # go through all faces
    for i = 1:param.num_data
        #nbors = A(i,:)
        curr_seg = param.t(i)
        #nbor_seg = setdiff(param.t(logical(A(i,:))),curr_seg)
        nbor_segs = param.t[logical(A[i,:])]
        nbor_seg = np.unique(nbor_segs)#nbor _segs(logical(nbor_segs - curr_seg))
        if(len(nbor_seg)>1):
            conflict_faces[i] = 1
            nbors = np.setdiff1d(nbor_seg,curr_seg)
        
    return (conflict_faces,nbors)
    

