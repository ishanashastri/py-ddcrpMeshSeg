import numpy as np

def package_data(X,Y,numSubj):
    # package data for later processing // data stored as a dictionary 
    # numSubj - number of subjects to be processed (For the datasets included
    # with this package, this is 1. If you have scans of multiple subjects set
    # this parameter appropriately. It could take any value between 1 and
    # length(data.X)).
    
    # massage data
    data = dict.fromkeys(['X','Y', 'num_bodies'])

    data["X"] = X 
 
    # transpose Y
    tY = np.transpose(Y,(1,0,2))
    #for jj in range(0,len(Y[ii][0])):
    #    print(Y[:,:,jj].transpose())
    data["Y"] = tY

    # number of distinct mesh subjects per partition
    # each subject has her own reference mesh
    data["num_bodies"] = numSubj #len(data["X"]) 

    return data



     

