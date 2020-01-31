import os
import scipy.io
import h5py
import numpy as np
from scipy.sparse import csr_matrix

def load_data(datapath, dataset_id):

    #load pairwise distance matrix
    f = h5py.File(os.path.join(datapath,"adjacency_matrix_Tosca%s.mat" % dataset_id), 'r')
    data = np.array(f['A']['data'])
    data = np.reshape(data,(-1,1))
    ir = np.array(f['A']['ir'])
    ir = np.reshape(ir,(-1,1))
    jc = np.array(f['A']['jc'])
    jc = np.reshape(jc, (-1,1))

    sec = []
    count = 0
    for row in range(0,len(data)):
        if row%4==0 and row!=0:
            count+=1
        sec.append([count])
    sec = np.reshape(sec, (-1,1))
    
    A = np.zeros((len(jc),len(jc)))
    for row in range(0,len(ir)):
        A[int(ir[row]),int(sec[row])]=data[row] #create adjacency matrix
        A[int(sec[row]),int(ir[row])]=data[row] #make symmetric
    

    #load symmetry mapping
    tri2opposite  = scipy.io.loadmat(os.path.join(datapath,"tri2opposite_Tosca%s.mat" % dataset_id), squeeze_me=True, struct_as_record=False)
    left_tris = tri2opposite.get("tri2opposite").left_tris[:]
    left_tris -= 1 #-1 for 0 indexing

    #load mesh-face point clouds. Each mesh face is represented as a single 3D
    #point -- the mean of its vertices.
    pcf = scipy.io.loadmat(os.path.join(datapath,"PointCloudFeatures_Tosca%s.mat" % dataset_id),squeeze_me=True, struct_as_record=False)
    pcf = pcf.get("pcf")
    
    return (A,left_tris,pcf)