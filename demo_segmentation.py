import os
import pickle as pickle
import random
import sys
sys.path.append('.')
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from Inference.BookKeeper import *
from Inference.SamplerState import *
from Models.ddCRP import *
from Models.MNIWlik import *
from Util.load_data import *
from Util.package_data import *
from Util.trace_plot import *
from Util.visualize_result import *


def demo_segmentation():
    # Demo for mesh segmentation
    # Reads from disk (./Data) meshes preprocessed into point clouds.
    # Writes to disk (./Results) the resulting segmentation.

    """ SET PARAMETERS 
    1) number of sampler iterations. Higher the better.
    Recommended setting > 150. Reasonable segmentations can often
    be achieved within the first 30 iterations. While these premature samples
    do not correspond to samples from the posterior distribution, they might 
    be sufficient to produce reasonable results. The segmentation quality
    gradually increases with increasing number of iterations."""

    NUMITER = 10
    
    """ SELECT DATASET TO ANALYZE
    2) We include two sets of meshes from the TOSCA dataset (Centaur and Horse).
    Each dataset consists of a single synthetic mesh in several different poses."""

    DATASET = 'Centaur' #could also select 'Horse'
    
    """ SET NOISE PARAMETER
    3) Noise parameter, this controls the amount of deviation from the
    predicted affine transformation.
    Noise --- HIGHER values will result in LARGER segments.""" 
    
    NOISESCALE = 9e-04

    savedir = './Results'
    if(not os.path.isdir(savedir)):
        os.mkdir(savedir)

    datapath = './Data'

    # LOAD DATA

    #load Mesh face neighbors, symmetric mapping along the midsagittal plane , 
    #and mesh faces as point clouds.  
    [A,left_tris,pcf] = load_data(datapath,DATASET)
    left_tris -= 1 
    A = np.take(A,left_tris, axis=0) #-1 for 0 indexing
    A = np.take(A, left_tris, axis=1)
   
    # create a ddCRP object.
    ddcrp = ddCRP(float(A[0,2]), A.astype('float64'))
    
    X = []
    for i in range(0,1):
        # reference coords.
        X = np.insert(pcf.X,3, 1, axis=1)

    Y = pcf.Y # these are the coordinates of triangles in different body poses.

    # visualize the reference mesh.
    plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter(pcf.X[:,0],pcf.X[:,1],pcf.X[:,2], 'b') 
    plt.title('Reference mesh visualized as a point cloud') 
    plt.show()

    # view(94,10) axis equal 
    # time.sleep(3.0) 
    # plt.close('all')

    data = package_data(X,Y,1) 

    # load precomputed ratio of gammas to speed up likelihood computation.
    rofg = scipy.io.loadmat(os.path.join(datapath,'Ratio_of_Gammas.mat'))
    rofg = np.asarray(rofg.get("Ratio_of_Gammas")[0][0][0])


    #############Set hyper-parameters governing the affine likelihoods
    #(See http://cs.brown.edu/~sghosh/papers/GhoshSudderthLoperBlack12NIPS.pdf for details)######## 

    # Expected variance
    S0 = NOISESCALE*np.identity(3) 
    n0 = 5 

    # mean transformation
    M = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    #PRECISON matrix. Higher = smaller var.
    K = NOISESCALE*np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0.1]])

    # Matrix Normal Inverse Wishart likelihood object
    mniw = MNIWlik(NOISESCALE,S0,n0,M,K,rofg) 

    ###########################################################################

    ######## Initialize and Run Sampler########################################
    print('Gibbs Sampler will be run for %d iterations.\n' % NUMITER) 

    # Create sampler state object
    sampler_state = SamplerState(data,NUMITER) 
    # Create bookkeeping object
    bookkeeper = BookKeeper(sampler_state,savedir) 
    # Run Sampler
    converged_sampler_state=sampler_state.run_sampler(ddcrp,mniw,data,bookkeeper,NUMITER) 
    # save results
    pickle.dump(converged_sampler_state, open(os.path.join(savedir,'MCMC_Chain.pkl'),"wb"))
    ###########################################################################

    ######################## Diagnostics ######################################
    # joint lik trace
    ll_c = np.full(NUMITER, np.nan) 
    ll = np.full(NUMITER, np.nan)
    prior = np.full(NUMITER, np.nan)
    for i in range(0,NUMITER):
       (ll_c[i], ll[i], prior[i]) = trace_plot(i,mniw,data,converged_sampler_state,ddcrp.d2_mat) 

    plt.figure()
    plt.plot(ll_c, 'r-')
    plt.title('joint log-lik')
    plt.show()
    MAP_SAMPLE = np.argmax(ll_c) 

    # visualize result 
    visualize_result(2,converged_sampler_state,MAP_SAMPLE,left_tris,DATASET) 
    #save(os.path.join(savedir,'State.mat')) #saves all workspace variables
    ###########################################################################

if __name__=="__main__":
    demo_segmentation()
