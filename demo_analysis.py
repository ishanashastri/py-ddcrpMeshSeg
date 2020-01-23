import glob
import os
import pickle as pickle

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io

from Util.load_data import *
from Analysis.measure_nonrigidity import *


def demo_analysis(DATASET):
    # Measures and visualizes non rigid regions given a segmentaiton.
    # We will load a precomupted_MCMC_sample and extract the segmentation 
    # from it. We will then measure non-rigidity exhibited by this
    # segmentation.
    
    # Reads from disk (./Data and ./Results) meshes preprocessed into 
    # point clouds as well as precomputed MCMC samples.
    
    # I/P: DATASET - 'Centaur' or 'Horse'
    # USAGE : demo_analysis('Centaur')
    
    
    datapath = './Data'
    savedir = './Results'
    [idk,left_tris,pcf] = load_data(datapath,DATASET)
    tri2opposite  = scipy.io.loadmat(os.path.join(datapath,"tri2opposite_Tosca%s.mat" % DATASET), squeeze_me=True, struct_as_record=False)
    #tri2opposite = tri2opposite["tri2opposite"][0][0][1] #ndarray
    left_tris = tri2opposite.get("tri2opposite").left_tris[:]
    left_tris -= 1 
    left_tris = np.reshape(left_tris, (-1,1))

    
    # load precomputed labels
    converged_sampler_state = scipy.io.loadmat(os.path.join(savedir,('precomputed_converged_MCMC_sample_%s.mat'%DATASET)),squeeze_me=True, struct_as_record=False)
    labels = converged_sampler_state.get("converged_sampler_state").t
   
    # Measure non rigidity
    [unk,average_tri_err] = measure_nonrigidity(pcf,labels)
    
   #######################################################################
    # visualize non rigidity ###############
    labels= np.zeros((1,np.shape(left_tris)[0]*2))
    labels[0,np.transpose(left_tris,(1,0))] = average_tri_err
    # optionally enforce symmetry by projecting the right mesh segentation on the left.
    right_tris = np.arange(0,np.shape(left_tris)[0]*2)
    right_tris = np.setdiff1d(right_tris,left_tris)
    labels[0,right_tris] = np.squeeze(average_tri_err)
    
    if(DATASET=="Centaur"):
        fnames = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(datapath, 'meshes', 'cen*.mat')))]
    else:
        fnames = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(datapath, 'meshes', 'hor*.mat')))]
    
    data = scipy.io.loadmat(os.path.join(datapath,'meshes',fnames[0]),squeeze_me=True, struct_as_record=False)
    
    surface = data['surface']
    vrts=np.zeros((len(surface.X),3))
    vrts[:,0] = surface.X
    vrts[:,1] = surface.Y
    vrts[:,2] = surface.Z
    faces = surface.TRIV-1

    fig = plt.figure()
    triang = mtri.Triangulation(vrts[:,0], vrts[:,2], faces)
    ax = fig.gca(projection='3d')
    pl = ax.plot_trisurf(triang, vrts[:,1], lw=0.2, cmap=plt.get_cmap("nipy_spectral"), edgecolor="none", alpha=0.5) 
    pl.set_array(np.squeeze(labels))
    pl.autoscale()
    plt.axis('off')
    plt.title('Non rigidity visualization. Red/Yellow = high nonrigidity')
    plt.show()

    """set(t,'FaceLighting','phong','AmbientStrength',0.5)
    light('Position',[5 0 5],'Style','infinite', 'Color', [.25 .25 .25])
    light('Position',[0 5 5],'Style','infinite', 'Color', [.25 .25 .25])
    light('Position',[-5 -5 5],'Style','infinite', 'Color', [.25 .25 .25])
    """
    
    non_rigidity = labels
    pickle.dump(non_rigidity, open(os.path.join(savedir,'PerPart_Nonrigidity.pkl'),"wb")) 


if __name__=="__main__":
    demo_analysis("Centaur")