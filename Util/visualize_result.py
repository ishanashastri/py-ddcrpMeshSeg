import glob
import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import scipy.io
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualize_result(f,c_MCMC_param,sa_m,left_tris,dataset):
    meshpath = './Data'
    param = c_MCMC_param

    # randpermute the order of labels to make colormap more meaningful.
    randt = np.zeros((1,len(param.t[sa_m,:])))
    mc1 = np.random.permutation(len(np.unique(param.t[sa_m,:])))
    for ii in range(0, len(np.unique(param.t[sa_m,:]))):
        randt[0,param.t[sa_m,:]==ii] = mc1[ii]
    
    labels = np.zeros((1,np.shape(left_tris)[0]*2))
    labels[0, left_tris] = randt

    # optionally enforce symmetry by projecting the right mesh segentation on the left.
    right_tris = np.setdiff1d(np.arange(0,np.shape(left_tris)[0]*2),left_tris)
    labels[0,right_tris] = randt

    if(dataset=="Centaur"):
        fnames = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(meshpath, 'meshes', 'cen*.mat')))]
    else:
        fnames = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(meshpath, 'meshes', 'hor*.mat')))]
    
    for i, name in enumerate(fnames): #show all meshes

        data = scipy.io.loadmat(os.path.join(meshpath,'meshes',name),squeeze_me=True, struct_as_record=False)
        
        surface = data['surface']
        vrts=np.zeros((len(surface.X),3))
        vrts[:,0] = surface.X
        vrts[:,1] = surface.Y
        vrts[:,2] = surface.Z
        faces = surface.TRIV-1

        fig = plt.figure()
        triang = mtri.Triangulation(vrts[:,0], vrts[:,1], faces)
        ax = fig.gca(projection='3d')
        pl = ax.plot_trisurf(triang, vrts[:,2], lw=0.2, cmap=plt.get_cmap("nipy_spectral"), edgecolor="none", alpha=.6)
        pl.set_array(np.squeeze(labels))
        pl.autoscale()
        plt.axis('off')

        """ Generating rotating gifs
        for ii in range(0,360,5): 
            ax.view_init(10,100-ii)
            plt.draw()
            plt.savefig('movie%d_%d.png'%(i,ii), transparent=True)
        """
        
        #Display graphs
        plt.pause(5) 
        plt.savefig('segmentation_%d.png' %i,transparent=True)
        plt.close()

    return (labels,ax)
