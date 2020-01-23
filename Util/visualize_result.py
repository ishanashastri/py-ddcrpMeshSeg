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

    ### randpermute the order of labels to make colormap more meaningful.
    randt = np.zeros((1,len(param.t[sa_m,:])))
    mc1 = np.random.permutation(len(np.unique(param.t[sa_m,:])))
    for ii in range(0, len(np.unique(param.t[sa_m,:]))):
        randt[0,param.t[sa_m,:]==ii] = mc1[ii]
    
    labels = np.zeros((1,np.shape(left_tris)[0]*2))
    labels[0, left_tris] = randt
    # optionally enforce symmetry by projecting the right mesh segentation on the left.
    #  right_tris = setdiff(1:size(left_tris,1)*2,left_tris)
    #  labels(right_tris) = randt

    if(dataset=="Centaur"):
        fnames = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(meshpath, 'meshes', 'cen*.mat')))]
    else:
        fnames = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(meshpath, 'meshes', 'hor*.mat')))]
    
    data = scipy.io.loadmat(os.path.join(meshpath,'meshes',fnames[0]),squeeze_me=True, struct_as_record=False)
    
    surface = data['surface']
    vrts=np.zeros((len(surface.X),3))
    vrts[:,0] = surface.X
    vrts[:,1] = surface.Y
    vrts[:,2] = surface.Z
    faces = surface.TRIV-1

    fig = plt.figure(f)
    triang = mtri.Triangulation(vrts[:,0], vrts[:,2], faces)
    ax = fig.gca(projection='3d')
    pl = ax.plot_trisurf(triang, vrts[:,1], lw=0.2, cmap=plt.get_cmap("nipy_spectral"), edgecolor="none", alpha=0.5) 
    pl.set_array(np.squeeze(labels))
    pl.autoscale()
    plt.axis('off')
    plt.show()
   
    #light = LightSource(90, 45)
    #illuminated_surface = light.shade(Z, cmap=cm.coolwarm)  
    """
    light('Position',[5 0 5],'Style','infinite', 'Color', [.25 .25 .25])
    light('Position',[0 5 5],'Style','infinite', 'Color', [.25 .25 .25])
    light('Position',[-5 -5 5],'Style','infinite', 'Color', [.25 .25 .25])
    """
    
    #set(gcf, 'Color', [1, 1, 1])
    #view(-180,-80)

    return (labels,ax)
