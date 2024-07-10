import pandas as pd

import numpy as np
#import seaborn as sns
import os
import math
import json
import argparse

from scipy import stats
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import homogeneity_score
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import RangeSlider

#import plotly.express as px

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def setAxLim2BBox(ax,BBox,yup=True):
    """Sets axis limits to match bounding box.

    Parameters:
    --------
    pathDir -- directory path string    

    Usage examples:
    >>> BBox = {'x0':-1.,'x1':+1.}
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot()
    >>> setAxLim2BBox(ax,BBox)
    >>> ax.get_xlim()
    (-1.0, 1.0)
    """
    if yup:
        if 'x0' in BBox: ax.set_xlim((BBox['x0'],BBox['x1']))
        if 'y0' in BBox: ax.set_zlim((BBox['y0'],BBox['y1']))
        if 'z0' in BBox:
            if hasattr(ax, 'get_zlim'):
                ax.set_ylim((BBox['z0'],BBox['z1']))
    else:
        if 'x0' in BBox: ax.set_xlim((BBox['x0'],BBox['x1']))
        if 'y0' in BBox: ax.set_ylim((BBox['y0'],BBox['y1']))
        if 'z0' in BBox: 
            if hasattr(ax, 'get_zlim'):
                ax.set_zlim((BBox['z0'],BBox['z1']))       

def makeBinsEdges(BBox,width=1,yup=True):
    """Makes bin edges.

    Parameters:
    --------
    BBox -- dictionary with bins grid edges
    width -- bins width 

    Return variables:
    --------
    xedges --  ndarray 
        Array of bin edges
    yedges --  ndarray 
        Array of bin edges
    zedges -- ndarray 
        Array of bin edges
    
    Examples:
    --------
    >>> BBox = {'x0':-2.,'x1':+2.,'y0':-2.,'y1':+2.}
    >>> xedges, yedges = makeBinsEdges(BBox,width=1)
    >>> xedges
    array([-2., -1.,  0.,  1.,  2.])
    >>> yedges
    array([-2., -1.,  0.,  1.,  2.])

    >>> BBox = {'x0':-3.,'x1':+3.,'y0':-2.,'y1':+2.,'z0':-1.,'z1':+1.}
    >>> xedges, yedges, zedges = makeBinsEdges(BBox,width=1)
    >>> xedges
    array([-3., -2., -1.,  0.,  1.,  2.,  3.])
    >>> yedges
    array([-2., -1.,  0.,  1.,  2.])
    >>> zedges
    array([-1.,  0.,  1.])
    """
    rows = BBox['y1']-BBox['y0']+1
    cols = BBox['x1']-BBox['x0']+1

    xedges = np.arange(BBox['x0'],+BBox['x1']+1,width)
    yedges = np.arange(BBox['y0'],+BBox['y1']+1,width)
    
    if 'z1' in BBox:
        stacks = BBox['z1']-BBox['z0']+1 
        zedges = np.arange(BBox['z0'],+BBox['z1']+1,width)
        return xedges, yedges, zedges 

    return xedges, yedges

def drawPath(path,dpath=None,BBox=None,ax=None,yup=True,colorbar=False,pointSize=1):
    """draw path 2D and 3D

    Parameters:
    --------
    path -- array 
    width -- bins width   
    
    Examples:
    --------
    >>> path = np.random.rand(10,4)
    >>> ax,sc = drawPath(path,dpath=None,BBox=None,ax=None)
    >>> cbar = plt.colorbar(sc, ax=ax)
    >>> 'Axes3D' in str(type(ax))
    True
    >>> ax.lines
    <Axes.ArtistList of 1 lines>
    """
    if ax == None:
        fig = plt.figure()
        if path.shape[1] == 4:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()
    ax.set_xlabel('x')

    if path.shape[1] == 4:
        t,x,y,z = path.T
        if hasattr(dpath, "__len__"): 
            qt,u,v,w = dpath.T
        if yup:
            #print('yup')
            ax.set_ylabel('z')
            ax.set_zlabel('y')
            ax.plot(x,z,y)
            sc = ax.scatter(x,z,y,c=t,s=pointSize)
            if hasattr(dpath, "__len__"):
                ax.quiver(x,z,y,u,w,v,color='gray',alpha=0.2)
                sc = ax.scatter(x+u,z+w,y+v,c=t,s=pointSize)
        else:
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.plot(x,y,z)
            sc = ax.scatter(x,y,z,c=t,s=pointSize)
            if hasattr(dpath, "__len__"):
                ax.quiver(x,y,z,u,v,w,color='gray',alpha=0.2)
                sc = ax.scatter(x+u,y+v,z+w,c=t,s=pointSize)
        if colorbar: plt.colorbar(sc, ax=ax)
    if path.shape[1] == 3:
        ax.set_ylabel('y')
        t,x,y = path.T
        ax.plot(x,y)
        sc = ax.scatter(x,y,c=t,s=pointSize)
        if colorbar: plt.colorbar(sc, ax=ax)
        if hasattr(dpath, "__len__"):
            qt,u,v = dpath.T
            ax.quiver(x,y,u,v,color='gray',alpha=0.2)       
    
    if isinstance(BBox , dict):
        setAxLim2BBox(ax,BBox,yup=yup)
    return ax,sc
        

def drawPath2DT(path,dpath=None,BBox=None,ax=None,yup=True,scale=None,colorbar=False,pointSize=1):
    if path.shape[1] == 4:
        t,x,y,z = path.T
        if hasattr(dpath, "__len__"): 
            qt,u,v,w = dpath.T
    if path.shape[1] == 3:
        t,x,y = path.T
        if hasattr(dpath, "__len__"): 
            qt,u,v = dpath.T    
    if ax == None:
        fig = plt.figure()
        ax = plt.figure().add_subplot()
    if yup==True:
        ax.plot(x,z)
        sc = ax.scatter(x,z,c=t,s=pointSize)
        if hasattr(dpath, "__len__"):  #not u==None:
            ax.quiver(x,z,u,w,color='k',alpha=0.2,angles='xy', scale_units='xy', scale=.5)
        ax.set_xlim((BBox['x0'],BBox['x1']))
        ax.set_ylim((BBox['z0'],BBox['z1']))
        ax.set_aspect('equal', adjustable='box')
        if colorbar: plt.colorbar(sc, ax=ax)
    else:
        ax.plot(x,y)
        sc = ax.scatter(x,y,c=t,s=pointSize)
        if hasattr(dpath, "__len__"):  #not u==None:
            ax.quiver(x,y,u,v,color='k',alpha=0.2,angles='xy', scale_units='xy', scale=.5)
        ax.set_xlim((BBox['x0'],BBox['x1']))
        ax.set_ylim((BBox['y0'],BBox['y1']))
        ax.set_aspect('equal', adjustable='box')
        if colorbar: plt.colorbar(sc, ax=ax)
        #if isinstance(BBox , dict):
        #    setAxLim2BBox(ax,BBox,yup=False)

def allPaths2D(paths,largerThan=0,ax=None,yup=True):
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot()
    for path in paths:
        #print(path)
        if len(path) > largerThan:
            t,x,y,z = path.T
            if yup:
                ax.scatter(x,z,s=1)
                ax.set_xlabel('x')
                ax.set_ylabel('z')
            else:
                ax.scatter(x,y,s=1)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
        
#def allPaths3D(xs,ys,zs,largerThan=0,ax=None):
#    if ax == None:
#        fig = plt.figure()
#        ax = fig.add_subplot(projection='3d')
#    for x,y,z in zip(xs,ys,zs):
#        if len(x) > largerThan:
#            ax.scatter(x,y,z,s=1)
#    ax.set_xlabel('x')
#    ax.set_ylabel('y')
#    ax.set_zlabel('z')
    
def allPaths3D(paths,largerThan=0,ax=None,yup=True):
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    if yup:
        for path in paths:
            if len(path) > largerThan:
                t,x,y,z = path.T
                ax.scatter(x,z,y,s=4)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
    else:
        for path in paths:
            if len(path) > largerThan:
                t,x,y,z = path.T
                ax.scatter(x,y,z,s=4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


def plotKDE(x,y,z,density,yup=True,ax=None):
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()
    if yup:
        sc = ax.scatter(x,z,y,c=density,s=1)
        fig.colorbar(sc, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
    else:
        sc = ax.scatter(x,y,z,c=density,s=1)
        fig.colorbar(sc, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

def occupancy2D(x,y,xedges,yedges):
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T#[::-1,:]
    return H,xedges,yedges
    
def occupancy3D(path,BBox,width=1):
    xedges, yedges,zedges = makeBinsEdges(BBox,width)
    t,x,y,z = path.T
    H, edges = np.histogramdd((x, y, z), bins=(xedges, yedges,zedges))
    #H = H.T#[::-1,:]
    #print('hist dd',H.shape, edges[0].size, edges[1].size, edges[2].size)
    return H,edges

def drawMarginals(H,path,edges,pointsSize=2,yup=True):
    xedges, yedges, zedges  = edges#makeBinsEdges(BBox,width=1.)
    t,x,y,z = path.T
    f,ax = plt.subplots(1,3)
    ax[0].hist(x,bins=xedges)
    ax[0].set_xlabel('x')
    ax[1].hist(y,bins=yedges)
    ax[1].set_xlabel('y')
    ax[2].hist(z,bins=zedges)
    ax[2].set_xlabel('z')
    f,ax=plt.subplots(1)
    ax.set_aspect('equal', adjustable='box')
    if not yup:
        X, Y = np.meshgrid(xedges, yedges, indexing='xy')
        ax.pcolormesh(X, Y, np.sum(H,axis=2).T)
        ax.scatter(x,y,s=pointsSize)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    else:
        X, Z = np.meshgrid(xedges, zedges, indexing='xy')
        ax.pcolormesh(X, Z, np.sum(H,axis=1).T)
        ax.scatter(x,z,s=pointsSize)
        ax.set_xlabel('x')
        ax.set_ylabel('z')


def displayH3Dstack(im3d,path,edges, step=1,yup=True):
    #data_montage = util.montage(im3d[::step], padding_width=4, fill=np.nan)
    xedges, yedges,zedges = edges#makeBinsEdges(BBox,width=1)
    #print('yedges',yedges)
    t,x,y,z = path.T
    if not yup:
        deep = im3d.shape[-1]
        _, ax = plt.subplots( math.ceil(deep/3),3,figsize=(10, 10))
        for i in range(deep):
            axrows = math.ceil(deep/3)
            r,c = int(i%axrows),int(i/axrows)
            #print(r,c)
            ax[r,c].set_aspect('equal', adjustable='box')
            X, Y = np.meshgrid(xedges,yedges, indexing='xy')
            #print(X.shape,Y.shape,im3d[:,:,i].shape)
            #ax[i].imshow(im3d[:,:,i])
            zbool = (z>zedge[i]) * (z<zedge[i+1]) 
            #print('zbool',np.sum(zbool))
            #print('x[zbool]',x[zbool])
            ax[r,c].pcolormesh(X, Y, im3d[:,:,i].T)
            ax[r,c].scatter(x[zbool],y[zbool],s=1)
            Xc, Yc = np.meshgrid((xedges[1:]+xedges[:-1])*0.5,(yedges[1:]+yedges[:-1])*0.5, indexing='xy')
            ax[r,c].scatter(Xc.flatten(),Yc.flatten(),s=1)
            ax[r,c].set_xlabel('x')
            ax[r,c].set_ylabel('y')
            #ax[i].pcolormesh(X = X, Y = Y, C = im3d[:,:,i].T)
            #ax[i,j].imshow(im3d[:,:,i*4+j], cmap=cmap)
    else:
        deep = im3d.shape[1]
        _, ax = plt.subplots( math.ceil(deep/3),3,figsize=(10, 10))
        for i in range(deep):
            axrows = math.ceil(deep/3)
            r,c = int(i%axrows),int(i/axrows)
            ax[r,c].set_aspect('equal', adjustable='box')
            X, Z = np.meshgrid(xedges,zedges, indexing='xy')
            ybool = (y>yedges[i]) * (y<yedges[i+1]) 
            ax[r,c].pcolormesh(X, Z, im3d[:,i,:].T)
            ax[r,c].scatter(x[ybool],z[ybool],s=1)
            ax[r,c].set_xlabel('x')
            ax[r,c].set_ylabel('z')
    #ax.set_axis_off()
    

def crateLinkageMatrix(model):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix
    
def plot_dendrogram(model, nclusters=2, **kwargs):
    # function copied from sklearn:
    # plot_agglomerative_dendrogram.html
    #
    # Create linkage matrix and then plot the dendrogram

    linkage_matrix = crateLinkageMatrix(model)

    # Plot the corresponding dendrogram
    dg =  dendrogram(linkage_matrix,
               color_threshold=sorted(model.distances_)[-nclusters+1], **kwargs)
    return dg

def showH(H,edges,path=None, ax=None,yup=True,markerSize=1):
    xedges, yedges,zedges = edges#makeBinsEdges(BBox,width=1)
    Xc, Yc, Zc = np.meshgrid((xedges[1:]+xedges[:-1])*0.5,(yedges[1:]+yedges[:-1])*0.5, (zedges[1:]+zedges[:-1])*0.5, indexing='xy')
    xc, yc, zc = Xc.flatten(), Yc.flatten(), Zc.flatten() 
    if not yup:
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim((xedges[0],xedges[-1]))
            ax.set_ylim((yedges[0],yedges[-1]))
            ax.set_zlim((zedges[0],zedges[-1]))
        Hswap = np.swapaxes(H,0,1)
        Hflat = Hswap.flatten()
        sc = ax.scatter(xc[Hflat>0], yc[Hflat>0], zc[Hflat>0],c=Hflat[Hflat>0],s=markerSize)
        if hasattr(path, "__len__"):
            t,x,y,z = path.T
            ax.plot(x,y,z)
    else:
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_zlabel('y')
            ax.set_xlim((xedges[0],xedges[-1]))
            ax.set_ylim((zedges[0],zedges[-1]))
            ax.set_zlim((yedges[0],yedges[-1]))
        Hswap = np.swapaxes(H,0,1)
        Hflat = Hswap.flatten()
        sc = ax.scatter(xc[Hflat>0], zc[Hflat>0], yc[Hflat>0],c=Hflat[Hflat>0],s=markerSize)
        if hasattr(path, "__len__"):
            t,x,y,z = path.T
            ax.plot(x,z,y)
        
def coOccupancy(Hs):
    nTr = len(Hs)
    corrOcc = np.zeros((nTr,nTr))

    for i in range(nTr):
        for j in range(nTr):
            vari=np.sum(Hs[i]**2)
            varj=np.sum(Hs[j]**2)
            corrOcc[i,j] = np.sum(Hs[i]*Hs[j])/np.sqrt(vari*varj)

    return corrOcc

def occupancyEuclDist(Hs):
    nTr = len(Hs)
    eucl = np.zeros((nTr,nTr))

    for i in range(nTr):
        for j in range(nTr):
            #vari=np.sum(Hs[i]**2)
            #varj=np.sum(Hs[j]**2)
            #corrOcc[i,j] = np.sum(Hs[i]*Hs[j])/np.sqrt(vari*varj)
            eucl[i,j] = np.sqrt(np.sum( (Hs[i] - Hs[j])**2 ))

    return eucl

def hierarchical_dendrogram(dist_mat):
    # Hierarcical cluster con sklearn
    model = AgglomerativeClustering(compute_full_tree=True,
                                    compute_distances=True,
                                    n_clusters=3, metric="precomputed",
                                    linkage="complete")

    cluster = model.fit_predict(dist_mat)

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    dg = plot_dendrogram(model, orientation='left', ax=axs)
    return dg

def plotSorted(matrix,sortedIndxs,ax=None):
    matrixSort = matrix[:, sortedIndxs][sortedIndxs]

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot()

    ax.imshow(matrixSort)
    ax.set_xticks(range(0,len(matrixSort)))
    ax.set_yticks(range(0,len(matrixSort)))
    ax.set_xticklabels(sortedIndxs)
    ax.set_yticklabels(sortedIndxs)

# Read single session csv in folder
def readSessionData(pathDir,sessionFile):
    dfU = pd.read_csv(pathDir+'/'+sessionFile+'.csv',index_col=False)
    #print('columns',dfU.columns)
    dfU.columns = dfU.columns.str.lstrip()
    return dfU

# Read data from csv in folder
def readData(pathDir):
    """Read data from csv in folder.

    Parameters:
    --------
    pathDir -- directory path string

    Examples:
    --------
    >>> ids, fileNames, dfUs, df = readData('csv-Test/')
    >>> ids
    [0, 1]
    >>> dfUs[0].index
    RangeIndex(start=0, stop=3, step=1)
    >>> 'time' in dfUs[0].columns
    True
    >>> df.index
    RangeIndex(start=0, stop=6, step=1)
    >>> 'time' in df.columns
    True
    >>> 'posx' in df.columns
    True
    >>> 'fx' in df.columns
    True
    >>> 'dirx' in df.columns
    True
    """
    dfUs = [] 
    ids = []
    fileNames = []

    listCSVs = [f for f in os.listdir(pathDir) if f.split('.')[1] == 'csv']

    for uId,f in enumerate(listCSVs):
        # add users ids
        ids.append(uId) 
        fileNames.append(f.split('.')[0])
        #print('uId', uId,'f',f)
        dfU = pd.read_csv(pathDir+'/'+f,index_col=False)
        #print('columns',dfU.columns)
        dfU.columns = dfU.columns.str.lstrip()
        #print('columns',dfU.columns)
        dfU['ID'] = uId
        dfU['filename'] =f.split('.')[0]
        dfUs.append(dfU)
    return ids, fileNames, dfUs, pd.concat(dfUs,ignore_index=True)

def readDataParsSession(pathDir,fileName):
    """Read json with sessions pars.

    Parameters:
    --------
    pathDir -- directory path string
    fileName -- session file name
    """
    sessionPars=None
    d = None
    file = pathDir+'/pars.json'
    #print('file',file,os.path.isfile(file))
    if os.path.isfile(file):
        with open(file) as f:
            d = json.load(f)
        #print('json')
        #print(d)
        if 'preprocessedVRsessions' in d:
            if fileName in d['preprocessedVRsessions']:
                sessionPars = d['preprocessedVRsessions'][fileName]
    #else:
    #    listCSVs = [f for f in os.listdir(pathDir) if f.split('.')[1] == 'csv']
    #    fileNames = [f.split('.')[0] for f in listCSVs]
    #    d = {'bbox':{},'sessions':[fname for fname in fileNames]}
    #    with open(file, 'w', encoding='utf-8') as f:
    #        json.dump(d, f, ensure_ascii=False, indent=4)        
    return sessionPars,d


def readDataPars(pathDir,fileNames):
    """Read json with sessions pars.

    Parameters:
    --------
    pathDir -- directory path string
    """
    file = pathDir+'-pars.json'
    if os.path.isfile(file):
        with open(file) as f:
            d = json.load(f)
            #print(d)
        for fname in fileNames:
            assert  fname in d['sessions'], 'error json file does not contain user'+fname+ 'file!!'
    else:
        d = {'bbox':{},'sessions':[fname for fname in fileNames]}
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=4)
    return d

def makeBBox(paths,dpaths,fpaths):
    """get lists paths,dpaths,fpaths and compute bounding box.

    Parameters:
    --------
    paths -- list of arrays 
    dpaths -- list of arrays
    fpaths -- list of arrays
    """
    #_,n = np.vstack(navs).T
    _,x,y,z = np.vstack(paths).T
    _,u,v,w = np.vstack(dpaths).T
    _,fx,fy,fz = np.vstack(fpaths).T
    #print('x',np.nanmin(x),np.nanmax(x))
    #print('u',np.nanmin(u),np.nanmax(u))
    if np.isnan(fx).all():
        x0 = np.nanmin([np.nanmin(x),np.nanmin(x+u)])
        x1 = np.nanmax([np.nanmax(x),np.nanmax(x+u)])
        y0 = np.nanmin([np.nanmin(y),np.nanmin(y+v)])
        y1 = np.nanmax([np.nanmax(y),np.nanmax(y+v)])
        z0 = np.nanmin([np.nanmin(z),np.nanmin(z+w)])
        z1 = np.nanmax([np.nanmax(z),np.nanmax(z+w)])
    else:
        x0 = np.nanmin([np.nanmin(x),np.nanmin(x+u),np.nanmin(fx)])
        x1 = np.nanmax([np.nanmax(x),np.nanmax(x+u),np.nanmax(fx)])
        y0 = np.nanmin([np.nanmin(y),np.nanmin(y+v),np.nanmin(fy)])
        y1 = np.nanmax([np.nanmax(y),np.nanmax(y+v),np.nanmax(fy)])
        z0 = np.nanmin([np.nanmin(z),np.nanmin(z+w),np.nanmin(fz)])
        z1 = np.nanmax([np.nanmax(z),np.nanmax(z+w),np.nanmax(fz)])
    #print('BBox',x0,x1,np.min(x+u),np.max(x+u))
    #print('ymin',np.min(y),np.min(y+v),np.nanmin(fy))
    BBox = {'x0':x0,'x1':x1,'y0':y0,'y1':y1,'z0':z0,'z1':z1}
    #plt.figure()
    #plt.plot(x+w)
    #plt.plot(x)
    #plt.show()
    return BBox


def getSesVars(path,dpath,fpath,nav):
    t,x,y,z = path.T
    t1,dx,dy,dz = dpath.T
    t2,fx,fy,fz = fpath.T
    t3,n = nav.T
    assert t[-1] == t1[-1], 'problems with times pt path and dpath' 
    assert t[-1] == t2[-1], 'problems with times pt path and fpath'
    assert t[-1] == t3[-1], 'problems with times pt path and nav'
    return t,x,y,z,dx,dy,dz,fx,fy,fz,n

def getVR(dfS):  
    """Gets VR state navigation modality from a session data frames.
    
    Parameters:
    --------
    dfS -- session dataframe

    Return variables:
    --------
    nav --  ndarray 
        Array with true values when navigation modality is VR
    """

    assert (('Time' in dfS.columns) or ('time' in dfS.columns))  , "neither Time or time in csv columns: "+dfU.columns
    timeCol = 'Time'
    if not 'Time' in dfS.columns:
        timeCol = 'time'    

    t =  dfS[timeCol].values
    nav = np.zeros((len(t),2))
    nav[:,0] = t
    #print(dfU['nav'] )
    boolVR = (dfS['nav'] == 'VR') 
    #print(boolVR )
    nav[:,1] = boolVR.values    
    return nav
    

def getAR(dfS):  
    """Gets AR state navigation modality from a session data frames.
    
    Parameters:
    --------
    dfS -- session dataframe

    Return variables:
    --------
    navs --  ndarray 
        Array with true values when navigation modality is AR
    """

    assert (('Time' in dfS.columns) or ('time' in dfS.columns))  , "neither Time or time in csv columns: "+dfU.columns
    timeCol = 'Time'
    if not 'Time' in dfS.columns:
        timeCol = 'time'    

    t =  dfS[timeCol].values
    nav = np.zeros((len(t),2))
    nav[:,0] = t
    #print(dfU['nav'] )
    boolVR = (dfS['nav'] == 'AR') 
    #print(boolVR )
    nav[:,1] = boolVR.values    
    return nav

def getVRs(ids,dfSs):
    """Gets a navigation modality from a list of session data frames.
    
    Parameters:
    --------
    ids -- sessions ids 
    dfUs -- sessions dataframe

    Return variables:
    --------
    navs --  list of ndarray 
        list of Array of navigation modality
    """

    navs = [ma.getVR(dfS) for uId,dfS in zip(ids,dfSs)]

    return navs

def getPath(dfS,listCols = ['posx','posy','posz']):
    """Gets a paths extracting equal length colums from a session data frames.

    Parameters:
    --------
    ids -- session ids 
    dfS -- session dataframe
    listCols -- list of columns keys for the columns to be extracted

    Return variables:
    --------
    navs --  ndarray 
        Array of navigation
    """

    assert (('Time' in dfS.columns) or ('time' in dfS.columns))  , "neither Time or time in csv columns: "+dfU.columns
    timeCol = 'Time'
    if not 'Time' in dfS.columns:
        timeCol = 'time'

    t =  dfS[timeCol].values
    path = np.zeros((len(t),len(listCols)+1))
    path[:,0] = t 
    for c,colName in enumerate(listCols):
        path[:,c+1] = dfS[colName].values 

    return path

def getPaths(ids,dfSs,listCols = ['posx','posy','posz']):
    """Gets a paths extracting equal length colums from a list of sessions data frames.

    Parameters:
    --------
    ids -- session ids 
    dfUs -- session dataframe
    listCols -- list of columns keys for the columns to be extracted

    Examples:
    --------
    >>> ids, fileNames, dfSs, df = readData('csv-Test/')
    >>> paths = getPaths(ids,dfSs)
    >>> len(paths)
    2
    >>> paths[0]
    array([[ 1.   , -0.57 ,  1.57 ,  3.5  ],
           [ 2.   , -0.569,  1.6  ,  3.49 ],
           [ 3.   , -0.566,  1.63 ,  3.48 ]])
    >>> dpaths = getPaths(ids,dfUs,['dirx','diry','dirz'])
    >>> len(dpaths)
    2
    >>> dpaths[0]
    array([[ 1.   ,  0.2  , -0.919, -0.341],
           [ 2.   ,  0.222, -0.779, -0.586],
           [ 3.   ,  0.254, -0.648, -0.718]])
    """

    paths = [getPath(dfS,listCols) for uId,dfS in zip(ids,dfSs)]  

    return paths

def getVarsFromSession(path,varsNames):
    """Gets variables from a sessions folder.

    Parameters:
    --------
    path -- path to sessions .csv 

    Examples:
    --------
    >>> paths = getVarsFromSession('csv-Test/','pos')
    >>> len(paths)
    2
    >>> paths[0]
    array([[ 1.   , -0.57 ,  1.57 ,  3.5  ],
           [ 2.   , -0.569,  1.6  ,  3.49 ],
           [ 3.   , -0.566,  1.63 ,  3.48 ]])
    >>> dpaths = getVarsFromSession('csv-Test/','dir')
    >>> len(dpaths)
    2
    >>> dpaths[0]
    array([[ 1.   ,  0.2  , -0.919, -0.341],
           [ 2.   ,  0.222, -0.779, -0.586],
           [ 3.   ,  0.254, -0.648, -0.718]])
    """
    #pathSes  = args.path #'bfanini-20231026-kjtgo0m0w-preprocessed-VR-session' #'/var/www/html/records/bfanini-20231026-kjtgo0m0w-preprocessed-VR-session'
    ids, fileNames, dfSs, df = readData(path)
    data = []

    #print('varsNames',varsNames)
    for varN in varsNames:
        #print('varN',varN)
        if varN == 'nav':
            data.append(getVRs(ids,dfSs)) #[ma.getVR(dfSs[uId]) for  uId in ids]
        elif varN == 'pos':
            paths=getPaths(ids,dfSs,['posx','posy','posz'])
            _,x,y,z = np.vstack(paths).T
            #print('x',np.nanmin(x),np.nanmax(x))
            data.append(paths)   
        elif varN == 'dir':
            dpaths = getPaths(ids,dfSs,['dirx','diry','dirz'])
            _,u,v,w = np.vstack(dpaths).T
            #print('u',np.nanmin(u),np.nanmax(u))
            data.append(dpaths)
        elif varN == 'f':
            data.append(getPaths(ids,dfSs,['fx','fy','fz']))   
    #paths = getPaths(ids,dfSs,['posx','posy','posz']) # [ma.getPath(dfSs[uId],['posx','posy','posz']) for uId in ids]
    #fpaths = getPaths(ids,dfSs,['fx','fy','fz']) # [ma.getPath(dfSs[uId],['fx','fy','fz']) for uId in ids]
    #dpaths = getPaths(ids,dfSs,['dirx','diry','dirz']) # [ma.getPath(dfSs[uId],['dirx','diry','dirz']) for uId in ids]
    #BBox = makeBBox(paths,dpaths,fpaths)
    #print(len(data))

    return ids, fileNames,data 


def makeRecordPlot(fname, dfS, colName = ['posx','posy','posz','dirx','diry','dirz','fx','fy','fz']):
    nav = False
    navAr = False
    if 'nav' in dfS.columns:
        time,nav = getVR(dfS).T
        time,navAr = getAR(dfS).T
    time = dfS['time']
    #bbox = makeBBox([path])
    colVals = [dfS[c].values for c in colName]
    
    #fig = plt.figure(figsize=(8, 6))
    fig, ax1 = plt.subplots(1,1, layout='constrained') #[0.15, 0.11, .85, .89])
    for l,ln in zip(colVals,colName):
        ax1.plot(time,l,label=ln)
    #ax1.plot(t,n,label='VR')
    print('nav',nav)
    if len(nav)>0 : ax1.fill_between(time, 0, 1, where=nav, alpha=0.4, transform=ax1.get_xaxis_transform(),color='green',label='VR')
    if len(nav)>0 : ax1.fill_between(time, 0, 1, where=navAr, alpha=0.4, transform=ax1.get_xaxis_transform(),color='green',label='AR')
    #lgd = ax1.legend(   bbox_to_anchor=(-0.14,))
    fig.legend(loc='outside right upper ')
    return fig


def makeSessionPreproFig(uId, path, dpath, fpath, nav, fname, bbox, SpanSelector=False):
    t,x,y,z,dx,dy,dz,fx,fy,fz,n = getSesVars(path,dpath,fpath,nav)
    plotLines = [x,y,z,dx,dy,dz,fx,fy,fz]
    lineName = ['posx','posy','posz','dirx','diry','dirz','fx','fy','fz']

    fig = plt.figure(figsize=(16, 12))
    if SpanSelector:
        plt.title('Press left mouse button and drag \n'+
                'to select a region in the top graph')
    axA = fig.add_subplot(2,2,1,projection='3d')
    ax1 = fig.add_axes([0.15, 0.11, 0.34, 0.35])
    ax2 = fig.add_subplot(2,2,4)
    axA,sc = drawPath(path, dpath=dpath, BBox=bbox, ax=axA)
    #axA.set_title('Session '+str(uId)+' file: '+fname)
    plt.colorbar(sc, ax=axA)

    #print(np.isfinite(fpath[:,1]).any())
    if np.isfinite(fpath[:,1]).any():
        axB = fig.add_subplot(2,2,2,projection='3d')
        axB,sc = drawPath(fpath, BBox=bbox,ax=axB) 
    
    for l,ln in zip(plotLines,lineName):
        ax1.plot(t,l,label=ln)
    #ax1.plot(t,n,label='VR')
    ax1.fill_between(t, 0, 1, where=n, alpha=0.4, transform=ax1.get_xaxis_transform(),color='green',label='VR')
    lgd = ax1.legend(bbox_to_anchor=(-0.14,1.))
    
    ax1.set_xlabel('t')
    ax1.set_title('Session '+str(uId)+' file: '+fname)   
    return fig,axA,axB,ax1,ax2


def makeSessionPreproFigPx(uId, path, dpath, fpath, nav, fname, bbox, SpanSelector=False):
    t,x,y,z,dx,dy,dz,fx,fy,fz,n = getSesVars(path,dpath,fpath,nav)
    plotLines = [x,y,z,dx,dy,dz,fx,fy,fz]
    lineName = ['posx','posy','posz','dirx','diry','dirz','fx','fy','fz']

    fig = plt.figure(figsize=(16, 12))
    if SpanSelector:
        plt.title('Press left mouse button and drag \n'+
                'to select a region in the top graph')
    axA = fig.add_subplot(2,2,1,projection='3d')
    ax1 = fig.add_axes([0.15, 0.11, 0.34, 0.35])
    ax2 = fig.add_subplot(2,2,4)
    axA,sc = drawPath(path, dpath=dpath, BBox=bbox, ax=axA)
    #axA.set_title('Session '+str(uId)+' file: '+fname)
    plt.colorbar(sc, ax=axA)

    #print(np.isfinite(fpath[:,1]).any())
    if np.isfinite(fpath[:,1]).any():
        axB = fig.add_subplot(2,2,2,projection='3d')
        axB,sc = drawPath(fpath, BBox=bbox,ax=axB) 
    
    for l,ln in zip(plotLines,lineName):
        ax1.plot(t,l,label=ln)
    #ax1.plot(t,n,label='VR')
    ax1.fill_between(t, 0, 1, where=n, alpha=0.4, transform=ax1.get_xaxis_transform(),color='green',label='VR')
    lgd = ax1.legend(bbox_to_anchor=(-0.14,1.))
    
    ax1.set_xlabel('t')
    ax1.set_title('Session '+str(uId)+' file: '+fname)   
    return fig,axA,axB,ax1,ax2


# def preprocessSingleSession2(uId,folderParh,fname,par,dfS,nav,path,dpath,fpath,BBox = None):
#     BBoxTemp = makeBBox([path],[dpath],[fpath])
#     #print(BBoxTemp)

#     t,x,y,z,dx,dy,dz,fx,fy,fz,n = getSesVars(path,dpath,fpath,nav)
#     plotLines = [x,y,z,dx,dy,dz,fx,fy,fz]
#     lineName = ['posx','posy','posz','dirx','diry','dirz','fx','fy','fz']

#     # #fig, [[ax3, ax4],[ax1, ax2]] = plt.subplots(2,2, figsize=(10, 10))
#     # #ax3.set_axis_off()
#     # #ax4.set_axis_off()
#     # fig = plt.figure(figsize=(16, 12))
#     # axA = fig.add_subplot(2,2,1,projection='3d')
#     # ax1 = fig.add_axes([0.15, 0.11, 0.34, 0.35])
#     # ax2 = fig.add_subplot(2,2,4)

#     # axA,sc = drawPath(path, dpath=dpath, BBox=BBoxTemp,ax=axA)
#     # #axA.set_title('Session '+str(uId)+' file: '+fname)
#     # plt.colorbar(sc, ax=axA)

#     # #print(np.isfinite(fpath[:,1]).any())
#     # if np.isfinite(fpath[:,1]).any():
#     #     axB = fig.add_subplot(2,2,2,projection='3d')
#     #     axB,sc = drawPath(fpath, BBox=BBoxTemp,ax=axB)

#     # for l,ln in zip(plotLines,lineName):
#     #     ax1.plot(t,l,label=ln)
#     # #ax1.plot(t,n,label='VR')
#     # ax1.fill_between(t, 0, 1, where=n, alpha=0.4, transform=ax1.get_xaxis_transform(),color='green',label='VR')
#     # lgd = ax1.legend(bbox_to_anchor=(-0.14,1.))
    
#     # ax1.set_xlabel('t')
#     # ax1.set_title('Session '+str(uId)+' file: '+fname)#+'\n'+
#     #                 #'Press left mouse button and drag \n'+
#     #                 #'to select a region in the top graph')
#     fig,axA,axB,ax1,ax2 = makeSessionPreproFig(uId, path, dpath, fpath, nav, fname, BBoxTemp, SpanSelector=False)
    
#     # Create the RangeSlider
#     slider_ax = fig.add_axes([0.20, 0.02, 0.60, 0.03])
#     if isinstance(par, dict):
#         t0 = par['t0']
#         t1 = par['t1']
#     else:
#         t0 = t[0]
#         t1 = t[-1]
#     slider = RangeSlider(slider_ax, "Time interval", t[0],  t[-1], valinit = (t0,t1) )

#     line2s = []
#     for l,ln in zip(plotLines,lineName):
#         if isinstance(par, dict):
#     #         ax1.axvline(t0,  alpha=0.1, color='red')
#     #         ax1.axvspan(t0,t1, alpha=0.1, color='red')
#     #         ax1.axvline(t1,  alpha=0.1, color='red')
#             indmin, indmax = np.searchsorted(t, (t0, t1))
#             #print('indmax',len(t), indmax)
#             #if t[indmax] == t1: indmax = indmax+1 
#             #print('indmax',len(t), indmax)
#             indmax = min(len(t), indmax+1)
#             #print('indmax',indmax)
#             line2dx, = ax2.plot(t[indmin:indmax], l[indmin:indmax])
#             ax2.set_xlim(t0, t1)
#         else:
#             line2dx, = ax2.plot([], [])
#             ax2.set_xlim((t[0],t[-1]))
#         line2s.append(line2dx)
#     # Create the Vertical lines on the histogram
#     lower_limit_line = ax1.axvline(slider.val[0], color='r')
#     span = ax1.axvspan(slider.val[0],slider.val[1], alpha=0.2, color='red')
#     upper_limit_line = ax1.axvline(slider.val[1], color='r')

#     def update(val):
#         # The val passed to a callback by the RangeSlider will
#         # be a tuple of (min, max)

#         ## Update the image's colormap
#         #im.norm.vmin = val[0]
#         #im.norm.vmax = val[1]

#         # Update the position of the vertical lines
#         lower_limit_line.set_xdata([val[0], val[0]])
#         #span.set_xdata(val[0],val[1], alpha=0.1, color='red')
#         upper_limit_line.set_xdata([val[1], val[1]])
    
#         # appadate span zoom
#         tmin, tmax = val[0],val[1]
#         indmin, indmax = np.searchsorted(t, (tmin, tmax))
#         indmax = min(len(t), indmax)
#         region_t = t[indmin:indmax]
#         regions = []
#         for l in plotLines:
#             region_dx = l[indmin:indmax]
#             regions.append(region_dx)
#         if len(region_t) >= 2:
#             for line2dx,region_dx in zip(line2s,regions):
#                 line2dx.set_data(region_t, region_dx)
#             #line2dy.set_data(region_t, region_dy)
#             #line2dz.set_data(region_t, region_dz)
#             ax2.set_xlim(region_t[0], region_t[-1])
#             ax2ymin = np.array(regions).min()
#             ax2ymax = np.array(regions).max()
#             ax2.set_ylim(ax2ymin-0.1, ax2ymax+0.1)
#         # Redraw the figure to ensure it updates
#         fig.canvas.draw_idle()
    
#     slider.on_changed(update)

#     plt.show()
#     keeper = False
#     tInt = None
#     kDf = None
#     keeperPath = folderParh+'-preprocessed-VR-sessions'
#     if not os.path.exists(keeperPath):
#         os.makedirs(keeperPath)
#     #print(ax2.get_xlim(),ax2.lines,ax2.lines[0].get_xdata())
#     if len(ax2.lines[0].get_xdata()) > 0:
#         keeper = True
#         print('VR Keep ',uId,fname,'t',ax2.get_xlim())
#         #keepUser.append(fname)
#         #keepUserTimeInterval.append(ax2.get_xlim())
#         tInt = ax2.get_xlim()
#         kDf = dfS[ (dfS['time']>=tInt[0]) * (dfS['time']<=tInt[1])]
#         print('keeper dataframe')
#         print(kDf)
#         kDf.to_csv(keeperPath+'/'+fname+'-preprocessed.csv',index=False,na_rep='NA')
#         plt.savefig(keeperPath+'/'+fname+'-viz.pdf')
#         #keeperDfs.append(kDf)
#     else:
#         print('Do not keep')
#         if fname in os.listdir(keeperPath):
#             os.remove(keeperPath+'/'+fname)
    
#     return keeper, tInt, kDf 

# def preprocessSessions(ids,folderParh,fileNames,pars,dfUs,navs,paths,dpaths,fpaths,BBox = None):
#     keepUser = []
#     keepUserTimeInterval = []
#     keeperDfs = []
#     for uId in ids:
#         print('uId',uId)
#         fname = fileNames[uId]
#         print('uId',uId)
#         prepro = False
#         #print(pars["preprocessedVRsession"])
#         if "preprocessedVRsessions" in pars:
#             if fname in pars["preprocessedVRsessions"]:
#                 print('pars[uId]',pars["preprocessedVRsessions"][fname])
#                 prepro = True

#         nav = navs[uId]
#         path= paths[uId]
#         dpath= dpaths[uId]
#         fpath= fpaths[uId]
#         df = dfUs[uId]
#         #print('dpath',dpath)
#         t,x,y,z,dx,dy,dz,fx,fy,fz,n = getSesVars(path,dpath,fpath,nav)
#         plotLines = [x,y,z,dx,dy,dz,fx,fy,fz]
#         lineName = ['posx','posy','posz','dirx','diry','dirz','fx','fy','fz']
        

#         BBoxTemp = makeBBox([path],[dpath],[fpath])
#         #print(BBoxTemp)

#         fig,axA,axB,ax1,ax2 = makeSessionPreproFig(uId, path, dpath, fpath, nav, fname, BBoxTemp, SpanSelector=False)

#         # fig = plt.figure(figsize=(16, 12))
#         # axA = fig.add_subplot(2,2,1,projection='3d')
#         # ax1 = fig.add_axes([0.15, 0.11, 0.34, 0.35])
#         # ax2 = fig.add_subplot(2,2,4)
#         # axA,sc = drawPath(path, dpath=dpath, BBox=BBoxTemp,ax=axA)
#         # #axA.set_title('Session '+str(uId)+' file: '+fname)
#         # plt.colorbar(sc, ax=axA)

#         # #print(np.isfinite(fpath[:,1]).any())
#         # if np.isfinite(fpath[:,1]).any():
#         #     axB = fig.add_subplot(2,2,2,projection='3d')
#         #     axB,sc = drawPath(fpath, BBox=BBoxTemp,ax=axB)
        
#         # for l,ln in zip(plotLines,lineName):
#         #     ax1.plot(t,l,label=ln)
#         # #ax1.plot(t,n,label='VR')
#         # ax1.fill_between(t, 0, 1, where=n, alpha=0.4, transform=ax1.get_xaxis_transform(),color='green',label='VR')
#         # lgd = ax1.legend(bbox_to_anchor=(-0.14,1.))
#         # ax1.set_xlabel('t')
#         # ax1.set_title('Session '+str(uId)+' file: '+fname+'\n'+
#         #                'Press left mouse button and drag \n'+
#         #                'to select a region in the top graph')
#         line2s = []
#         for l,ln in zip(plotLines,lineName):
#             if prepro:
#                 t0 = pars["preprocessedVRsessions"][fname]['t0']
#                 t1 = pars["preprocessedVRsessions"][fname]['t1']
#                 ax1.axvline(t0,  alpha=0.1, color='red')
#                 ax1.axvspan(t0,t1, alpha=0.1, color='red')
#                 ax1.axvline(t1,  alpha=0.1, color='red')
#                 indmin, indmax = np.searchsorted(t, (t0, t1))
#                 #print('indmax',len(t), indmax)
#                 #if t[indmax] == t1: indmax = indmax+1 
#                 #print('indmax',len(t), indmax)
#                 indmax = min(len(t), indmax+1)
#                 #print('indmax',indmax)
#                 line2dx, = ax2.plot(t[indmin:indmax], l[indmin:indmax])
#                 ax2.set_xlim(t0, t1)
#             else:
#                 line2dx, = ax2.plot([], [])
#                 ax2.set_xlim((t[0],t[-1]))
#             line2s.append(line2dx)


#         def onselect(tmin, tmax):
#             indmin, indmax = np.searchsorted(t, (tmin, tmax))
#             indmax = min(len(t), indmax)
#             region_t = t[indmin:indmax]
#             regions = []
#             for l in plotLines:
#                 region_dx = l[indmin:indmax]
#                 regions.append(region_dx)
#             if len(region_t) >= 2:
#                 for line2dx,region_dx in zip(line2s,regions):
#                     line2dx.set_data(region_t, region_dx)
#                 #line2dy.set_data(region_t, region_dy)
#                 #line2dz.set_data(region_t, region_dz)
#                 ax2.set_xlim(region_t[0], region_t[-1])
#                 ax2ymin = np.array(regions).min()
#                 ax2ymax = np.array(regions).max()
#                 ax2.set_ylim(ax2ymin-0.1, ax2ymax+0.1)
#                 fig.canvas.draw_idle()
#                 plt.savefig(keeperPath+'/'+fname+'-viz.pdf')



#         span = SpanSelector(
#             ax1,
#             onselect,
#             "horizontal",
#             useblit=True,
#             props=dict(alpha=0.2, facecolor="tab:blue"),
#             interactive=True,
#             drag_from_anywhere=True
#         )
#         plt.savefig(folderParh+'/'+fname+'-viz.pdf')
#         plt.show()
#         keeperPath = folderParh+'-preprocessed-VR-sessions'
#         if not os.path.exists(keeperPath):
#             os.makedirs(keeperPath)
#         #print(ax2.get_xlim(),ax2.lines,ax2.lines[0].get_xdata())
#         if len(ax2.lines[0].get_xdata()) > 0:
#             print('VR Keep ',uId,fname,'t',ax2.get_xlim())
#             keepUser.append(fname)
#             keepUserTimeInterval.append(ax2.get_xlim())
#             tInt = ax2.get_xlim()
#             kDf = df[ (df['time']>=tInt[0]) * (df['time']<=tInt[1])]
#             df.to_csv(keeperPath+'/'+fname+'-preprocessed.csv',index=False,na_rep='NA')
#             print('keeper dataframe')
#             print(kDf)
#             keeperDfs.append(kDf)
#         else:
#             print('Do not keep')

#     print("Box",BBox)
#     VRkeepers = {k:{"t0":tInt[0],"t1":tInt[1]} for k,tInt in zip(keepUser,keepUserTimeInterval)}  
#     print("VRkeepers",VRkeepers)  
#     d = {"preprocessedVRsessions":VRkeepers}
#     return d,keeperDfs
#     #with open(file, 'w', encoding='utf-8') as f:
#     #    json.dump(d, f, ensure_ascii=False, indent=4)  


def writeJson(folderPath, pars):
    #print('folderParh',folderParh)
    file = folderPath+'-pars.json'
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(pars, f, ensure_ascii=False, indent=4) 

def writeDenstity_tojson(density, xc, yc, zc, BBox, width, recordsFolderName, records, filename, npoints=800):
    densityLTh = 10**-6
    #volume = width*width*width
    data = np.vstack([ xc[ density > densityLTh ], yc[ density > densityLTh ], zc[ density > densityLTh], density[ density > densityLTh]]).T
    dataSorted = data[data[:,3].argsort()[::-1]]
    dataSorted = dataSorted[:npoints]

    dataOcc = [{'x':x,'y':y,'z':z,'density':o} for x,y,z,o in dataSorted ]
    occDict = {'records folder':recordsFolderName,'records':records,'bbox':BBox,'voxelsize':width,'points':dataOcc}

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(occDict, f, ensure_ascii=False, indent=4)

def writeOccupancy_tojson(H, BBox, width, recordsFolderName, records, filename):
    xedges, yedges,zedges = makeBinsEdges(BBox,width)
    Xc, Yc, Zc = np.meshgrid((xedges[1:]+xedges[:-1])*0.5,(yedges[1:]+yedges[:-1])*0.5, (zedges[1:]+zedges[:-1])*0.5, indexing='xy')
    xc, yc, zc = Xc.flatten(), Yc.flatten(), Zc.flatten() 

    Hswap = np.swapaxes(H,0,1)
    Hflat = Hswap.flatten()

    data = np.vstack([xc[Hflat>0],yc[Hflat>0],zc[Hflat>0],Hflat[Hflat>0]]).T
    dataSorted = data[data[:,3].argsort()[::-1]]

    dataOcc = [{'x':x,'y':y,'z':z,'density':o} for x,y,z,o in dataSorted ]
    occDict = {'records folder':recordsFolderName,'records':records,'bbox':BBox,'voxelsize':width,'points':dataOcc}

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(occDict, f, ensure_ascii=False, indent=4)

#-----------------------------------------
# kde utilities
#-----------------------------

def writeKDE_tojson(kde, BBox, width, recordsFolderName, records, filename, npoints=800):
    xedges, yedges,zedges = makeBinsEdges(BBox,width)
    Xc, Yc, Zc = np.meshgrid((xedges[1:]+xedges[:-1])*0.5,(yedges[1:]+yedges[:-1])*0.5, (zedges[1:]+zedges[:-1])*0.5, indexing='xy')
    xc, yc, zc = Xc.flatten(), Yc.flatten(), Zc.flatten() 

    xyz = np.vstack([xc, yc, zc])
    density = kde(xyz)

    densityLTh = 10**-6
    volume = width*width*width
    data = np.vstack([ xc[ density*volume > densityLTh ], yc[ density*volume > densityLTh ], zc[ density*volume > densityLTh], density[ density*volume > densityLTh]]).T
    dataSorted = data[data[:,3].argsort()[::-1]]
    #print('dataSorted',dataSorted.shape,dataSorted.dtype)
    dataSorted = dataSorted[:npoints]#.astype(np.float16)
    #print('dataSorted',dataSorted.shape,dataSorted.dtype)

    dataOcc = [{'x':x,'y':y,'z':z,'density':o} for x,y,z,o in dataSorted ]
    occDict = {'records folder':recordsFolderName,'records':records,'bbox':BBox,'voxelsize':width,'points':dataOcc}

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(occDict, f, ensure_ascii=False, indent=4)

def write2D_KDE_tojson(kde, BBox, width, recordsFolderName, records, filename, npoints=800):
    xedges, yedges, zedges  = makeBinsEdges(BBox,width=.1)
    Xc, Zc = np.meshgrid((xedges[1:]+xedges[:-1])*0.5, (zedges[1:]+zedges[:-1])*0.5, indexing='xy')
    xc, zc = Xc.flatten(), Zc.flatten()
    xz = np.vstack([xc, zc])
    density = kde(xz)

    densityLTh = 10**-6
    area = width*width
    data = np.vstack([ xc[ density*area > densityLTh ], zc[ density*area > densityLTh], density[ density*area > densityLTh]]).T
    dataSorted = data[data[:,2].argsort()[::-1]]
    #print('dataSorted',dataSorted.shape,dataSorted.dtype)
    dataSorted = dataSorted[:npoints]#.astype(np.float16)
    #print('dataSorted',dataSorted.shape,dataSorted.dtype)

    dataOcc = [{'x':x,'z':z,'density':o} for x,z,o in dataSorted ]
    occDict = {'records folder':recordsFolderName,'records':records,'bbox':BBox,'voxelsize':width,'points':dataOcc}

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(occDict, f, ensure_ascii=False, indent=4)

def write2D_kmeans_KDE_tojson(kde, BBox, width, kmeans, recordsFolderName, records, filename, npoints=800):
    xedges, yedges, zedges  = makeBinsEdges(BBox,width=.1)
    Xc, Zc = np.meshgrid((xedges[1:]+xedges[:-1])*0.5, (zedges[1:]+zedges[:-1])*0.5, indexing='xy')
    xc, zc = Xc.flatten(), Zc.flatten()
    xz = np.vstack([xc, zc])
    density = kde(xz)

    densityLTh = 10**-6
    area = width*width
    data = np.vstack([ xc[ density*area > densityLTh ], zc[ density*area > densityLTh], density[ density*area > densityLTh]]).T
    dataSorted = data[data[:,2].argsort()[::-1]]
    #print('dataSorted',dataSorted.shape,dataSorted.dtype)
    dataSorted = dataSorted[:npoints]#.astype(np.float16)
    #print('dataSorted',dataSorted.shape,dataSorted.dtype)
    labels = kmeans.predict(dataSorted[:,:2])
    #print(labels,labels.shape,data.shape)
    #print(np.unique(labels))

    dataOcc = []
    for cl in range(kmeans.n_clusters):
        dataSortedCl=[[x,z,o] for l,(x,z,o) in zip(labels,dataSorted) if l == cl]
        dataOcc.append( { 'cluster':cl,  'points' : [{'x':x,'z':z,'density':o} for x,z,o in dataSortedCl] })
    occDict = {'records folder':recordsFolderName,'records':records,'bbox':BBox,'voxelsize':width,'clusters':dataOcc}
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(occDict, f, ensure_ascii=False, indent=4)
    return occDict

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
def spherical_to_cartesian(theta, phi):
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return x, y, -z # add minus because the sistem is clockwise

def cartesian_to_spherical(x, y, z):
    theta = np.arccos(-z/ np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.sign(y)*np.arccos(x/np.sqrt(x ** 2 + y ** 2)) #np.arctan2(x, y) + (x >= 0)*np.pi
    return theta, phi


def panoramic_kde_spherical_coord(xConc,yConc,zConc,kde):
    xyz = np.vstack([xConc,yConc,zConc])
    density = kde(xyz)    # polar coordinates
    #norm = np.sum(density)
    #print('norm',norm)
    theta,phi = cartesian_to_spherical(xConc,zConc,yConc)
    return phi,theta, density

def panoramic_spherical_kde(kde,binSize): #binSize is in degrees
    #in degrees
    Thetac, Phic = np.meshgrid(np.linspace(0, np.pi,int(180/binSize)), np.linspace(-np.pi,np.pi,int(360/binSize)),  indexing='xy')
    thetac, phic = Thetac.flatten(), Phic.flatten()
    xc, zc, yc  = spherical_to_cartesian(thetac,phic)

    xyz = np.vstack([xc, yc, zc])
    sph_density = kde(xyz)
    norm1 = np.sum(sph_density)
    sph_density = sph_density/(norm1*binSize*binSize)
    #print('norm',norm1,np.sum(sph_density*binSize*binSize),sph_density.shape)

    return xc, yc, zc, thetac, phic, sph_density

def write_panoramic_kde_tojson(density, xc, yc, zc, BBox, binSize, recordsFolderName, records, filename, npoints=800):
    densityLTh = 10**-6
    #volume = width*width*width
    data = np.vstack([ xc[ density > densityLTh ], yc[ density > densityLTh ], zc[ density > densityLTh], density[ density > densityLTh]]).T
    dataSorted = data[data[:,3].argsort()[::-1]]
    dataSorted = dataSorted[:npoints]

    dataOcc = [{'x':x,'y':y,'z':z,'density':o} for x,y,z,o in dataSorted ]
    occDict = {'records folder':recordsFolderName,'records':records,'bbox':BBox,'sphericalBinSize':binSize,'points':dataOcc}

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(occDict, f, ensure_ascii=False, indent=4)

def make_panoramic_kde(xConc,zConc,yConc,bbox,pathSes,pathOut,fileNames,binSize = 2, th = 0.0001, write=False):
    xyz = np.vstack([xConc,yConc,zConc])
    kde = stats.gaussian_kde(xyz)
    
    phi,theta, density = panoramic_kde_spherical_coord(xConc,yConc,zConc,kde)
    xc, yc, zc, thetac, phic, sph_density = panoramic_spherical_kde(kde,binSize)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(phi,theta,c=density)
    #plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1)
    #print('max density',sph_density.max()*binSize*binSize)
    ax.scatter(phic[sph_density*binSize*binSize>th],thetac[sph_density*binSize*binSize>th],c=sph_density[sph_density*binSize*binSize>th])

    if False:
        kde_theta_phy = SphericalKDE(phi,theta)
        sph_density2 = kde_theta_phy(phi,theta)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(phi,theta,c=np.exp(sph_density2))
        plt.figure()
        plt.scatter(np.exp(sph_density2),density)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(xc[sph_density*binSize*binSize>th], zc[sph_density*binSize*binSize>th], yc[sph_density*binSize*binSize>th], c=sph_density[sph_density*binSize*binSize>th],s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title('3D spherical kde')

    if write:
        plt.savefig('panoramic-dir-kde.png')
        kdefname = '/panoramic-dir-3d-kde.json'
        write_panoramic_kde_tojson(sph_density, xc, yc, zc, bbox, binSize=binSize, recordsFolderName=pathSes, records=fileNames, filename=pathOut+kdefname)
        #ma.writeKDE_tojson(kde, bbox, width=0.2, recordsFolderName=pathSes, records=fileNames, filename=pathOut+kdefname)

def  make_3d_kde(xConc,zConc,yConc,bbox,pathSes,pathOut,fileNames,th=0.1,width=0.1,write=False,prefix='pos'):
    xyz = np.vstack([xConc,yConc,zConc])
    kde = stats.gaussian_kde(xyz)
    #density = kde(xyz)

    xedges, yedges, zedges  = makeBinsEdges(bbox,width)

    Xc, Yc, Zc = np.meshgrid((xedges[1:]+xedges[:-1])*0.5,(yedges[1:]+yedges[:-1])*0.5, (zedges[1:]+zedges[:-1])*0.5, indexing='xy')
    xc, yc, zc = Xc.flatten(), Yc.flatten(), Zc.flatten()
    xyz = np.vstack([xc, yc, zc])
    density = kde(xyz)


    # y-up
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(xc[density>th], zc[density>th], yc[density>th], c=density[density>th],s=1)
    #fig.colorbar(sc, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_xlim((xedges[0],xedges[-1]))
    ax.set_zlim((yedges[0],yedges[-1]))
    ax.set_ylim((zedges[0],zedges[-1]))
    ax.set_title('3D kde')
    plt.savefig('3d-kde.png')

    

    if write:
        kdefname = '/'+prefix+'-3d-kde.json'
        #if not 'pos' in prefix:
        #    kdefname = '/dir-3d-kde.json'
        writeKDE_tojson(kde, bbox, width=width, recordsFolderName=pathSes, records=fileNames, filename=pathOut+kdefname)

def make_2d_kde(xConc,zConc,bbox, pathSes, pathOut,fileNames, th=0.1, width=0.1,write=False):
    xz = np.vstack([xConc,zConc])
    kde = stats.gaussian_kde(xz)
    #density = kde(xz)
    
    xedges, yedges, zedges  = makeBinsEdges(bbox,width)
    Xc, Zc = np.meshgrid((xedges[1:]+xedges[:-1])*0.5, (zedges[1:]+zedges[:-1])*0.5, indexing='xy')
    xc, zc = Xc.flatten(), Zc.flatten()
    xz = np.vstack([xc, zc])
    density = kde(xz)
    f = np.reshape(density.T, Xc.shape)

    fig = plt.figure()
    ax = fig.gca()

    #sc = plt.scatter(xc[density>th], zc[density>th], c=density[density>th],s=1)
    #fig.colorbar(sc, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_xlim((xedges[0],xedges[-1]))
    ax.set_ylim((zedges[0],zedges[-1]))
    cfset = ax.contourf(Xc, Zc, f)
    plt.title('2D kde')
    plt.savefig('2d-kde.png')

    if write:
        kdefname = '/pos-2d-kde.json'
        if dir == True:
            kdefname = '/dir-2d-kde.json'
        write2D_KDE_tojson(kde, bbox, width=width, recordsFolderName=pathSes, records=fileNames, filename=pathOut+kdefname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process tracks from immersive and not-immersive explorations of Aton 3D models which were captured with Merkhet.')
    parser.add_argument('--path', type=dir_path, help='Path to folder.',default='csv-Merkhet/')
    parser.add_argument('--kde', action=argparse.BooleanOptionalAction, help='Performe kernel density estimation (kde)',default=False)
    parser.add_argument('--preprocess', action=argparse.BooleanOptionalAction, help='Preprocess data',default=False)
    
    args = parser.parse_args()
    #print('path',args.path)
    runPrePro = args.preprocess 
    print('run runPrePro:',runPrePro)
    runKde = args.kde 
    print('run kde:',runKde)

    # adjust folder path
    folderParh = args.path
    if folderParh[-1] == '/': folderParh = folderParh[:-1]
    print('path',folderParh)

    # csv-Vest/ https://sketchfab.com/3d-models/the-upper-vestibule-e74928dc62fe457892e52dd97b6aa6e0 
    # csv-Merkhet/ data acquired with merkhet   
    ids,fileNames,dfUs,df = readData(folderParh)
    navs = getVRs(ids,dfUs)
    paths = getPaths(ids,dfUs,['posx','posy','posz'])
    fpaths = getPaths(ids,dfUs,['fx','fy','fz'])
    dpaths = getPaths(ids,dfUs,['dirx','diry','dirz'])
    print(ids)
    print(fpaths[0].shape, paths[0].shape)
    pars = readDataPars(folderParh)
    #print(ids==ids2)
    #print(paths2[0].shape)
    print('df')
    print(df)
    df.to_csv(folderParh+'-paths.csv',index=False,na_rep='NA')
    
    if runPrePro:
        BBox = makeBBox(paths,dpaths,fpaths)
        d = preprocessSessions(navs,paths,dpaths,BBox)
        print(d)

    keepUser = list(pars["users"].keys())
    assert len(keepUser) > 0, "Data must be preprocessed, use --preprocess flag."
    print('users pars',pars["users"]['1'],pars["users"][str(1)])

    for uId in ids:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        path = paths[uId]
        if len(dpaths)>0:
            dpath = dpaths[uId]
            plt.figure()
            t,x,y,z = dpath.T
            plt.plot(t,x,label='x')
            plt.plot(t,y,label='y')
            plt.plot(t,z,label='z')
            plt.legend()
            plt.xlabel('t')
            plt.title('User '+str(uId))
            if str(uId) in keepUser:
                print('keep',pars["users"][str(uId)])
                t0 = pars["users"][str(uId)]['t0']
                t1 = pars["users"][str(uId)]['t1']
                print('time',t0,t1)
                plt.axvline(t0)
                plt.axvline(t1)
        else:
            dpath=None
        ax,sc = drawPath(path, dpath=dpath, BBox=BBox,ax=ax)
        ax.set_title('User '+str(uId))
        

        fpath = fpaths[uId]
        if np.isfinite(fpath).any(): 
            ax,sc = drawPath(fpath, BBox=BBox,ax=ax)
        plt.colorbar(sc, ax=ax)
       

        
        

    allPaths3D(paths)
    allPaths3D(dpaths)
    if not np.isnan(fpaths[0][0,1]):
        allPaths3D(fpaths)
    

    if runKde:
        pathsConc =  np.vstack(dpaths)
        tConv,xConc,yConc,zConc = pathsConc.T#np.concatenate(xs),np.concatenate(ys),np.concatenate(zs)

        xyz = np.vstack([xConc,yConc,zConc])
        kde = stats.gaussian_kde(xyz)
        density = kde(xyz)

        plotKDE(xConc,yConc,zConc,density)
    
    if False:
        dHs = []

        for path in dpaths:
            H,edges = occupancy3D(path,BBox)
            dHs.append(H)

        for H,path in zip(dHs,dpaths):
            drawMarginals(H,path,BBox)
            drawPath(path,ax=None)
            displayH3Dstack(H,path,BBox, step=1)
            showH(H,BBox,path=path)
    plt.show()

