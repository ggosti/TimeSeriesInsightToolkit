#!/home/ggosti/anaconda3/bin python

import timeSeriesInsightToolkit as tsi
import json
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#from spherical_kde import SphericalKDE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate kde from immersive and not-immersive explorations of Aton 3D models which were captured with Merkhet.')
    parser.add_argument('--path', type=tsi.dir_path, help='Path to folder with sessions.')
    parser.add_argument('--opath', type=tsi.dir_path, help='Path to output folder.')
    parser.add_argument('--dir', action='store_true', help='Use direction (dir) as opposed to positions (pos).')
    parser.add_argument('--proj3d', action='store_true', help='Generete 3D kde.')
    parser.add_argument('--proj2d', action='store_true', help='Generete 2D kde.')
    parser.add_argument('--panoramic', action='store_true', help='Use for panoramic scene.')
    parser.add_argument('--width', type=float, default=0.4, help='Set discretization width.')
    parser.set_defaults(proj3d=False, proj2d=False,dir=False,panoramic=False)

    args = parser.parse_args()
    print('path',args.path)

    proj2d = args.proj2d
    print('proj2d', proj2d)
    proj3d = args.proj3d
    print('proj3d', proj3d)

    dir = args.dir
    print('dir',dir)

    panoramic = args.panoramic
    print('panoramic',panoramic)

    width = args.width
    print('width',width)

    pathSes  = args.path #'/var/www/html/records/bfanini-20231026-kjtgo0m0w-gated-preprocessed-VR-sessions' #'./bfanini-20231026-kjtgo0m0w-gated-preprocessed-VR-sessions'
    pathOut = args.opath
    print('input',pathSes)
    print('output',pathOut)


    ids, fileNames, dfSs, df = tsi.readData(pathSes)
    paths = tsi.getPaths(ids,dfSs,['posx','posy','posz'])
    bbox = tsi.makeBBox(paths,paths,paths)
    
    dpaths = tsi.getPaths(ids,dfSs,['dirx','diry','dirz'])
    bbox = tsi.makeBBox(paths,dpaths,dpaths) #,fpaths)
    

    ncols = 4
    f,axs = plt.subplots( math.ceil(len(paths)/ncols), 4, sharex=True, sharey=True )
    #f2,axs2 = plt.subplots( math.ceil(len(paths)/ncols), 4, sharex=True, sharey=True , projection='3d')
    #fig = plt.figure(figsize=(10,10))


    for i,(uId,ax) in enumerate(zip(ids,axs.flat)):
        path = paths[uId] #ts[uId],xs[uId],ys[uId],zs[uId]
        dpath = dpaths[uId]
        tsi.drawPath2DT(path,dpath,BBox=bbox,ax=ax)
        #ax2 = fig.add_subplot(math.ceil(len(paths)/ncols), 4, i, projection='3d')
        #ma.drawPath(path,dpath,BBox=bbox,ax=ax2)
        if i < 5:
            tsi.drawPath(path,dpath,BBox=bbox)#,ax=ax2)

    if False:
        if np.isnan(fpaths[0]).all():
            f,axs = plt.subplots( math.ceil(len(paths)/ncols), 4, sharex=True, sharey=True )
            for uId,ax in zip(ids,axs.flat):
                fpath = fpaths[uId] #ts[uId],xs[uId],ys[uId],zs[uId]
                tsi.drawPath2DT(fpath,BBox=bbox,ax=ax,yup=True)

    tsi.allPaths3D(paths)

    #--------------------------
    #KDE 3D
    #--------------------------
    if proj3d:
        if dir == True:
            paths = dpaths

        pathsConc =  np.vstack(paths)#[:1000,:]
        tConv,xConc,yConc,zConc = pathsConc.T#np.concatenate(xs),np.concatenate(ys),np.concatenate(zs)

        #xyz = np.vstack([xConc,yConc,zConc])
        #kde = stats.gaussian_kde(xyz)
        #density = kde(xyz)
        #ma.plotKDE(xConc,yConc,zConc,density)

        if panoramic:
            if dir == True:
                norm = np.sqrt((xConc**2) + (yConc**2)+(zConc**2))
                #plt.figure()
                #plt.hist(norm,bins=100)
                xConc,yConc,zConc = xConc/norm,yConc/norm,zConc/norm
                tsi.make_panoramic_kde(xConc,zConc,yConc,bbox,pathSes,pathOut,fileNames)
        else:
            #print('3d kde')
            tsi.make_3d_kde(xConc,zConc,yConc,bbox,pathSes,pathOut, fileNames, th=0.000001,width=width,write=True)

    if proj2d:
        #--------------------------
        #KDE 2D
        #--------------------------


        pathsConc =  np.vstack(paths)
        tConv,xConc,yConc,zConc = pathsConc.T#np.concatenate(xs),np.concatenate(ys),np.concatenate(zs)

        #xz = np.vstack([xConc,zConc])
        #kde = stats.gaussian_kde(xz)
        #density = kde(xz)

        #ma.plotKDE(xConc,yConc,zConc,density)

        #plt.figure()
        #plt.scatter(xConc,zConc,c=density,s=1)

        tsi.make_2d_kde(xConc,zConc,bbox, pathSes, pathOut, fileNames, th=0.0001, width=width, write=True)



    plt.show()



    
