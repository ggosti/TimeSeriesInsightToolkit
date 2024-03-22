import timeSeriesInsightToolkit as tsi
import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate png preview of records from group')
    parser.add_argument('--path', type=dir_path, help='Path to folder with sessions.')
    parser.add_argument('--opath', type=dir_path, help='Path to output folder.')

    args = parser.parse_args()

    pathSes  = args.path
    print('path',pathSes)
    #print('opath',args.opath)
    ids, fileNames, [paths,dpaths] = tsi.getVarsFromSession(pathSes,['pos','dir'])

    groupName = str(pathlib.PurePath(pathSes).stem)
    print('group',groupName)

    if args.opath == None:
        opath = pathlib.PurePath(pathSes)
        print('opath',opath)
    else:
        opath  = pathlib.PurePath(args.opath)
        print('output path',opath)
        #opath =  opath / 'prep' 
        opath =  opath / ( groupName+'-prev')
        print('output path',opath)
        if not os.path.exists(opath):
            os.makedirs(opath)

    for i,(fname,path,dpath) in enumerate(zip(fileNames,paths,dpaths)):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')
        t,x,y,z = path.T
        ax,sc = tsi.drawPath(path,dpath=dpath,BBox=None,ax=ax)
        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        pngName = fname+'-prev.png'
        print('fname',pngName)
        plt.savefig(opath / pngName, transparent=True)
        plt.close()
    #plt.show()
    
