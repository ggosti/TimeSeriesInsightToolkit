#!~/anaconda3/envs/mat/bin/python

import timeSeriesInsightToolkit as tsi
import json
import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gate sessions from immersive and not-immersive explorations of Aton 3D models which were captured with Merkhet. '
                                     +'Sessions are gated according to time duration and variance')
    parser.add_argument('--path', type=dir_path, help='Path to folder with sessions.')
    parser.add_argument('--path2', type=dir_path, help='Add argument to compare session in a second folder. Path to second folder with sessions.')
    parser.add_argument('--panoramic', action='store_true', help='use directions as opposed to positions for panoramic views in which the viewer is at the center')
    parser.add_argument('--write',action='store_true',help='Select to save ungated sessions.')
    #parser.add_argument('--show',action='store_true',help='Select to show scatter plot.') 
    parser.add_argument('--noqt',action='store_true',help='Select to disable qt.')

    parser.set_defaults(panoramic=False,write=False,noqt=False) #show=False,noqt=False)

    args = parser.parse_args()
    print('path',args.path)
    if isinstance(args.path2, str): print('path2',args.path2)
    write = args.write
    print('write',write)
    panoramic = args.panoramic
    print('panoramic',panoramic)
    #show = args.show
    #print('show',show)
    noqt = args.noqt
    print('noqt',noqt)    

    if noqt:
        matplotlib.use('Agg')


    pathSesRec = args.path #'bfanini-20231026-kjtgo0m0w-preprocessed-VR-sessions' #'/var/www/html/records/bfanini-20231026-kjtgo0m0w-preprocessed-VR-sessions'
    pathSes = pathSesRec
    if '/' == pathSes[-1]:
        pathSes = pathSes[:-1] #takeout the slash and than the folder name
        print(pathSes)
    if 'preprocessed-VR-sessions' in pathSes: 
        pathSes = pathSes[:-len('/preprocessed-VR-sessions')] 
        print(pathSes)
    baseName = os.path.basename(pathSes)

    ids, fileNames, dfSs, df = tsi.readData(pathSesRec)
    #print(dfSs)
    if not panoramic:
        paths=tsi.getPaths(ids,dfSs,['posx','posy','posz'])
        _,x,y,z = np.vstack(paths).T
    else:
        paths = tsi.getPaths(ids,dfSs,['dirx','diry','dirz'])
        _,u,v,w = np.vstack(paths).T
    #paths = ma.getVarsFromSession(pathSes,['pos'])[0]
    #ids, fileNames, [paths,dpaths] = ma.getVarsFromSession(pathSes,['pos','dir'])
    #if panoramic: paths=dpaths

    thTime = 35
    thVar = 0.1#1.#0.1#1.#0.4 #2.5

    totTimes = []
    totVars = []
    for path in paths:
        t,x,y,z = path.T
        totTime = t[-1]-t[0]
        totTimes.append(totTime)
        totVar = np.var(x)+np.var(y)+np.var(z)
        totVars.append(totVar) 

    pathSes2  = args.path2 

    if isinstance(pathSes2, str): 
        paths2 = tsi.getVarsFromSession(pathSes2,'pos')

        totTimes2 = []
        totVars2 = []
        for path in paths2:
            t,x,y,z = path.T
            totTime = t[-1]-t[0]
            totTimes2.append(totTime)
            totVar = np.var(x)+np.var(y)+np.var(z)
            totVars2.append(totVar)         


    f,axs = plt.subplots(2,2,figsize=(10,10))

    timeBins = np.linspace(0,np.max(totTimes)*1.1,100)
    varBins = np.linspace(0,np.max(totVars)*1.1,100)

    if isinstance(pathSes2, str): axs[0,0].hist(totTimes2,bins=timeBins)
    axs[0,0].hist(totTimes,bins=timeBins)
    axs[0,0].axvline(thTime,color='gray')
    axs[0,0].set_xlabel('session time (s)')


    if isinstance(pathSes2, str): axs[1,1].hist(totVars2,bins=varBins)
    axs[1,1].hist(totVars,bins=varBins)
    axs[1,1].axvline(thVar,color='gray')
    axs[1,1].set_xlabel('variance')

    #plt.figure()
    if isinstance(pathSes2, str): axs[1,0].scatter(totTimes2,totVars2)
    axs[1,0].scatter(totTimes,totVars)
    axs[1,0].axvline(thTime,color='gray')
    axs[1,0].axhline(thVar,color='gray')
    axs[1,0].set_xlabel('session time (s)')
    axs[1,0].set_ylabel('variance')
    axs[1,0].set_xlim((timeBins[0],timeBins[-1]))
    axs[1,0].set_ylim((varBins[0],varBins[-1]))

    plt.savefig(baseName+'-gateVars.png')
    plt.savefig(pathSes+'/gateVars.png')
    if not noqt:
        plt.show()

    if write:
        file = pathSes+'/pars.json'
        print('read',file)
        if os.path.isfile(file):
            print('read',file)
            with open(file) as f:
                d = json.load(f)
        print(d['preprocessedVRsessions'])
        d['gated'] = {'thVar grater than':thVar}
        d['preprocessedVRsessions-gated'] = {}
        
        gatedPath = pathSes+'/preprocessed-VR-sessions-gated'
        if not os.path.exists(gatedPath):
            os.makedirs(gatedPath)
        
        for i, fName,dfS,totVar,totTime, path in zip(ids, fileNames, dfSs,totVars, totTimes, paths):
            #print('fName',fName)
            if totVar > thVar and totTime > thTime:
                session = fName[:-len('-preprocessed')]
                print('session',session)
                assert session in d['preprocessedVRsessions'], "session not in preprocessedVRsessions "+session
                d['preprocessedVRsessions-gated'][session] = d['preprocessedVRsessions'][session]
                if 'ID' in dfS.columns: dfS = dfS.drop(columns=['ID', 'filename'])
                dfS.to_csv(gatedPath+'/'+session+'-preprocessed.csv',index=False,na_rep='NA')
            else:
                session = fName[:-len('-preprocessed')]
                print('session',session)
                tsi.drawPath(path, dpath=None, BBox=None)
        if not noqt: plt.show()

        tsi.writeJson(pathSes,d)

#ids, fileNames, dfSs, df = ma.readData(pathSes)
#j=0
#for i,path,dpath,totTime,totVar in zip(range(len(paths)),paths,dpaths,totTimes,totVars):
#    if totTime < thTime or totVar < thVar:
#        ma.drawPath(path, dpath=dpath, BBox=BBox)
#        totTimes2.append(totTime)
#        totVars2.append(totVar)
#    else:
#        t,x,y,z = path.T
#        ids2.append(i)
#        paths2.append(path)
#        dpaths2.append(dpath)
#        #print(i,j,np.var(x)+np.var(y)+np.var(z),totVar)
#        j=j+1
