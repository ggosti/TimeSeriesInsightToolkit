# rawGroups.py

import os
from flask import abort

import timeSeriesInsightToolkit as tsi

def get_sub_dirs(path):
    print('path',path,os.listdir(path))
    groups = [x for x in os.listdir(path) if os.path.isdir(path+'/'+x)] #os.walk(path)]
    print('groups',groups)
    return groups

def read_raw():
    path = '/var/www/html/records/raw'
    groups = get_sub_dirs(path)
    return groups

def read_raw_groups(group):
    path = '/var/www/html/records/raw'+'/'+group
    if os.path.isdir(path): 
        groups = [x for x in os.listdir(path) if '.csv' in x]
    else:
        abort(
            404, f"{group} not found in raw"
        )
    return groups


def read_proc():
    path = '/var/www/html/records/proc'
    groups = get_sub_dirs(path)
    return groups


def read_all_proc_group_versions(group):
    path = f'/var/www/html/records/proc/{group}'
    if os.path.isdir(path): 
        version = get_sub_dirs(path) #os.listdir(path) #[x for x in os.listdir(path) if '.csv' in x]
    else:
        abort(
            404, f"{group} not found in processed"
        )
    return version

def read_proc_group_version(version,group):
    path = f'/var/www/html/records/proc/{group}/{version}/'
    if os.path.isdir(path): 
        groups = [x for x in os.listdir(path) if '.csv' in x]
    else:
        abort(
            404, f"{group} not found in processed"
        )
    return groups

def read_proc_group_version_record(version,group,record):
    path = f'/var/www/html/records/proc/{group}/{version}/'
    if os.path.isfile(path+record): 
        dfS = tsi.readSessionData(path,record)
    else:
        abort(
            404, f"{group} not found in processed"
        )
    
    print(record)
    print(dfS)
    
    return {'record':record,'dfS': dfS.values.tolist()} #dfS.to_dict('records'),'record':record}
