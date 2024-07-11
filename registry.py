# rawGroups.py

import os
from flask import abort

def get_sub_dirs(path):
    print('path',path,os.listdir(path))
    groups = [x for x in os.listdir(path) if os.path.isdir(path+'/'+x)] #os.walk(path)]
    print('groups',groups)
    return groups

def read_raw():
    path = '/var/www/html/records/raw'
    groups = get_sub_dirs(path)
    return groups

def read_raw_group(group):
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


def read_proc_version_group(version,group):
    path = f'/var/www/html/records/proc/{version}/{group}'
    if os.path.isdir(path): 
        groups = [x for x in os.listdir(path) if '.csv' in x]
    else:
        abort(
            404, f"{group} not found in processed"
        )
    return groups
