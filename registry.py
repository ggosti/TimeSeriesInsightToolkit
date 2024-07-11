# rawGroups.py

import os

def get_sub_dirs(path):
    print('path',path)
    groups = [x for x in os.listdir(path) if os.path.isdir(x)] #os.walk(path)]
    print('groups',groups)
    return groups

def read_raw():
    path = '/var/www/html/records/raw'
    groups = get_sub_dirs(path)
    return groups

def read_proc():
    path = '/var/www/html/records/proc'
    groups = get_sub_dirs(path)
    return groups

def read_group(group):
    path = '/var/www/html/records/raw'
    groups = get_sub_dirs(path+'/'+group)

    return groups