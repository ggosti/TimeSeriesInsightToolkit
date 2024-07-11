# rawGroups.py

import os

def get_sub_dirs(path):
    print('path',os.walk(path))
    groups = [x[0] for x in os.walk(path)]
    return groups

def read_raw():
    path = '/var/www/html/records/raw'
    groups = get_sub_dirs(path)
    return groups

def read_proc():
    path = '/var/www/html/records/proc'
    groups = get_sub_dirs(path)
    return groups