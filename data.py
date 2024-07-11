# rawGroups.py

import os

def read_raw():
    path = '/var/www/html/records/raw'
    # get variables from records    
    groups = os.listdir(path)
    return groups

def read_proc():
    path = '/var/www/html/records/proc'
    # get variables from records    
    groups = os.listdir(path)
    return groups