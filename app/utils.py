
import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import os
from app.models import Project, Group, Record

def get_records(path):
    dfs = []
    for record_name in os.listdir(path):
        record_path = os.path.join(path, record_name)
        if os.path.isfile(record_path) and record_path.endswith('.csv'):
            df = pd.read_csv(record_path)
            dfs.append([df,record_name,record_path])
    return dfs


def load_data(project_dir):
    """
    Utility function for loading datastructure give the path of the directory with the projects.

    >>> projects = load_data("./test/records/raw/")
    >>> [(p.id, p.name) for p in projects]
    [(1, 'event1'), (2, 'event2')]
    >>> project = projects[0]
    >>> [(g.id, g.name) for g in project.groups]
    [(1, 'group1'), (2, 'group2')]
    >>> group = project.groups[0]
    >>> [(r.id, r.name, r.step,r.version) for r in group.records]
    [(1, 'abc.csv', 'raw', None)]

    >>> projects = load_data("./test/records/proc/")
    >>> [(p.id, p.name) for p in projects]
    [(1, 'event1'), (2, 'event2')]
    >>> project = projects[0]
    >>> [(g.id, g.name) for g in project.groups]
    [(1, 'group1'), (2, 'group2')]
    >>> group = project.groups[0]
    >>> [(r.id, r.name, r.step,r.version) for r in group.records]
    [(1, 'abc1.csv', 'proc', 'preprocessed-VR-sessions'), (2, 'abc1.csv', 'proc', 'preprocessed-VR-sessions-gated')]
    

    """

    if 'raw/' in project_dir:
        step = 'raw'
    if 'proc/' in project_dir:
        step = 'proc'

    # Load projects
    projects = []
    i = 1
    for project_name in os.listdir(project_dir):
        project_path = os.path.join(project_dir, project_name)
        #print('project_path',project_path)
        if os.path.isdir(project_path):
            project = Project(i, f"{project_name}",project_path)
            i = i + 1
            projects.append(project)
    #print('projects', [(p.id, p.name, p.path) for p in projects])
    

    groups = []
    i = 1
    for project in projects:
        for group_name in os.listdir(project.path):
            group_path = os.path.join(project.path, group_name)
            #print('group_path',group_path)
            #if os.path.isfile(group_path) and group_path.endswith('.csv'):
            if os.path.isdir(group_path):
                group = Group(i, f"{group_name}",group_path)
                i = i + 1
                groups.append(group)
                project.add_group(group)
    #print('groups', [(g.id, g.name, g.path) for g in groups])

    records = []
    i = 1
    if step == 'raw':
        for group in groups:
            dfs = get_records(group.path)
            for df,record_name,record_path in dfs:
                record = Record(i, record_name, record_path, 'raw', df)
                i = i + 1
                records.append(record)
                group.add_record(record)

    if step == 'proc':
        for group in groups:
            for ver in os.listdir(group.path):
                ver_path = os.path.join(group.path, ver)
                if os.path.isdir(ver_path):
                    dfs = get_records(ver_path)
                    for df,record_name,record_path in dfs:
                        record = Record(i, record_name, record_path, 'proc',df)
                        record.set_ver(ver)
                        i = i + 1
                        records.append(record)
                        group.add_record(record)



    return projects



if __name__ == "__main__":
    projects = load_data("./test/records/raw/")
    print('projects', [(p.id, p.name, p.path) for p in projects])
    for project in projects:
         print('groups in ', project.name,' : ' , [(g.id, g.name, g.path) for g in project.groups])
    for project in projects:
        for group in project.groups:
            print('records in ', project.name, group.name,' : ' , [(r.id, r.name, r.path,r.step,r.version) for r in group.records])