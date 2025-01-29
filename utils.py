import pandas as pd
import os
from models import Project, Group, Record

def get_records(path):
    dfs = []
    for record_name in os.listdir(path):
        record_path = os.path.join(path, record_name)
        if os.path.isfile(record_path) and record_path.endswith('.csv'):
            df = pd.read_csv(record_path)
            dfs.append([df,record_name,record_path])
    return dfs


def load_data(project_dir,projects,groups,records):
    """
    Utility function for loading datastructure give the path of the directory with the projects.

    >>> projects = []
    >>> groups = []
    >>> records = []
    >>> projects, groups, records = load_data("./test/records/raw/", projects, groups, records)
    >>> [(p.id, p.name) for p in projects]
    [(1, 'event1'), (2, 'event2')]
    >>> project = projects[0]
    >>> [(g.id, g.name) for g in project.groups]
    [(1, 'group1'), (2, 'group2')]
    >>> group = project.groups[0]
    >>> [(r.id, r.name, r.step,r.version) for r in group.records]
    [(1, 'U1.csv', 'raw', None), (2, 'U2.csv', 'raw', None), (3, 'U3.csv', 'raw', None)]

    >>> projects = []
    >>> groups = []
    >>> records = []
    >>> projects, groups, records = load_data("./test/records/proc/", projects, groups, records)
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
    projectsInner = []
    i = len(projects) +1
    for project_name in os.listdir(project_dir):
        project_path = os.path.join(project_dir, project_name)
        #print('project_path',project_path)
        if os.path.isdir(project_path):
            project = Project(i, f"{project_name}",project_path)
            i = i + 1
            projectsInner.append(project)
            project.step = step
    projects = projects + projectsInner
    #print('projectsInner', [(p.id, p.name, p.path) for p in projectsInner])
    

    # Load groups
    groupsInner = []
    i = len(groups) + 1
    for project in projectsInner:
        for group_name in os.listdir(project.path):
            group_path = os.path.join(project.path, group_name)
            #print('group_path',group_path)
            #if os.path.isfile(group_path) and group_path.endswith('.csv'):
            if os.path.isdir(group_path):
                group = Group(i, f"{group_name}",group_path)
                i = i + 1
                groupsInner.append(group)
                project.add_group(group)
                group.project = project
    groups = groups + groupsInner
    #print('groupsInner', [(g.id, g.name, g.path) for g in groupsInner])

    # Load records
    i = len(records) + 1
    if step == 'raw':
        for group in groups:
            dfs = get_records(group.path)
            for df,record_name,record_path in dfs:
                record = Record(i, record_name, record_path, 'raw', df)
                i = i + 1
                records.append(record)
                group.add_record(record)
                record.group = group

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
                        record.group = group



    return projects,groups,records



if __name__ == "__main__":
    steps = ['raw','proc']#'raw',
    projects = []
    groups = []
    records = []

    for step in steps:  
        projects,groups,records = load_data(f"./test/records/{step}/",projects,groups,records)
        print('projects',[p.name for p in projects])
        print('groups',[g.name for g in groups])


    print('projects', [(p.id, p.name, p.path) for p in projects])
    for project in projects:
         print('groups in ', project.name,' : ' , [(g.id, g.name, g.path) for g in project.groups])
    for group in groups:
        print('records in ', project.name, group.name,' : ' , [(r.id, r.name, r.path,r.step,r.version) for r in group.records])