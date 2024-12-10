

# Define classes Project, Group, and Record with relationships and data methods.

class Group:
    """
    Group model for managing groups of records which are part of a project.

    >>> group1 = Group(group_id=1, name="Test Group 1", path = "/path/project/group1")
    >>> group2 = Group(group_id=2, name="Test Group 2", path = "/path/project/group2")

    >>> group1.name
    'Test Group 1'

    >>> group2.name
    'Test Group 2'
    """
    def __init__(self, group_id, name, path):
        self.id = group_id
        self.name = name
        self.path = path #Column(String, nullable=True)
        self.startDate = None  #Column(Date, nullable=True)
        self.endDate = None #endDate #Column(Date, nullable=True)
        self.notes = {} #Column(String, nullable=True)
        self.parent_group = None
        self.records = []

    def add_record(self, record):
        self.records.append(record)


class Project:
    """
    Project model for managing projects folders containing groups and other attributes.


    >>> project1 = Project(project_id=1, name="Test Project 1",path = "/path/project1")
    >>> project2 = Project(project_id=2, name="Test Project 2",path = "/path/project1")

    >>> project1.name
    'Test Project 1'

    >>> project2.name
    'Test Project 2'

    >>> group1 = Group(group_id=1, name="Test Group 1", path = "/path/project/group1")
    >>> group2 = Group(group_id=2, name="Test Group 2", path = "/path/project/group2")
    >>> group3 = Group(group_id=1, name="Test Group 3", path = "/path/project/group3")


    >>> group1.name
    'Test Group 1'

    >>> group2.name
    'Test Group 2'

    >>> project1.add_group(group1)

    >>> [g.name for g in  project1.groups]
    ['Test Group 1']

    >>> project2.add_group(group2)
    >>> project2.add_group(group3)

    >>> [g.name for g in  project2.groups]
    ['Test Group 2','Test Group 3']

    """
    def __init__(self, project_id, name, path):
        self.id = project_id
        self.name = name
        self.path = path #Column(String, nullable=True)
        self.startDate = None  #Column(Date, nullable=True)
        self.endDate = None #endDate #Column(Date, nullable=True)
        self.notes = {} #Column(String, nullable=True)
        self.parentProject = None

        self.groups = []

    def add_group(self, group):
        self.groups.append(group)


#class Record:
#    def __init__(self, record_id, data):
#        self.record_id = record_id
#        self.data = data
