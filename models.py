

class Project:
    """
    project model for managing projects folders containing groups and other attributes.


    >>> revent1 = RawEvent(id=1, name="Test Raw Event 1")
    >>> session.add(revent1)
    >>> revent2 = RawEvent(id=2, name="Test Raw Event 2")
    >>> session.add(revent2)
    >>> session.commit()

    >>> revent1.name
    'Test Raw Event 1'

    >>> revent2.name
    'Test Raw Event 2'

    >>> session.query(RawEvent).first().name
    'Test Raw Event 1'
    """
    def __init__(self, project_id, name, path):
        self.project_id = project_id
        self.name = name
        self.startDate = None  #Column(Date, nullable=True)
        self.endDate = None #endDate #Column(Date, nullable=True)
        self.notes = {} #Column(String, nullable=True)
        self.path = path #Column(String, nullable=True)
        self.parentProject = None

        self.groups = []

    def add_group(self, group):
        self.groups.append(group)





# Define classes Project, Group, and Record with relationships and data methods.
class Project:
    def __init__(self, project_id, name):
        self.project_id = project_id
        self.name = name
        self.groups = []

    def add_group(self, group):
        self.groups.append(group)

class Group:
    def __init__(self, group_id, name):
        self.group_id = group_id
        self.name = name
        self.records = []

    def add_record(self, record):
        self.records.append(record)

class Record:
    def __init__(self, record_id, data):
        self.record_id = record_id
        self.data = data