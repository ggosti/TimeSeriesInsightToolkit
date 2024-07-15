# rawGroups.py

#import base64
from io import BytesIO
import os
from flask import abort, send_file

import timeSeriesInsightToolkit as tsi

def get_sub_dirs(path):
    print('path',path,os.listdir(path))
    groups = [x for x in os.listdir(path) if os.path.isdir(path+'/'+x)] #os.walk(path)]
    print('groups',groups)
    return groups

def read_raw():
    path = '/var/www/html/records/raw'
    events = get_sub_dirs(path)
    return events

def read_raw_event(eid):
    path = f'/var/www/html/records/raw/{eid}'
    if os.path.isdir(path): 
        groups = get_sub_dirs(path)
    else:
        abort(
            404, f"{eid} not found in raw"
        )
    return groups

def read_raw_groups(eid,gid):
    path = f'/var/www/html/records/raw/{eid}/{gid}/'
    if os.path.isdir(path): 
        groups = [x for x in os.listdir(path) if '.csv' in x]
    else:
        abort(
            404, f"{gid} not found in raw"
        )
    return groups


def read_proc():
    path = '/var/www/html/records/proc'
    events = get_sub_dirs(path)
    return events

def read_proc_event(eid):
    path = f'/var/www/html/records/proc/{eid}/'
    if os.path.isdir(path): 
        groups = get_sub_dirs(path)
    else:
        abort(
            404, f"{eid} not found in raw"
        )
    return groups

def read_proc_group_versions(eid,gid):
    path = f'/var/www/html/records/proc/{eid}/{gid}'
    if os.path.isdir(path): 
        version = get_sub_dirs(path) #os.listdir(path) #[x for x in os.listdir(path) if '.csv' in x]
    else:
        abort(
            404, f"{gid} not found in processed"
        )
    return version

def read_proc_group_version(eid,version,gid):
    path = f'/var/www/html/records/proc/{eid}/{gid}/{version}/'
    if os.path.isdir(path): 
        groups = [x for x in os.listdir(path) if '.csv' in x]
    else:
        abort(
            404, f"{gid} not found in processed"
        )
    return groups

def read_proc_group_version_record(eid,version,gid,record):
    path = f'/var/www/html/records/proc/{eid}/{gid}/{version}/'
    if os.path.isfile(path+record+'.csv'): 
        dfS = tsi.readSessionData(path,record)
    else:
        abort(
            404, f"{gid} not found in processed"
        )
    
    print(record)
    print(dfS)
    
    return {'record':record,'dfS': dfS.to_dict('records')} #dfS.values.tolist()} #dfS.to_dict('records'),'record':record}

def plot_record(eid,version,gid,record):
    path = f'/var/www/html/records/proc/{eid}/{gid}/{version}/'
    if os.path.isfile(path+record+'.csv'): 
        dfS = tsi.readSessionData(path,record)
    else:
        abort(
            404, f"{gid} not found in processed"
        )

    print(record)
    print(dfS)
    plt,fig = tsi.makeRecordPlot(record, dfS, colName = ['posx','posy','posz','dirx','diry','dirz','fx','fy','fz'])

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)  # Close the figure to free memory

    # Return the image as a response
    return send_file(buf, mimetype='image/png')
    #fig.savefig(buf, format="png")
    ## Embed the result in the html output.
    #data = base64.b64encode(buf.getbuffer()).decode("ascii")
    #return f"<img src='data:image/png;base64,{data}'/>"


def plot_record_timeiterval(eid,version,gid,record,tstart,tend):
    path = f'/var/www/html/records/proc/{eid}/{gid}/{version}/'
    if os.path.isfile(path+record+'.csv'): 
        dfS = tsi.readSessionData(path,record)
    else:
        abort(
            404, f"{gid} not found in processed"
        )

    print(record)
    print(dfS)
    plt,fig = tsi.makeRecordPlot(record, dfS, colName = ['posx','posy','posz','dirx','diry','dirz','fx','fy','fz'],tstart=tstart,tend=tend)

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)  # Close the figure to free memory

    # Return the image as a response
    return send_file(buf, mimetype='image/png')
    #fig.savefig(buf, format="png")
    ## Embed the result in the html output.
    #data = base64.b64encode(buf.getbuffer()).decode("ascii")
    #return f"<img src='data:image/png;base64,{data}'/>"