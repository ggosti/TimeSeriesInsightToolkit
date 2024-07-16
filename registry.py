# rawGroups.py

#import base64
from io import BytesIO
import os
from flask import abort, send_file, jsonify
import matplotlib.pyplot as plt
from PIL import Image

import timeSeriesInsightToolkit as tsi

def load_eps_to_buffer(file_path):
    # Open the EPS file
    with Image.open(file_path) as img:
        # Create an in-memory bytes buffer
        buffer = BytesIO()
        
        # Save the image to the buffer in a different format (e.g., PNG)
        img.save(buffer, format='PNG')
        
        # Move the cursor to the beginning of the buffer
        buffer.seek(0)
    
    return buffer

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

def read_raw_group_version_record_columns(eid,gid,record,columns):
    path = f'/var/www/html/records/raw/{eid}/{gid}/'
    print('columns',columns)
    if os.path.isfile(path+record+'.csv'): 
        dfS = tsi.readSessionData(path,record)
        filtered_data = dfS[['time']+columns].to_dict(orient='list')
    else:
        abort(
            404, f"{gid} not found in processed"
        )
    
    return jsonify(filtered_data)
    #print(record)
    #print(dfS)
    #
    #return {'record':record,'dfS': dfS.to_dict('records')} #dfS.values.tolist()} #dfS.to_dict('records'),'record':record}

def raw_record_prev(eid,gid,record):
    path = f'/var/www/html/records/raw/{eid}/{gid}/'
    #if os.path.isfile(path+record+'.csv'):
    if os.path.isfile(path+record+'-prev.png'):
            buf = load_eps_to_buffer(path+record+'-prev.png')
    else:
        abort(
            404, f"{record} preview not found in processed"
        )

    # Return the image as a response
    return send_file(buf, mimetype='image/png')

def raw_record_prev_create(eid,gid,record):
    path = f'/var/www/html/records/proc/{eid}/{gid}/'
    if os.path.isfile(path+record+'.csv'):
        #if os.path.isfile(path+record+'-prev.png'):
        print('Generate file')
        dfS = tsi.readSessionData(path,record)
        ppath = tsi.getPath(dfS,listCols = ['posx','posy','posz'])
        dpath = tsi.getPath(dfS,listCols = ['dirx','diry','dirz'])
        print(record)
        print(dfS)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')
        ax,sc = tsi.drawPath(path=ppath,dpath=dpath,BBox=None,ax=ax)
        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        fig.savefig(path+record+'-prev.png', transparent=True)
    else:
        abort(
            404, f"{record} csv not found in processed"
        )

    buf = load_eps_to_buffer(path+record+'-prev.png')

    # Return the image as a response
    return send_file(buf, mimetype='image/png')

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
        records = [x for x in os.listdir(path) if '.csv' in x]
    else:
        abort(
            404, f"{gid} not found in processed"
        )
    return records

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
    
    return jsonify(dfS.to_dict(orient='records')) #dfS.values.tolist()} #dfS.to_dict('records'),'record':record}

def read_proc_group_version_record_columns(eid,version,gid,record,columns):
    path = f'/var/www/html/records/proc/{eid}/{gid}/{version}/'
    print('columns',columns)
    if os.path.isfile(path+record+'.csv'): 
        dfS = tsi.readSessionData(path,record)
        filtered_data = dfS[['time']+columns].to_dict(orient='list')
    else:
        abort(
            404, f"{gid} not found in processed"
        )
    
    return jsonify(filtered_data)
    #print(record)
    #print(dfS)
    #
    #return {'record':record,'dfS': dfS.to_dict('records')} #dfS.values.tolist()} #dfS.to_dict('records'),'record':record}

def proc_record_prev(eid,version,gid,record):
    path = f'/var/www/html/records/proc/{eid}/{gid}/{version}/'
    #if os.path.isfile(path+record+'.csv'):
    if os.path.isfile(path+record+'-prev.png'):
            buf = load_eps_to_buffer(path+record+'-prev.png')
    else:
        abort(
            404, f"{record} png not found in processed"
        )

    # Return the image as a response
    return send_file(buf, mimetype='image/png')

def proc_record_prev_create(eid,version,gid,record):
    path = f'/var/www/html/records/proc/{eid}/{gid}/{version}/'
    if os.path.isfile(path+record+'.csv'):
        #if os.path.isfile(path+record+'-prev.png'):
        print('Generate file')
        dfS = tsi.readSessionData(path,record)
        ppath = tsi.getPath(dfS,listCols = ['posx','posy','posz'])
        dpath = tsi.getPath(dfS,listCols = ['dirx','diry','dirz'])
        print(record)
        print(dfS)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')
        ax,sc = tsi.drawPath(path=ppath,dpath=dpath,BBox=None,ax=ax)
        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        fig.savefig(path+record+'-prev.png', transparent=True)
    else:
        abort(
            404, f"{record} csv not found in processed"
        )

    buf = load_eps_to_buffer(path+record+'-prev.png')

    # Return the image as a response
    return send_file(buf, mimetype='image/png')

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