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
    groups = get_sub_dirs(path)
    return groups

def read_raw_groups(group):
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


def read_all_proc_group_versions(group):
    path = f'/var/www/html/records/proc/{group}'
    if os.path.isdir(path): 
        version = get_sub_dirs(path) #os.listdir(path) #[x for x in os.listdir(path) if '.csv' in x]
    else:
        abort(
            404, f"{group} not found in processed"
        )
    return version

def read_proc_group_version(version,group):
    path = f'/var/www/html/records/proc/{group}/{version}/'
    if os.path.isdir(path): 
        groups = [x for x in os.listdir(path) if '.csv' in x]
    else:
        abort(
            404, f"{group} not found in processed"
        )
    return groups

def read_proc_group_version_record(version,group,record):
    path = f'/var/www/html/records/proc/{group}/{version}/'
    if os.path.isfile(path+record+'.csv'): 
        dfS = tsi.readSessionData(path,record)
    else:
        abort(
            404, f"{group} not found in processed"
        )
    
    print(record)
    print(dfS)
    
    return {'record':record,'dfS': dfS.to_dict('records')} #dfS.values.tolist()} #dfS.to_dict('records'),'record':record}

def plot_record(version,group,record):
    path = f'/var/www/html/records/proc/{group}/{version}/'
    if os.path.isfile(path+record+'.csv'): 
        dfS = tsi.readSessionData(path,record)
    else:
        abort(
            404, f"{group} not found in processed"
        )

    print(record)
    print(dfS)
    plt,fig = tsi.makeRecordPlot(record, dfS)

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
