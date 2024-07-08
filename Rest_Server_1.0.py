import base64
from io import BytesIO

from flask import Flask, jsonify, request
import numpy as np
from matplotlib.figure import Figure

import timeSeriesInsightToolkit as tsi

# Crea un'istanza della classe Flask e configura l'applicazione.
app = Flask(__name__)

# Configurazione di flask

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Benvenuto al server REST!'})

@app.route('/saluto', methods=['GET'])
def saluto():
    return jsonify({'message': 'Ciao, come stai?'})

@app.route('/utente/<nome>', methods=['GET'])
def saluto_utente(nome):
    return jsonify({'message': f'Ciao {nome}, benvenuto al server REST!'})

@app.route('/moltiplicazione/<int:num1>/<int:num2>', methods=['GET'])
def moltiplicazione(num1, num2):
    risultato = num1 * num2
    return jsonify({'risultato': risultato})

@app.route('/addizione', methods=['POST'])
def addizione():
    data = request.get_json()
    somma = data['numero1'] + data['numero2']
    return jsonify({'risultato': somma})


# function that measures duration
def measureDuration(path):
    pathSesRec = '/var/www/html/records/'+path

    # get variables from records    
    ids, fileNames, dfSs, df = tsi.readData(pathSesRec)
    #print(dfSs)
    paths=tsi.getPaths(ids,dfSs,['posx','posy','posz'])
    _,x,y,z = np.vstack(paths).T
    #paths = tsi.getPaths(ids,dfSs,['dirx','diry','dirz'])
    #_,u,v,w = np.vstack(paths).T

    # get durations for each record
    totTimes = []
    for path in paths:
        t,x,y,z = path.T
        totTime = t[-1]-t[0]
        totTimes.append(totTime)
    return totTimes #{'duration': totTimes,'variance': totVars}

# function that measures variance
def measureVariance(path):
    pathSesRec = '/var/www/html/records/'+path

    # get variables from records    
    ids, fileNames, dfSs, df = tsi.readData(pathSesRec)
    #print(dfSs)
    paths=tsi.getPaths(ids,dfSs,['posx','posy','posz'])
    _,x,y,z = np.vstack(paths).T
    #paths = tsi.getPaths(ids,dfSs,['dirx','diry','dirz'])
    #_,u,v,w = np.vstack(paths).T

    # get variance for each record
    totVars = []
    for path in paths:
        t,x,y,z = path.T
        totVar = np.var(x)+np.var(y)+np.var(z)
        totVars.append(totVar) 
    return totVars #{'duration': totTimes,'variance': totVars}

# function that measures duration and variance
def measureDurationVariance(path):
    return {'duration': measureDuration(path),'variance': measureVariance(path)}

# generate scatter plot for rest API call
def scatterPlot(var1,var2,th1,th2):
    # Generate the figure **without using pyplot**.
    fig = Figure(figsize=(10,10))
    axs = fig.subplots(2,2)

    var1Bins = np.linspace(0,np.max(var1)*1.1,100)
    var2Bins = np.linspace(0,np.max(var2)*1.1,100)

    axs[0,0].hist(var1,bins=var1Bins)
    axs[0,0].axvline(th1,color='gray')
    axs[0,0].set_xlabel('session time (s)')

    axs[1,1].hist(var2,bins=var2Bins)
    axs[1,1].axvline(th2,color='gray')
    axs[1,1].set_xlabel('variance')

    #plt.figure()
    axs[1,0].scatter(var1,var2)
    axs[1,0].axvline(th1,color='gray')
    axs[1,0].axhline(th2,color='gray')
    axs[1,0].set_xlabel('session time (s)')
    axs[1,0].set_ylabel('variance')
    axs[1,0].set_xlim((var1Bins[0],var1Bins[-1]))
    axs[1,0].set_ylim((var2Bins[0],var2Bins[-1]))
    return fig

# function that generates REST API to measure duration and variance
#@app.route('/measure/duration/variance/', methods=['GET'])
#def measure_duration_varaince():
#    return jsonify({'error: add group 1 and group 2':[]})

#@app.route('/measure/duration/variance/<group2>', methods=['GET'])
#def measure_duration_varaince(group2):
#    path = f'proc/{group2}/preprocessed-VR-sessions'
#    #return jsonify({'path':path})
#    return jsonify(measureDurationVariance(path))

#@app.route('/<group1>/<group2>/measures/duration/variance', methods=['GET'])
#def measure_duration_varaince(group1,group2):
#    path = f'{group1}/{group2}/preprocessed-VR-sessions'
#    #return jsonify({'path':path})
#    return jsonify(measureDurationVariance(path))

@app.route('/<group1>/<group2>/measures', methods=['GET'])
def measure(group1,group2):
    path = f'{group1}/{group2}/preprocessed-VR-sessions'
    
    filters = request.args.to_dict()
    #keys = filters.keys()
    print('filters',filters)
    
    if not ('keys' in filters): measures = measureDurationVariance(path) 
    else:
        measures = {}
        reqMeasures = filters['keys']
        print('reqMeasures',reqMeasures)
        if 'duration' in reqMeasures: measures['duration'] = measureDuration(path)
        if 'variance' in reqMeasures: measures['variance'] = measureVariance(path)

    #return jsonify({'path':path})
    return jsonify(measures)


# # scatter plot of duration and variance
# @app.route('/scatter/duration/variance/<group1>/<group2>', methods=['GET'])
# def scatter_duration_varaince(group1,group2):
#     path = f'{group1}/{group2}/preprocessed-VR-sessions'
#     totTimes = measureDuration(path)
#     totVars = measureVariance(path)

#     thTime = 35
#     thVar = 0.1 #1. #0.1 #1. #0.4 #2.5
#     fig = scatterPlot(totTimes,totVars,thTime,thVar)

#     # Save it to a temporary buffer.
#     buf = BytesIO()
#     fig.savefig(buf, format="png")
#     # Embed the result in the html output.
#     data = base64.b64encode(buf.getbuffer()).decode("ascii")
#     return f"<img src='data:image/png;base64,{data}'/>"

# scatter plot of two defined vairables
@app.route('/scatter/<var1>/<var2>/<group1>/<group2>', methods=['GET'])
def scatter_api(var1,var2,group1,group2):
    path = f'{group1}/{group2}/preprocessed-VR-sessions'
    if var1 == 'duration': var1 = measureDuration(path)
    if var2 == 'variance': var2 = measureVariance(path)

    thTime = 35
    thVar = 0.1 #1. #0.1 #1. #0.4 #2.5
    fig = scatterPlot(var1,var2,thTime,thVar)

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"


# scatter plot of two defined vairables with gatting tresholds
@app.route('/scatter/<var1>/<var2>/<float:th1>/<float:th2>/<group1>/<group2>', methods=['GET'])
def scatter_th_api(var1,var2,th1,th2,group1,group2):
    path = f'{group1}/{group2}/preprocessed-VR-sessions'
    if var1 == 'duration': var1 = measureDuration(path)
    if var2 == 'variance': var2 = measureVariance(path)

    #th1 = 35.
    #th2 = 0.1 #1. #0.1 #1. #0.4 #2.5
    fig = scatterPlot(var1,var2,th1,th2)

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

# scatter plot of two defined vairables with gatting tresholds
@app.route('/<group1>/<group2>/scatter', methods=['GET'])
def scatter_th_api(group1,group2):
    path = f'{group1}/{group2}/preprocessed-VR-sessions'
    filters = request.args.to_dict()
    print('filters',filters)

    if not ('var1' in filters):
        print('error') 
    else:
        print(filters['var1'])

    if var1 == 'duration': var1 = measureDuration(path)
    if var2 == 'variance': var2 = measureVariance(path)

    #th1 = 35.
    #th2 = 0.1 #1. #0.1 #1. #0.4 #2.5
    fig = scatterPlot(var1,var2,th1,th2)

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

if __name__ == '__main__':
    #path = 'proc/bfanini-20231026-kjtgo0m0w/preprocessed-VR-sessions'
    #result = measureDurationVariance(path)
    #print(result)
    app.run(host='192.167.233.88', port=8087),# debug=True)
