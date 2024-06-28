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


# function that measures duration and variance
def measureDurationVariance(path):
    pathSesRec = '/var/www/html/records/'+path

    # parse path to records
    #if '/' == pathSes[-1]:
    #    pathSes = pathSes[:-1] #takeout the slash and than the folder name
    #    print(pathSes)
    #if 'preprocessed-VR-sessions' in pathSes: 
    #    pathSes = pathSes[:-len('/preprocessed-VR-sessions')] 
    #    print(pathSes)
    # get variables from records    
    ids, fileNames, dfSs, df = tsi.readData(pathSesRec)
    #print(dfSs)
    paths=tsi.getPaths(ids,dfSs,['posx','posy','posz'])
    _,x,y,z = np.vstack(paths).T
    #paths = tsi.getPaths(ids,dfSs,['dirx','diry','dirz'])
    #_,u,v,w = np.vstack(paths).T

    # get durations and variance for each record
    totTimes = []
    totVars = []
    for path in paths:
        t,x,y,z = path.T
        totTime = t[-1]-t[0]
        totTimes.append(totTime)
        totVar = np.var(x)+np.var(y)+np.var(z)
        totVars.append(totVar) 
    return {'duration': totTimes,'variance': totVars}

# function that generates REST API to measure duration and variance
#@app.route('/measure/duration/variance/', methods=['GET'])
#def measure_duration_varaince():
#    return jsonify({'error: add group 1 and group 2':[]})

#@app.route('/measure/duration/variance/<group2>', methods=['GET'])
#def measure_duration_varaince(group2):
#    path = f'proc/{group2}/preprocessed-VR-sessions'
#    #return jsonify({'path':path})
#    return jsonify(measureDurationVariance(path))

@app.route('/measure/duration/variance/<group1>/<group2>', methods=['GET'])
def measure_duration_varaince(group1,group2):
    path = f'{group1}/{group2}/preprocessed-VR-sessions'
    #return jsonify({'path':path})
    return jsonify(measureDurationVariance(path))

@app.route('/scatter/duration/variance/<group1>/<group2>', methods=['GET'])
def scatter_duration_varaince(group1,group2):
    path = f'{group1}/{group2}/preprocessed-VR-sessions'
    #return jsonify({'path':path})
    dictDurVar = measureDurationVariance(path)
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2])
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
    app.run(debug=True)
