from flask import Flask, jsonify, request

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

@app.route('/measure/duration/variance/<path>', methods=['GET'])
def show_duration_varaince(path):
    pathSes = '/var/www/html/records/'+path
    # parse path to records
    if '/' == pathSes[-1]:
        pathSes = pathSes[:-1] #takeout the slash and than the folder name
        print(pathSes)
    if 'preprocessed-VR-sessions' in pathSes: 
        pathSes = pathSes[:-len('/preprocessed-VR-sessions')] 
        print(pathSes)
        ids, fileNames, dfSs, df = tsi.readData(pathSesRec)
    #print(dfSs)

    # get variables from records    
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
    
    return jsonify({'duration': totTimes, 
                    'variance': totVars})

if __name__ == '__main__':
    app.run(debug=True)