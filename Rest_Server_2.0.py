from datetime import timedelta
from functools import wraps

from flask import Response
from flask import Flask
from flask import jsonify
from flask import request

from flask_jwt_extended import create_access_token, verify_jwt_in_request, get_jwt
from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import jwt_required
from flask_jwt_extended import JWTManager

# Crea un'istanza della classe Flask e configura l'applicazione.
app = Flask(__name__)
# Configurazione di flask

# Configura la chiave segreta per JWT
app.config['JWT_SECRET_KEY'] = 'la_tua_chiave_segreta_per_jwt'
jwt = JWTManager(app)

# Mappatura degli utenti fittizi con username e password
users = {
    'utente1': 'password1', # admin
    'utente2': 'password2'
}

# Here is a custom decorator that verifies the JWT is present in the request,
# as well as insuring that the JWT has a claim indicating that this user is
# an admin
def admin_required():
    def wrapper(fn):
        @wraps(fn)
        def decorator(*args, **kwargs):
            verify_jwt_in_request()
            claims = get_jwt()
            if claims["role"] == "admin":
                return fn(*args, **kwargs)
            else:
                return jsonify(msg="Admins only!"), 403

        return decorator

    return wrapper


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
    try:
        data = request.get_json()
        somma = data['numero1'] + data['numero2']
        return jsonify({'risultato': somma})
    except Exception as ex:
        print(ex)
        return Response("Internal Server Error", status=500)
    finally:
        pass

# Create a route to authenticate your users and return JWTs. The
# create_access_token() function is used to actually generate the JWT.
@app.route('/login', methods=['POST'])
def login():
    try:
        username = request.json.get('username', None)
        password = request.json.get('password', None)

        # Verifica le credenziali
        if username in users and users[username] == password:
            additional_claims = None
            if username == list(users.keys())[0]:
                # Crea il token di accesso.
                additional_claims = {"aud": "some_audience", "role": "admin"}
            else:
                additional_claims = {"aud": "some_audience", "role": "user"}

            access_token = create_access_token(identity=username, expires_delta=timedelta(days=3), additional_claims=additional_claims)
            return jsonify(access_token=access_token), 200
        else:
            return jsonify({"msg": "Credenziali non valide"}), 401
    except Exception as ex:
        print(ex)
        return Response("Internal Server Error", status=500)

# not protected
@app.route('/', methods=['GET'])
@jwt_required(optional=True)
def home():
    current_user = get_jwt_identity()
    if current_user:
        return jsonify({'message': 'Benvenuto al server REST!' + current_user})
    else:
        return jsonify(logged_in_as="anonymous user")

@app.route('/protected', methods=['GET'])
@jwt_required()  # use this after route
def protected():
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200

@app.route('/admin', methods=['GET'])
@admin_required()
def admin():
    current_user = get_jwt_identity()
    return jsonify("Admin ==>" + current_user), 200


if __name__ == '__main__':
    app.run(debug=True)