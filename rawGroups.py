import os

def read_all():
    path = '/var/www/html/records/raw'
    # get variables from records    
    groups = os.listdit()
    return jsonify({'groups':groups})