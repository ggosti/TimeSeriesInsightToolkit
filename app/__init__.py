from flask import Flask

# Create the Flask app instance
app = Flask(__name__)

# Import routes to ensure they are registered
from app import routes