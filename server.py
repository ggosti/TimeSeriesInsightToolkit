from flask import Flask, jsonify, render_template, jsonify, request #send_from_directory
from connexion import FlaskApp
#import connexion
#from flask_restx import Api, Resource, fields
from models import Project, Group, Record
from utils import load_data

import numpy as np
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

# Example: In-memory data structure
steps = ['raw', 'proc']#
projects = []
groups = []
records = []

for step in steps:  
    projects,groups,records = load_data(f"./test/records/{step}/",projects,groups,records)
    #print('groups', step, [g.name for g in groups])

def get_steps():
    """Get all steps"""
    return jsonify([s for s in steps])

def get_projects():
    """Get all projects"""
    return jsonify([{"id": p.id, "name": p.name, "step": p.step} for p in projects])

def get_groups():
    """Get all groups"""
    groups = []
    for project in projects:
        groups = groups + [{"id": g.id, "name": g.name, "group": g.project.name, "step": project.step} for g in project.groups] 
    #groups =  [{"id": g.id, "name": g.name, "group": g.parent_group, "step": project.parent_step} for g in groups]
    return jsonify(groups)

def get_records():
    """Get all records"""
    records = []
    for project in projects:
        for group in project.groups:
            records = records + [{"id": r.id, "name": r.name, "version":r.version, 
                                  "group": r.group.name, "project": project.name, "step": project.step} for r in group.records] 
    #groups =  [{"id": g.id, "name": g.name, "group": g.parent_group, "step": project.parent_step} for g in groups]
    return jsonify(records)

# Crea un'istanza della classe Flask e configura l'applicazione.
#app = Flask(__name__)
# Flask class instace with added functionalities
flask_app  = FlaskApp(__name__, specification_dir="./")    #connexion.App(__name__, specification_dir="./")
# Load the Swagger specification
flask_app.add_api("swagger.yml")

# Get underlying Flask app
server = flask_app.app


@flask_app.route("/")
def home():
    return render_template("home.html")



# Generate sample data function
def generate_data(n=100, seed=42):
    np.random.seed(seed)
    return pd.DataFrame({
        "x": np.random.uniform(0, 10, n),
        "y": np.random.uniform(0, 10, n)
    })

# Route to create a Dash app dynamically
@flask_app.route("/dash/<dataset_id>/")
def serve_dash(dataset_id):
    """Dynamically create a Dash app based on the dataset_id in the URL."""
    print(dataset_id)
    # data = generate_data(n=100, seed=int(dataset_id))  # Change data per dataset_id
    
    # Create a new Dash app instance
    dash_app = dash.Dash(
        __name__, server=server, routes_pathname_prefix=f"/dash/{dataset_id}/"
    )

    # # Define layout
    # dash_app.layout = html.Div([
    #     dcc.Graph(id="scatter-plot"),
    #     html.Label("Vertical Line (X filter)"),
    #     dcc.Slider(id="x-slider", min=0, max=10, step=0.1, value=5),
    #     html.Label("Horizontal Line (Y filter)"),
    #     dcc.Slider(id="y-slider", min=0, max=10, step=0.1, value=5),
    # ])

    # # Callback to update the scatterplot
    # @dash_app.callback(
    #     Output("scatter-plot", "figure"),
    #     [Input("x-slider", "value"), Input("y-slider", "value")]
    # )
    # def update_plot(x_filter, y_filter):
    #     fig = px.scatter(data, x="x", y="y", title=f"Scatterplot for Dataset {dataset_id}")
        
    #     # Add filtering lines
    #     fig.add_shape(type="line", x0=x_filter, x1=x_filter, y0=0, y1=10, line=dict(color="red", width=2))
    #     fig.add_shape(type="line", x0=0, x1=10, y0=y_filter, y1=y_filter, line=dict(color="blue", width=2))
        
    #     return fig

    return dash_app.index()

# # Scatter plot of recods with data filtering
# df = pd.read_csv("./test/records/raw/event1/group1/U2.csv")
# print(df.columns)
# print(df[" posx"])

# # Dynamic route prefix function
# def get_prefix():
#     """Extracts the dynamic prefix from the request."""
#     path = request.path
#     parts = path.split("/")
#     if len(parts) > 2:
#         return f"/{parts[1]}/dash/"
#     return "/dash/"

# # Create Dash app with dynamic route prefix
# with server.app_context():
#     dynamic_prefix = get_prefix()
#     dashApp = dash.Dash(__name__, server=server, routes_pathname_prefix=dynamic_prefix)

# dashApp.layout = html.Div([
#     dcc.Graph(id="scatter-plot"),
#     html.Label("Vertical Line (X filter)"),
#     dcc.Slider(id="x-slider", min=0, max=10, step=0.1, value=5),
#     html.Label("Horizontal Line (Y filter)"),
#     dcc.Slider(id="y-slider", min=0, max=10, step=0.1, value=5),
# ])

# @dashApp.callback(
#     Output("scatter-plot", "figure"),
#     [Input("x-slider", "value"), Input("y-slider", "value")]
# )
# def update_plot(x_filter, y_filter):
#     fig = px.scatter(x=df["time"], y=df[" posx"], title="Scatterplot with Filters")
    
#     # Add filtering lines
#     fig.add_shape(type="line", x0=x_filter, x1=x_filter, y0=0, y1=10, line=dict(color="red", width=2))
#     fig.add_shape(type="line", x0=0, x1=10, y0=y_filter, y1=y_filter, line=dict(color="blue", width=2))
    
#     return fig



if __name__ == '__main__':
    flask_app.run(port=8087, debug=True) #(host='192.167.233.88', port=8087, debug=True)

