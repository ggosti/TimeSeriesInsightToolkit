import os, sys
sys.path.insert(0, os.path.abspath(".."))

from flask import Flask, jsonify, send_from_directory
from flask_restx import Api
from app.models import Project, Group, Record
from app.utils import load_data

app = Flask(__name__)
api = Api(
    app,
    doc='/docs',
    title = "Project Explorer API",
    description = "API for exploring projects, groups, and records.",
    version = "1.0.0"
)  # Swagger UI available at /docs

# Load and serve Swagger JSON
@app.route('/swagger.json')
def swagger_spec():
    """Serve the Swagger JSON file."""
    return send_from_directory('app/', 'swagger.json')

# Example: In-memory data structure
projects = load_data("./test/records/raw/")

@app.route('/projects', methods=['GET'])
def get_projects():
    """Get all projects"""
    return jsonify([{"id": p.id, "name": p.name} for p in projects])

@app.route('/projects/<int:project_id>/groups', methods=['GET'])
def get_groups(project_id):
    """Get all groups in a project"""
    project = next((p for p in projects if p.project_id == project_id), None)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    return jsonify([{"id": g.group_id, "name": g.name} for g in project.groups])

@app.route('/projects/<int:project_id>/groups/<int:group_id>/records', methods=['GET'])
def get_records(project_id, group_id):
    """Get all records in a group"""
    project = next((p for p in projects if p.project_id == project_id), None)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    group = next((g for g in project.groups if g.group_id == group_id), None)
    if not group:
        return jsonify({"error": "Group not found"}), 404
    return jsonify([{"id": r.record_id, "data": r.data} for r in group.records])

if __name__ == "__main__":
    print(get_projects())


