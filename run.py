from app.routes import app

if __name__ == "__main__":
    # Configure Swagger UI to use the JSON spec
    app.config['SWAGGER_UI_DOC_EXPANSION'] = 'list'
    app.config['SWAGGER_UI_JSONEDITOR'] = True
    app.config['SWAGGER_UI_URL'] = '/swagger.json'
    app.run(debug=True)