import app_image
from flask import Flask, request, jsonify, render_template
import json

counter = 0

def generate_json(*args):
    dict_to_json = dict()
    for attribute in args:
        attribute = tuple(attribute)
        dict_to_json[attribute[0]] = attribute[1]

    return json.dumps(dict_to_json)


def run_flask_server(host, port, debug, model, graph):
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def signup():
        return render_template('landing.html')

    @app.route('/main', methods=['GET'])
    def go_to_main():
        return render_template('main.html')

    @app.route('/process', methods=['POST'])
    def process():
        if request.method == 'POST':
            global counter
            counter += 1
            request_image_file_data = jsonify(request.form['file'])
            image = app_image.AppImage(request_image_file_data, model, graph, counter)
            classification = image.classify_image()
            message = app_image.AppImage.explain_classification(classification)
            return_json = generate_json(('class', classification.name), ('message', message))

            return return_json

    app.run(host=host, port=port, debug=debug)




