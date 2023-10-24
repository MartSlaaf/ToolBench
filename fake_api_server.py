from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Example JSON configuration with API names, parameters, and dummy execution results
api_config = [{"name": "ls",
               "url": "ls",
               "description": "Lists a directory content. Without any parameters given, lists the content of current directory.",
               "method": "POST",
               "required_parameters": [],
               "optional_parameters": [
                    {"name": "addr", 
                    "type": "STRING", 
                    "description": "Address of a directory to list. Defaults to the current directory",
                    "default": "."},]},
              {"name": "linenum",
               "url": "linenum",
               "description": "Counts number of lines in files. Basically is a result of execution of `cat filename | wc -l`",
               "method": "POST",
               "required_parameters": [
                   {"name": "addr", 
                    "type": "STRING", 
                    "description": "Address of a directory to list. Defaults to the current directory"},],
               "optional_parameters": []}
        ]

import os
def ls(addr='.'):
    observation = str(os.listdir(addr))
    return observation

def linenum(addr):
    observation = str(os.system(f"cat {addr} | wc -l"))
    return observation

methods = {'ls': ls, 'linenum': linenum}

@app.route('/methods', methods=['GET'])
def list_methods():
    return jsonify(api_config)

@app.route('/<method>', methods=['POST'])
def call_method(method):
    """
    Endpoint to call a given method with provided parameters.
    """
    if method not in methods.keys():
        return jsonify({"error": "Method not found"}), 404

    # expected_params = set(api_config["methods"]['method']["params"])
    # received_params = set(request.json.keys())

    # if expected_params != received_params:
    #     return jsonify({"error": "Invalid parameters"}), 400

    try:
        observation = methods[method](**request.json)
    except Exception as e:
        observation = str(e)

    return jsonify(observation)

if __name__ == '__main__':
    app.run(debug=True, port=5000)