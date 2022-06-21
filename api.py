from flask import Flask, jsonify
import pickle

app = Flask(__name__)


@app.route("/")
def index():
    return "<h1>Usage: url/predict/<password>/<modelparams></h1>"

@app.route("/predict/<password>/<modelparams>")
def predict(password, modelparams):
    #TODO Import Model
    if password == "FSIS2022":
        if modelparams:
            result = modelparams
            #result = model.predict(modelparams)
        else:
            result = {"ETA":"10 Tage"}
        return jsonify(result)
    else:
        result = "Falscher Schlüssel"
        return jsonify(result)


