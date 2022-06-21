from flask import Flask, jsonify
import pickle

app = Flask(__name__)

@app.route("/predict/<modelparams>")
def predict(modelparams):
    #TODO Import Model
    if modelparams:
        result = modelparams
        #result = model.predict(modelparams)
    else:
        result = {"ETA":"10 Tage"}
    return jsonify(result)

if __name__ == "__main__":
    app.run() 