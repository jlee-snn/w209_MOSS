from flask import Flask, render_template
app = Flask(__name__)
import pandas as pd
import os

APP_FOLDER = os.path.dirname(os.path.realpath(__file__))

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/getData")
def getData():
    results = pd.read_csv("static/data/test_with_scores_example_1.csv")
    return results.to_json(orient='records')