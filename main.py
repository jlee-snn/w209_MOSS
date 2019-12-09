from flask import Flask
import os
import pandas as pd
#import magic
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
import requests
import lime
import sklearn
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn.datasets import fetch_20newsgroups
from lime.lime_text import LimeTextExplainer
import json
import pickle
from sklearn.externals import joblib

UPLOAD_FOLDER = '/Users/Joseph_S_Lee/Repos/test_moss/uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['txt', 'csv', 'pickle', 'pkl'])
ALLOWED_MODEL_EXTENSIONS = set(['pickle', 'pkl'])
ALLOWED_DATA_EXTENSIONS = set(['txt', 'csv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def explain_frame(df,class_names,model,idx = 100,num_features =6):
	explainer = LimeTextExplainer(class_names=class_names)
	exp = explainer.explain_instance(df["Text"].iloc[idx], model.predict_proba, num_features=num_features)
	return exp.as_list()

@app.route('/')
def upload_form():
	return render_template('dashboard.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the files part
		if 'files[]' not in request.files:
			flash('No file part')
			return redirect(request.url)
		files = request.files.getlist('files[]')
		for file in files:
			print(file)
			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_DATA_EXTENSIONS:
					df = pd.read_csv(app.config['UPLOAD_FOLDER'] + '/' + filename)
					#print(df)
				if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_MODEL_EXTENSIONS:
					model = joblib.load(app.config['UPLOAD_FOLDER'] + '/' + filename)
					#print(model)
		text_data = df['0'].values

		vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
		test_vectors = vectorizer.fit_transform(text_data)
		print(model.predict(test_vectors))


		from lime import lime_text
		from sklearn.pipeline import make_pipeline
		c = make_pipeline(vectorizer, model)  # this will need to be replaced with a pickle loader assuming model is trainined ahead of time
		class1 = [c.predict_proba([i])[0][0] for i in text_data]
		class2 = [c.predict_proba([i])[0][1] for i in text_data]

		class_names = ['atheism', 'christian']

		test_with_scores = pd.DataFrame()
		test_with_scores["Text"] = text_data
		test_with_scores[class_names[0]] = class1
		test_with_scores[class_names[1]] = class2



		#print(d3_tbl["Text"])


		test_with_scores = test_with_scores.reset_index() # give new index
		test_with_scores.to_csv("/Users/Joseph_S_Lee/Repos/test_moss/uploads/test_with_scores.csv")
		#d3_tbl = test_with_scores.to_dict(orient='records')

		d = dict((int(val), explain_frame(df = test_with_scores, class_names = class_names, model = c, idx = val, num_features = 6))
		                  for val in test_with_scores['index'].values[10:13])

		jsonoutput = json.dumps(d)
		f = open("lime_by_row_example_1_20subset.json","w")
		f.write(jsonoutput)
		f.close()



		username='jlee-snn'
		password='Bleuberry1234$'
		filename = "test_with_scores.csv"
		filename2 = "lime_by_row_example_1_20subset.json"

		content=open("/Users/Joseph_S_Lee/Repos/test_moss/uploads/test_with_scores.csv", 'r').read()
		content2=open("lime_by_row_example_1_20subset.json", 'r').read()
		r = requests.post('https://api.github.com/gists',json.dumps({'files':{filename:{"content":content}}}),auth=requests.auth.HTTPBasicAuth(username, password))
		print(r.json()['html_url'])
		r2 = requests.post('https://api.github.com/gists',json.dumps({'files':{filename2:{"content":content2}}}),auth=requests.auth.HTTPBasicAuth(username, password))
		print(r2.json()['html_url'])




		flash('File(s) successfully uploaded')
		return render_template('index.html')





if __name__ == "__main__":
	app.run()
