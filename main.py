from flask import Flask
import os
import pandas as pd
#import magic
from flask import jsonify
import plotly.graph_objects as go
import plotly
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score
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
from flask import Flask, jsonify, request, render_template
import lime
#UPLOAD_FOLDER = '/Users/richardryu/desktop/can/MIDS/w209_MOSS/uploads'
UPLOAD_FOLDER = '/Users/Joseph_S_Lee/Repos/test_moss/uploads'

pd.set_option('display.max_colwidth', -1)

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def create_plot():


    N = 40
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


master_dict ={
   "1":[
      [
         "morality",
         0.04709557582402992
      ],
      [
         "alt",
         -0.022875682529886858
      ],
      [
         "religious",
         -0.016390207571827384
      ],
      [
         "don",
         -0.01374229788492519
      ],
      [
         "an",
         -0.011742561631778337
      ],
      [
         "atheist",
         -0.011058579649537627
      ]
   ],
   "2":[
      [
         "ethics",
         -0.767467
      ],
      [
         "revival",
         -0.0758
      ],
      [
         "bob",
         -0.4567
      ],
      [
         "don",
         -0.0146
      ],
      [
         "andover",
         -0.75748
      ],
      [
         "christ",
         -0.0164
      ]
   ]
}


first_row_example = master_dict.get("1")
first_row_df = pd.DataFrame(first_row_example)
first_row_df.columns = ["Word","Weight"]
print(first_row_df)





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
	return render_template('dashboard.html', text = "None")

@app.route("/api")
def api_info():
	if request.method == 'POST':
		print('Incoming..')
		print(request.get_json())  # parse as JSON
		return(request.get_json())
	else:
		message = {'greeting':'Hello from Flask!'}
		return jsonify(message)  # serialize and use JSON headers

@app.route("/", methods=['GET'])
def dashboard_update():
    return render_template("dashboard.html", text = "Blah")

@app.route('/get_doc_id',methods=['POST'])
def get_doc_id():
    data = request.get_json()
    x = int(data.get('docID'))
    print(type(x))
    string_x = str(x)

    return jsonify(string_x)

    #docID = data.get('docID')
    #return redirect('upload.html')

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
					#df = pd.read_csv(app.config['UPLOAD_FOLDER'] + '/' + filename)
					df = pd.read_csv("./data/news_group_test_example.csv")
					#print(df)
				if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_MODEL_EXTENSIONS:
					#model = joblib.load(app.config['UPLOAD_FOLDER'] + '/' + filename)
					model = joblib.load('./data/rf_model.pkl')
					#vectorizer= joblib.load('./data/vectorizer.pkl')
					#print(model)


		vectorizer= joblib.load('./data/vectorizer.pkl')

		test_vectors = vectorizer.transform(df["text"].values)

		print(model.predict_proba(test_vectors))
		print(df)
		print(model)

		text_data = df['text'].values
		target_data = df['target'].values
		print(target_data)
		#vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
		test_vectors = vectorizer.transform(text_data)
		model_name = type(model).__name__
		print(model_name)

		print("PArt1")
		from lime import lime_text
		from sklearn.pipeline import make_pipeline
		#c = make_pipeline(vectorizer, model)  # this will need to be replaced with a pickle loader assuming model is trainined ahead of time
		class1 = model.predict_proba(test_vectors)[0][0]
		class2 = model.predict_proba(test_vectors)[0][1]

		#pred = [c.predict([i]) for i in test_vectors]
		predictions_test = model.predict(test_vectors)
		predictions_proba_test = model.predict_proba(test_vectors)

		print(predictions_proba_test)
		print("Part2")

		from sklearn.metrics import accuracy_score
		acc_score = accuracy_score(target_data, predictions_test)

		from sklearn.metrics import precision_score
		prec_score = precision_score(target_data, predictions_test, average='macro')

		#print(accuracy_score(target_data, predictions_test, normalize=False))
		#print(predictions_test)
		from sklearn.metrics import recall_score
		recall_score = recall_score(target_data, predictions_test, average='macro')
		class_names = ['Negative Class', 'Positive Class']

		test_with_scores = pd.DataFrame()
		test_with_scores["Text"] = text_data
		test_with_scores[class_names[0]] = class1
		test_with_scores[class_names[1]] = class2
		test_with_scores["True_Label"] = target_data
		test_with_scores["Predicted_Label"] = predictions_test
		test_with_scores = test_with_scores.iloc[0:5] # GLobal cutoff




		#print(d3_tbl["Text"])




		#test_with_scores = test_with_scores.reset_index() # give new index
		test_with_scores.to_csv("./uploads/test_with_scores.csv")


		print(test_with_scores.head())
		#d3_tbl = test_with_scores.to_dict(orient='records')

		c = make_pipeline(vectorizer, model)

		d = dict((int(val), explain_frame(df = test_with_scores, class_names = class_names, model = c, idx = val, num_features = 6))
		                  for val in test_with_scores.index[0:5])

		jsonoutput = json.dumps(d)
		f = open("lime_by_row_example_1_20subset.json","w")
		f.write(jsonoutput)
		f.close()

		filename = "test_with_scores.csv"
		filename2 = "lime_by_row_example_1_20subset.json"

		content=open("./uploads/test_with_scores.csv", 'r').read()
		content2=open("lime_by_row_example_1_20subset.json", 'r').read()

		html_object = test_with_scores.to_html(table_id="example",classes = "display")
		clean_html = html_object.replace("\n","")
		#clean_html = clean_html.replace("\\[.*?\\]","")

		#chart_data = json.dumps(master_dict, indent=2)
		#data = {'chart_data': chart_data}

		# First rows
		var_frame = pd.DataFrame()
		var_frame["Feature"] = vectorizer.vocabulary_.keys()
		var_frame["Weight"] = model.feature_importances_
		var_frame = var_frame.sort_values("Weight",ascending = False).head(20).reset_index(drop = True)
		print(var_frame)

		#first_row_example = master_dict.get("1")

		fig = go.Figure(go.Bar(
            y=var_frame["Feature"],
            x=var_frame["Weight"],
            orientation='h'))

		graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


		plot = graphJSON
		#plot = create_plot()

		#print(data)
		flash('File(s) successfully uploaded')
		return render_template('dashboard.html',text=str(model_name),accuracy= str(int(100*acc_score)) + " %",precision= str(int(100*prec_score)) + " %",recall= str(int(100*recall_score)) + " %",
		tables=[clean_html],plot=plot,lime_data = jsonoutput)



@app.route('/receivedata', methods=['POST'])
def receive_data():
    print(request.form['doc_id'])



if __name__ == "__main__":
	app.run()
