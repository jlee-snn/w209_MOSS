import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
import json
import plotly.graph_objects as go
import plotly
import plotly.graph_objs as go
from sklearn.datasets import fetch_20newsgroups
import pickle
import pandas as pd
from sklearn.externals import joblib

categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']


vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)

print(rf.feature_importances_)
print(vectorizer.vocabulary_.keys())
var_frame = pd.DataFrame()
var_frame["Feature"] = vectorizer.vocabulary_.keys()
var_frame["Weight"] = rf.feature_importances_
var_frame = var_frame.sort_values("Weight",ascending = False).head(20).reset_index(drop = True)
print(var_frame)

fig = go.Figure(go.Bar(
    y=var_frame["Feature"],
    x=var_frame["Weight"],
    orientation='h'))

graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


#plot = graphJSON
#plot = create_plot()


pred = rf.predict(test_vectors)
print(sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary'))


# Create test data set for app usage
example_df = pd.DataFrame()
example_df["text"] = newsgroups_test.data
example_df["target"] = newsgroups_test.target
example_df = example_df.reset_index()

example_df.to_csv("./data/news_group_test_example.csv")

filename = './data/rf_model.pkl'
pickle.dump(rf, open(filename, 'wb'))

filename2 = './data/vectorizer.pkl'
pickle.dump(vectorizer, open(filename2, 'wb'))


df = pd.read_csv("./data/news_group_test_example.csv")

model2 = joblib.load('./data/rf_model.pkl')
vectorizer2= joblib.load('./data/vectorizer.pkl')

test_vectors2 = vectorizer2.transform(df["text"].values)

print(model2.predict_proba(test_vectors2))
print(df)
print(model2)
