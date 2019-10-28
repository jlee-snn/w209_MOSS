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


categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']


vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)


pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')

from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)  # this will need to be replaced with a pickle loader assuming model is trainined ahead of time


class1 = [c.predict_proba([i])[0][0] for i in newsgroups_test.data]
class2 = [c.predict_proba([i])[0][1] for i in newsgroups_test.data]
#
test_with_scores = pd.DataFrame()
test_with_scores["Text"] = newsgroups_test.data
test_with_scores[class_names[0]] = class1
test_with_scores[class_names[1]] = class2

test_with_scores = test_with_scores.reset_index() # give new index


# write to csv
test_with_scores.to_csv("test_with_scores_example_1.csv")


# EXPLAINING ONE INSTANCE, we will need a way to allow users to select each text observation and view the EXPLAINATION for it
def explain_frame(df,class_names,model,idx = 83,num_features = 6):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(df["Text"].iloc[idx], model.predict_proba, num_features=6)
    return exp.as_list()

# creating dictionary object
d = dict((int(val), explain_frame(df = test_with_scores, class_names = class_names, model = c, idx = val, num_features = 6))
                  for val in test_with_scores['index'].values[0:20])
json = json.dumps(d)
f = open("lime_by_row_example_1_20subset.json","w")
f.write(json)
f.close()
