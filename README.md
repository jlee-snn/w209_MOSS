## w209 Visualization Project (MOSS)

## What is MOSS?

### Background:
Machine Learning (ML) practices are evolving at a rapid pace as many groups, businesses, and organizations work to develop tools to automate model training, scoring, and productionization.

Many common examples of such capabilities can be seen in H2O, DataRobot, Dataiku, and many more.

While it is only becoming easier and more streamlined to implement a variety of models, the issue with many current ML practices lie within the concept of Black box modeling.  Popular algorithms such as XGBoost and GBM still persist a bottleneck due to lack of model score interpretability and understanding.  Tree ensemble models and even Deep Learning are notorious for lack of interpretability unlike counterpart models like Logistic Regression.  Banks and the financial industry wish to adopt the most complex and accurate ML algorithms, but are unable to implement most ML models due to the financial regulations required for the documentation and explainability of each aspect of the model.

Regardless of the industry, Data Scientists experience one or more of the following issues:
Given that the model is nonlinear, which feature should I remove to improve the model score?
Does a particular feature need to be dropped, added, or engineered into another feature?
Should I just let the model decide which features based off of variable ranking?
Does the feature set make sense with the business application of the model?  Can we easily identify features that are either spurious or causing overfit?


### Project Objective:
The purpose of our visualizations will be to support and guide data scientists in answering these questions once they have generated their models so that they can gain insights into how best to improve their models.
Project Concept:
Our final project will create a visualization dashboard tool for data scientists to understand their machine learning models.  We intend to have the dashboard apply Local Interpretable Model-Agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP) to ML models and visualize the output using D3.  We will house all visualizations in Flask to ensure visualizations are dynamic and will change depending on the model/data set that is loaded in.  Our proof-of-concept models will be drawn from generic public Kaggle projects and data sets as the input for our dashboard.

We expect data scientists to use our dashboard by inputting both a data set and model.  Then, our dashboard will run the LIME/SHAP explainability methods on the models and output the model results and visualizations that allow the data scientist to explore the LIME/SHAP results of the model.  The dashboard will serve as a tool for both the data scientists to understand the data and for the data scientist to explain the results to others.


### Key Objectives:
Create test cases
Using generic public Kaggle projects we will create 3-5 trained models to be used as input into the eventual dashboard
Create LIME and/or SHAP visualizations using D3
Create 3D Cluster Visualizations using D3 (Optional)
House all visualizations into Flask
Ensure visualizations are dynamic and changes depending on the model and data set loaded


## Installation and Setup

1. Clone the repo

```
$ git clone https://github.com/jlee-snn/w209_MOSS
$ cd w209_MOSS
```

2. Initialize and activate a virtualenv:
```
$ python3 -m venv pymoss
$ source pymoss/bin/activate
```


## Build application from source

1. Install the dependencies (while in pymoss environment):
  ```
  $ pip install -r requirements.txt
  ```

2. Run the development server:
  ```
  $ python3 app.py
  ```

3. Navigate to [http://localhost:5000](http://localhost:5000)
