from flask import Flask, render_template, request, session, url_for, Response
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import string
import pygal
from werkzeug.utils import redirect
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from random import randint
import time
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def registration():
    return render_template('uploaddataset.html')

@app.route('/uploaddataset', methods=["POST", "GET"])
def uploaddataset_csv_submitted():
    if request.method == "POST":
        csvfile = request.files['csvfile']
        result = csvfile.filename
        file = "D:/datasets/moneylaunder/" + result
        print(file)
        session['filepath'] = file
        return render_template('uploaddataset.html', msg='sucess')
    return render_template('uploaddataset.html')

@app.route('/viewdata', methods=["POST", "GET"])
def viewdata():
    session_var_value = session.get('filepath')
    df = pd.read_csv(session_var_value)
    x = pd.DataFrame(df)
    return render_template("view.html", data=x.to_html(index=False),rows=x.shape[0])

@app.route('/preprocessing', methods=["POST", "GET"])
def preprocessing():
    global X,y
    session_var_value = session.get('filepath')
    df = pd.read_csv(session_var_value)
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates()
    df = df.drop(['Id','Name','Mail','Services','Url','Location','Activedays','FollowerId','Time'],axis=1)
    le = LabelEncoder()
    df['Status'] = le.fit_transform(df['Status'])
    df.to_csv("D:/datasets/moneylaunder/preprocess.csv",index=False)
    X = df.drop(['Type'], axis=1)
    y = df['Type']
    x = pd.DataFrame(df)
    return render_template("preprocessing.html", data=df.to_html(index=False),rows=df.shape[0])

@app.route('/training', methods=["POST", "GET"])
def training():
    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return render_template("modelperformance.html")

@app.route('/modelperformance', methods=["POST", "GET"])
def selected_model_submitted():
    global accuracyscore,precision,recall
    if request.method == "POST":
        selectedalg = int(request.form['algorithm'])
        if (selectedalg == 1):
            model = AdaBoostClassifier()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracyscore = accuracy_score(y_test, y_pred)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="DecisionTree")

        if (selectedalg == 2):
            model = RandomForestClassifier(n_estimators=20)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracyscore = accuracy_score(y_test, y_pred)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="RandomForest")

        if (selectedalg == 3):
            model = GaussianNB()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracyscore = accuracy_score(y_test, y_pred)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="NaiveBayes")

        if (selectedalg == 4):
            model = SVC(kernel='rbf')
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracyscore = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="SVM")
    return render_template('modelperformance.html')


@app.route('/testing', methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        list1 = []
        status = request.form['status']
        friends = request.form['friends']
        recharge = request.form['recharge']
        banks = request.form['banks']
        gifts = request.form['gifts']
        followers = request.form['followers']
        followee = request.form['followee']

        totalexp=100
        list1.extend([status,friends,recharge,totalexp,banks,gifts,followers,followee])

        model = SVC()
        model.fit(x_train, y_train)
        predi = model.predict([list1])
        print(predi)
        return render_template('prediction.html', msg='Prediction Success', predvalue=predi)
    return render_template('prediction.html')

@app.route('/graph',methods=['POST','GET'])
def graph():
    line_chart = pygal.Bar()
    line_chart.title = 'METRICS:: ACCURACY, PRECISION & RECALL'
    line_chart.add('Accuracy', accuracyscore)
    line_chart.add('Precision', precision)
    line_chart.add('Recall', recall)

    graph_data = line_chart.render_data_uri()
    return render_template("graph.html", graph_data=graph_data)


if __name__ == '__main__':
    app.secret_key = ".."
    app.run()