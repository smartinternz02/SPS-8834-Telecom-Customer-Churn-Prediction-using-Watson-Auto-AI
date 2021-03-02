import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_pred',methods=['POST'])
def y_pred():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    numerical_indices = (0 , 1 , 3 , 4 , 5 , 6 , 7 , 9 , 10 , 11)
    floatable_indices = (8 , 12)
    for i in numerical_indices:
        x_test[0][i] = int(x_test[0][i])
    for i in floatable_indices:
        x_test[0][i] = float(x_test[0][i])
    geo = [0.0 , 0.0 , 0.0]
    geo[x_test[0][4]] = 1.0
    gender = [0.0 , 0.0]
    gender[x_test[0][5] - 3] = 1.0
    d = {'0': [geo[0]],
    '1': [geo[1]],
    '2': [geo[2]],
    '3': [gender[0]],
    '4': [gender[1]],
    'CreditScore': [x_test[0][5]],
    'Age': [x_test[0][6]],
    'Tenure': [x_test[0][7]],
    'Balance': [x_test[0][8]],
    'NumOfProducts': [x_test[0][9]],
    'HasCrCard': [x_test[0][10]],
    'IsActiveMember': [x_test[0][11]],
    'EstimatedSalary': [x_test[0][12]]}
    #columns = ['0', '1', '2', '3', '4', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    df = pd.DataFrame(data=d)
    #df = pd.DataFrame (x_test,columns=['RowNumber','CustomerId','Surname','CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary'])
    print (df)
    # start
    dfcopy = df.copy()
    #SC = StandardScaler()
    #X_test_ss = SC.fit_transform(dfcopy)
    #X_testing = pd.DataFrame(X_test_ss, index=df.index, columns=dfcopy.columns)
    # end
    prediction = model.predict(dfcopy)
    print("tadaaaa")
    print(prediction)
    output=prediction[0]
    if output == 0:
        textoutput = 'withdraw'
    elif output == 1:
        textoutput = 'maintain'
    return render_template('index.html', 
  prediction_text=
  '{} is going to {} subscription'.format(x_test[0][2],textoutput))
if __name__ == "__main__":
    app.run(debug=True)

"""
@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)
    """