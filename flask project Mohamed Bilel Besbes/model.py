#importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
import pickle

## importing dataset
df = pd.read_csv("data\Churn_Modelling.csv")
dfcopy = df.copy()

## data visualisation

df.head(5)
df.info()

#frequency histogram
ax = df.hist(bins=25, grid=False, figsize=(18,18), color='#1DB954', zorder=2, rwidth=0.9)

# defining the coorelation matrix
sns.heatmap(df.corr())

# other display options you can just uncomment them to put them into use

#sns.heatmap(df.corr(), vmax=.8, square=True , cmap='coolwarm')

#plt.figure(figsize=(15,7))
#sns.heatmap(df.corr(), vmin=-1, cmap='coolwarm', annot=True)
#df.corr().Exited.sort_values(ascending=False)

#frequency graph example of credit score

plt.figure(figsize=(10, 7))
df.CreditScore.plot.hist(grid=True, bins=20, rwidth=0.9)
plt.xlabel('CreditScore')
plt.grid(axis='y', alpha=0.75)

## one hot encoding process

# obtaining the categorical columns
object_cols = [i for i in dfcopy.columns if dfcopy[i].dtype not in ('float64','int64')]
object_cols

#OneHotEncoding
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dfcopy[['Geography', 'Gender']]))
num_X = dfcopy.drop(object_cols + ['RowNumber','CustomerId'], axis=1)
newdf = pd.concat([OH_cols,num_X], axis=1)
newdf.info()
newdf.columns

x = newdf.iloc[:, :13]
y = newdf.iloc[:, 13]
X_train, X_test, y_train, y_test = train_test_split(x,y, train_size=0.8 , test_size=0.2, random_state=0)

## feature scaling

#SC = StandardScaler()
#X_train_ss = SC.fit_transform(X_train)
#X_test_ss = SC.transform(X_test)

# StandardScaler gives back numpy arrays upon transformations

#X_train = pd.DataFrame(X_train_ss, index=X_train.index, columns=X_train.columns)
#X_test = pd.DataFrame(X_test_ss, index=X_test.index, columns=X_test.columns)

## Training The Model Using Classification Algorithms

# hyperparameter tuning : searching for the optimal set of hyperparameters
params = {
        'silent': [False],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [100]}
# creating a model for testing
xgb_init = XGBClassifier()
# applying RandomizedSearchCV
random_cv = RandomizedSearchCV(xgb_init, param_distributions=params, n_iter=5, scoring="roc_auc", n_jobs=1, cv=5, verbose=3)
# fitting the data to find the optimal set of hyperparameters
random_cv.fit(X_train, y_train)

# creating a model with the optimal set of hyperparameters from RandomizedSearchCV
classff = random_cv.best_estimator_
# fitting the data
classff.fit(X_train,y_train)
# predicting y hat
y_pred = classff.predict(X_test)

## evaluating the model

# measurement of accuracy
acc = 1.0 - mean_squared_error(y_test,y_pred)
acc

#confusion matrix of TP FP TN FN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

## creating the pickeled model
pickle.dump(classff,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))











