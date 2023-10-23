import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import linear_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

df = pd.read_csv('winequality-red-1.csv')
data = np.array(df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']].values)
dt_Train,dt_Test = train_test_split(data,test_size = 0.3 ,shuffle = True)
X_train = dt_Train[:,:-1]
y_train = dt_Train[:,-1]
X_test = dt_Test[:,:-1]
y_test = dt_Test[:,-1]

pla = Perceptron().fit(X_train,y_train)
y_pre = pla.predict(X_test)
count = 0
for i in range(0,len(y_pre)):
    if(y_test[i] == y_pre[i]):
        count += 1
        
print('Ty le du doan dung: ', count/len(y_pre))
print('Ty le du doan sai: ', 1 - (count/len(y_pre)))
print('Precision: ',precision_score(y_test,y_pre))
print('Recall: ',recall_score(y_pre,y_test))
print('f1_score: ',f1_score(y_test,y_pre))