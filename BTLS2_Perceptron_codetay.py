import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
import sys
sys.setrecursionlimit(30000)

df = pd.read_csv('winequality-red-1.csv')
data = np.array(df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']].values)
dt_Train,dt_Test = train_test_split(data,test_size = 0.3 ,shuffle = True)
X_train = dt_Train[:,:-1]
y_train = dt_Train[:,-1]
X_test = dt_Test[:,:-1]
y_test = dt_Test[:,-1]
X_train = np.insert(X_train, 0, 1, axis=1)
X_test = np.insert(X_test, 0, 1, axis=1)
W = np.array([1, 1, 1,1,1,1,1,1,1,1,1,1])
t= 0

def check ( X_train, W):
    return np.sign(np.dot(W.T,X_train))
def Stop( W,t,X_train):
    for i in range (len(X_train)):
        if (t == 5000):
            break
        y_pre = check(X_train[i].T,W)
        if( y_pre==-1):
            y_pre=0
        if ( y_pre != y_train[i]):
            W = W+check(X_train[i].T,W)*X_train[i]
            t = t+1
            return Stop( W,t,X_train)
    return W


W = Stop(W,0,X_train)
count=0
for i in range(0,len(X_test)):
        y_pre = check(X_test[i].T,W)
        if ( y_pre==-1):
             y_pre=0
        if (y_pre == y_test[i]):
             count = count+1
print("Ty le du doan dung: ",count/len(y_test))
print('Ty le du doan sai: ', 1 - (count/len(y_test)))