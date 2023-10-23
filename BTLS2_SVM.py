import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’}
df = pd.read_csv('winequality-red-1.csv')
data = np.array(df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']].values)
dt_Train,dt_Test = train_test_split(data,test_size = 0.3 ,shuffle = True)
X_train = dt_Train[:,:-1]
y_train = dt_Train[:,-1]
X_test = dt_Test[:,:-1]
y_test = dt_Test[:,-1]


svm = make_pipeline(StandardScaler(), SVC(kernel='poly',gamma='auto')).fit(X_train,y_train)
y_pre = svm.predict(X_test)
count = 0
for i in range(0,len(y_pre)):
    if(y_pre[i] == y_test[i]):
        count += 1

print('Ty le du doan dung: ', count/len(y_pre))
print('Ty le du doan sai: ', 1 - (count/len(y_pre)))
print('Precision_micro: ',precision_score(y_test,y_pre,average='micro'))
print('Precision_macro: ',precision_score(y_test,y_pre,average='macro'))
print('Recall_micro: ',recall_score(y_pre,y_test,average='micro'))
print('Recall_macro: ',recall_score(y_pre,y_test,average='macro'))
print('f1_score_micro: ',f1_score(y_test,y_pre,average='micro'))
print('f1_score_macro: ',f1_score(y_test,y_pre,average='macro'))







