import pandas as pd
import numpy as np 

try:
    df = pd.read_csv('Creditcard_data.csv')
except:
    df = pd.read_csv('https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv')

X = df.iloc[:,0:30]
y = df.iloc[:,30]

cols = ['ML-Model','RandomUnderSampler','RandomOverSampler','TomekLinks','SMOTE','NearMiss']
acc_mat = pd.DataFrame(columns = cols)
models = ['xgboost','logistic','random_forest','svm','knn']
acc_mat['ML-Model'] = ['xgboost','logistic','random_forest','svm','knn']
samples = []

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
samp_techniques = ['RandomUnderSampler','RandomOverSampler','TomekLinks','SMOTE','NearMiss']

rus = RandomUnderSampler(random_state=24, replacement=True)
X_rus, y_rus = rus.fit_resample(X, y)
samples.append([X_rus,y_rus])

ros = RandomOverSampler(random_state=24)
X_ros, y_ros = ros.fit_resample(X, y)
samples.append([X_ros,y_ros])

tl = RandomOverSampler(sampling_strategy='majority')
X_tl, y_tl = ros.fit_resample(X, y)
samples.append([X_tl,y_tl])

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)
samples.append([X_smote,y_smote])

nm = NearMiss()
X_nm, y_nm = nm.fit_resample(X, y)
samples.append([X_nm,y_nm])

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

counter = 0
for model in models:
    if model == 'xgboost':
        mod = XGBClassifier()
    elif model == 'logistic':
        mod = LogisticRegression(max_iter=2000)
    elif model == 'random_forest':
        mod = RandomForestClassifier()
    elif model == 'svm':
        mod = SVC()
    elif model == 'knn':
        mod = KNeighborsClassifier(n_neighbors=4)
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(samples[i][0],samples[i][1])
        mod.fit(X_train,y_train)
        y_predict = mod.predict(X_test)
        acc_mat.at[counter,str(samp_techniques[i])] = accuracy_score(y_predict, y_test)
    
    counter += 1


#print(acc_mat)
acc_mat.to_csv('submission.csv',index=False)