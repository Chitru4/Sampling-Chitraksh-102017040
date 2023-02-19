import pandas as pd
import numpy as np 

## Reading data

try:
    df = pd.read_csv('Creditcard_data.csv')
except:
    df = pd.read_csv('https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv')

X = df.iloc[:,0:30]
y = df.iloc[:,30]

cols = ['ML-Model','SimpleRandom','Stratified','Cluster','Systematic','Reservoir']
acc_mat = pd.DataFrame(columns = cols)
models = ['xgboost','logistic','random_forest','svm','knn']
acc_mat['ML-Model'] = ['xgboost','logistic','random_forest','svm','knn']
samples = []
samp_techniques = ['SimpleRandom','Stratified','Cluster','Systematic','Reservoir']

## Balancing data


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=24)
X, y = ros.fit_resample(X, y)


## Sampling data

from sklearn.model_selection import train_test_split

n = int(((1.96**2)*(0.5**2))//(0.05**2))



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
        if i == 0:
            # Simple Random sampling
            X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = n/1526)
        else:
            # Stratified sampling
            X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y,train_size = n/1526)
        mod.fit(X_train,y_train)
        y_predict = mod.predict(X_test)
        acc_mat.at[counter,str(samp_techniques[i])] = accuracy_score(y_predict, y_test)
    
    counter += 1


print(acc_mat)
acc_mat.to_csv('submission_best.csv',index=False)