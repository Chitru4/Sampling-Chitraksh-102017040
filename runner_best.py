import pandas as pd
import numpy as np 

## Reading data

try:
    df = pd.read_csv('Creditcard_data.csv')
except:
    df = pd.read_csv('https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv')

X = df.iloc[:,0:30].values
y = df.iloc[:,30].values

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


def cluster_sampling(X, y, cluster_size):
    n_samples = X.shape[0]

    n_clusters = n_samples // cluster_size

    X_train = np.empty((0, X.shape[1]))
    y_train = np.empty((0,))
    X_test = np.empty((0, X.shape[1]))
    y_test = np.empty((0,))

    for i in range(n_clusters):
        cluster_indices = np.arange(i * cluster_size, (i + 1) * cluster_size)

        test_index = np.random.choice(cluster_indices)

        train_indices = np.delete(cluster_indices, np.where(cluster_indices == test_index))
        X_train_cluster = X[train_indices]
        y_train_cluster = y[train_indices]
        X_test_cluster = X[test_index].reshape(1, -1)
        y_test_cluster = y[test_index].reshape(1,)

        X_train = np.vstack((X_train, X_train_cluster))
        y_train = np.concatenate((y_train, y_train_cluster))
        X_test = np.vstack((X_test, X_test_cluster))
        y_test = np.concatenate((y_test, y_test_cluster))

    return X_train, X_test, y_train, y_test

def reservoir_sampling(X, y, k):
    X_train = np.empty((0, X.shape[1]))
    y_train = np.empty((0,))
    X_test = np.empty((0, X.shape[1]))
    y_test = np.empty((0,))

    X_train = X[:k]
    y_train = y[:k]

    for i in range(k, X.shape[0]):
        j = np.random.randint(0, i + 1)

        if j < k:
            replace_index = np.random.choice(k)
            X_train[replace_index] = X[i]
            y_train[replace_index] = y[i]
        else:
            X_test = np.vstack((X_test, X[i]))
            y_test = np.concatenate((y_test, y[i].reshape(1,)))

    return X_train, X_test, y_train, y_test


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
        elif i == 1:
            # Stratified sampling
            X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y,train_size = n/1526)
        elif i == 2:
            # Cluster sampling
            X_train, X_test, y_train, y_test = cluster_sampling(X,y,2)
        elif i == 3:
            # Systematic sampling
            sampling_rate = 2
            n_samples = X.shape[0]
            n_samples_train = n_samples // sampling_rate
            n_samples_test = n_samples - n_samples_train
            train_indices = np.arange(n_samples_train) * sampling_rate
            test_indices = np.delete(np.arange(n_samples), train_indices)
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]
        elif i == 4:
            # Reservoir sampling
            X_train, X_test, y_train, y_test = reservoir_sampling(X, y, 10)

        mod.fit(X_train,y_train)
        y_predict = mod.predict(X_test)
        acc_mat.at[counter,str(samp_techniques[i])] = accuracy_score(y_predict, y_test)
    
    counter += 1


print(acc_mat)
acc_mat.to_csv('submission_best.csv',index=False)