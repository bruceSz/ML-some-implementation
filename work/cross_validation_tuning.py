
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import copy
import pandas as pd

def ft_selection():
    da = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv",index_col=0)
    fts = ['TV','radio','newspaper']
    X = da[fts]
    y = da['sales']

    for ft in fts:
        tmp_ft = copy.copy(fts)
        tmp_ft.remove(ft)
        lr = LinearRegression()
        tmp_X = X[tmp_ft]
        print("ft %s"%ft)
        print(np.sqrt(-cross_val_score(lr,tmp_X,y,cv=10,scoring='neg_mean_squared_error')).mean(0))



def kf_selection():
    iris = load_iris()
    X = iris.data
    y = iris.target
    kf = KFold(n_splits=5,shuffle=False)
    for train_index, test_index in kf.split(X):
        print ("Train index start:%d,end:%d"%(train_index[0],train_index[train_index.shape[0]-1]))
        print("Test index start %d,end:%d"%(test_index[0],test_index[test_index.shape[0]-1]))
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    print(scores.mean())
    k_scores = []
    k_range = [i for i in range(1,31)]
    #1. seems k=20 has the best accuracy
    for k in range(1,31):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
        k_scores.append(scores.mean())
    # 2.
    knn = KNeighborsClassifier(n_neighbors=20)
    print(cross_val_score(knn,X,y,cv=10,scoring='accuracy').mean())
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    print(cross_val_score(lr,X,y,cv=10,scoring='accuracy').mean())
    #plt.plot(k_range,k_scores)
    #plt.xlabel("Value of k for KNN")
    #plt.ylabel("Cross-validated Accuracy")
    #plt.show()

def simple_train():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))

def main():
    #kf_selection()
    ft_selection()


if __name__ == "__main__":
    main()