
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn

def dt_gs():
    iris = load_iris()
    X = iris.data
    y = iris.target

    md = list(range(1,20))
    sm = list(range(1,5))
    param_grid = dict(max_depth=md,min_samples_leaf=sm)

    dt = DecisionTreeClassifier()
    grid = GridSearchCV(dt,param_grid=param_grid,cv=10,scoring='accuracy')
    grid.fit(X,y)
    print(grid.best_score_)
    print(grid.best_params_)
    print(grid.best_estimator_)

def knn_gs():
    iris = load_iris()
    X = iris.data
    y = iris.target

    k_range = list(range(1,31))
    weights = ['uniform','distance']
    param_grid = dict(n_neighbors=k_range,weights=weights)
    knn = KNeighborsClassifier()
    grid = RandomizedSearchCV(knn, param_grid, cv=10,scoring='accuracy')
    grid.fit(X,y)
    train_scores = (grid.cv_results_['mean_train_score'])
    test_scores = (grid.cv_results_['mean_test_score'])
    print(grid.best_params_)
    print(grid.best_estimator_)
    print(grid.best_score_)
    k_range_index = [i for i in range(1,61)]
    #plt.plot(k_range,train_scores)
    plt.plot(test_scores)
    plt.show()

if __name__ == "__main__":
    #main()
    knn_gs()