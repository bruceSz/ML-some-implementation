from matplotlib.colors import ListedColormap
import  matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets,svm
from sklearn.svm import SVC
from sklearn.datasets import make_moons,make_circles,make_classification
from sklearn.preprocessing import StandardScaler

def main():
    X,y = make_circles(noise=0.2,factor=0.5,random_state=1)
    X = StandardScaler().fit_transform(X)
    cm = plt.cm.RdBu
    cm_bright  = ListedColormap(['#FF0000', '#0000FF'])

    from sklearn.model_selection import GridSearchCV
    params = {
        "C":[0.1,0.5,1,10],
        "gamma":[1,0.1,0.01]
    }
    grid = GridSearchCV(SVC(),param_grid=params,cv=5)
    #grid.fit(X,y)
    #print("Best parameters are %s with a score of %0.2f"%(grid.best_params_,grid.best_score_))
    x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02),
                        np.arange(y_min,y_max,0.02))

    #1 bigger the gamma, kernel result quicker getting to be zero ,then less support vector considered. more `fit`
    #2 the bigger the c, slack partitioned is is considered more.
    clf = SVC(C=100,gamma=1)
    clf.fit(X,y)
    z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm,alpha=0.8)

    #fig,ax = plt.subplots()
    plt.scatter(X[:,0],X[:,1],c=y,cmap=cm_bright)
    #plt.set_title("Input data")
    #plt.set_xticks(())
    #plt.set_yticks(())
    plt.show()



if __name__ == "__main__":
    main()