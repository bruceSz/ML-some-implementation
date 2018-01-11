
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np


def visualize(da):
    print(da.head())
    sns.pairplot(da, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=5, aspect=0.7, kind='reg')
    plt.show()


def main():
    da = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv",index_col=0)
    fts = ['TV','radio','newspaper']
    tar = ['sales']
    X = da[fts]
    y = da[tar]
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=4)
    linreg = LinearRegression()

    linreg.fit(X_train,y_train)
    print(linreg.intercept_)
    print(linreg.coef_)
    y_pred = linreg.predict(X_test)
    print(mean_absolute_error(y_test,y_pred))
    print(mean_squared_error(y_test,y_pred))
    print(np.sqrt(mean_squared_error(y_test,y_pred)))
    # drop one ft will decrease the rmse.



if __name__ == "__main__":
    main()