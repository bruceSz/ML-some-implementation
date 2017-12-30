
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LassoCV


def norm_map(da, fn):
    for idx,row in da.iterrows():
        for col in da.columns:
            normalized = fn(row[col],col)
            row[col] = normalized
        da.loc[idx] = row
    return da

def main():
    da = pd.read_csv("../data/winequality-red.csv",sep=';')
    da = da[:10].astype('float64')
    print(type(da))
    dataLen, dataWid = da.shape
    print(" len: %d, col number:%d"%(dataLen,dataWid))

    cols = da.columns
    xMeans = pd.Series(index=cols)
    xSD = pd.Series(index=cols)
    xNorm = []
    labelNorm = []
    for c in da.columns:
        xMeans[c] = (np.mean(da[c]))
        xSD[c] = (np.std(da[c]))

    da = norm_map(da, lambda x, c: float(x - xMeans[c]) / xSD[c])
    #norm_map(da,lambda x,c:float(x-xMeans[c])/xSD[c])
        #da.loc[idx] = row
        #print(row)
    cols_x = da.columns[:-1]
    col_y = da.columns[-1]
    X = da[cols_x].as_matrix()
    Y = da[col_y].as_matrix()
    print(X.shape)
    print(Y.shape)

    wineM = LassoCV(cv=10).fit(X,Y)

    print(wineM.coef_)
    print(wineM.alpha_)
    print(wineM.alphas_)
    plt.figure()
    plt.plot(wineM.alphas_,wineM.mse_path_,":")
    plt.plot(wineM.alphas_,wineM.mse_path_.mean(axis=1),
             label ="Average MSE  Across Folds", linewidth=2)
    plt.axvline(wineM.alpha_, linestyle='--',
                label='CV Estimate of Best alpha')
    plt.semilogx()
    plt.legend()
    ax = plt.gca()
    ax.invert_xaxis()
    plt.xlabel("alpha")
    plt.ylabel("Mean Square Error")
    plt.axis('tight')
    plt.show()


    #for c in da.columns:
    #    row_s = da[c]

def _map(da, fn):
    for index, row in da.iterrows():   # 获取每行的index、row
        for col_name in da.columns:
            row[col_name] = fn(row[col_name]) # 把结果返回给data
    #return data



def traverse_example():
    inp = [{'c1': 10, 'c2': 100}, {'c1': 11, 'c2': 110}, {'c1': 12, 'c2': 120}]
    df = pd.DataFrame(inp)
    cols = df.columns
    df = df.astype('float64')

    xmeans = pd.Series(index=cols)
    xstd = pd.Series(index=cols)

    for c in df.columns:
        xmeans[c] = (np.mean(df[c]))
        xstd[c] = (np.std(df[c]))
    norm_map(df,lambda x,c:float(x-xmeans[c])/xstd[c])
    #norm_map(df, lambda ele: ele + 1)
    print(df)

if __name__ == "__main__":
    main()
    #traverse_example()

