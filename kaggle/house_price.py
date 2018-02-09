
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_TRAIN = "../data/boston_house_p/train.csv"
_TEST = "../data/boston_house_p/test.csv"

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn import metrics


def load_data():
    train = pd.read_csv(_TRAIN)
    test = pd.read_csv(_TEST)
    return train,test

def visulize():
    from collections import Counter
    train,test = load_data()
    print(train.describe())
    c = Counter(train['medv'])
    print(sorted(c.items(),cmp= lambda  x,y: - cmp(x[1] ,  y[1]))[:10])
    train_cols = [u'crim', u'zn', u'indus', u'chas', u'nox', u'rm', u'age', u'dis',
                  u'rad', u'tax', u'ptratio', u'black', u'lstat']
    train_cols.append('medv')
    excep  = train.loc[:,train_cols]
    print(excep.corr())
    sns.heatmap(excep.corr(),annot=True)
    #plt.bar(c.keys(),c.values())
    plt.show()


def dump_to_sub(model,fts,scaller,tran_rev,name):
    train,test = load_data()
    submit_cols = ['ID','medv']
    df = pd.DataFrame()
    test[fts] = scaller.transform(test[fts])
    test['medv'] = tran_rev(model.predict(test[fts]))
    df[submit_cols] = test[submit_cols]
    df.to_csv("../data/boston_house_p/"+name,index=False)


def base_algo():
    train,test = load_data()
    r_train_cols = [u'ID', u'crim', u'zn', u'indus', u'chas', u'nox', u'rm', u'age', u'dis',
       u'rad', u'tax', u'ptratio', u'black', u'lstat']
    train_cols = [ u'crim', u'zn', u'indus', u'chas', u'nox', u'rm', u'age', u'dis',
       u'rad', u'tax', u'ptratio', u'black', u'lstat']
    test_cols = ['medv']
    all_cols  = train_cols + test_cols
    print(train.columns)
    print(train.info())
    print(test.info())
    m = RandomForestRegressor()
    scaller = StandardScaler()
    from sklearn.preprocessing import FunctionTransformer
    trans = FunctionTransformer(np.log1p)

    train[train_cols] = scaller.fit_transform(train[train_cols])
    train['new_target'] = trans.transform(train[test_cols])
    fts_scallter = StandardScaler()
    train[train_cols] = fts_scallter.fit_transform(train[train_cols])
    tran_rev = np.expm1
    train_x,test_x,train_y,test_y  = train_test_split(train[train_cols],train['new_target'])
    m.fit(train_x,train_y)
    to_validate =  train.loc[test_y.index,test_cols]
    print(np.sqrt(metrics.mean_squared_error(tran_rev(m.predict(test_x)),to_validate)))
    m.fit(train[train_cols],train['new_target'])
    dump_to_sub(m,train_cols,fts_scallter,tran_rev,"raw_rf.csv")
    #print(m.score(test_x[train_cols],test_y[test_cols]))

def dnn_algo():
    train, test = load_data()
    r_train_cols = [u'ID', u'crim', u'zn', u'indus', u'chas', u'nox', u'rm', u'age', u'dis',
                    u'rad', u'tax', u'ptratio', u'black', u'lstat']
    train_cols = [u'crim', u'zn', u'indus', u'chas', u'nox', u'rm', u'age', u'dis',
                  u'rad', u'tax', u'ptratio', u'black', u'lstat']

    test_cols = ['medv']
    all_cols = train_cols + test_cols
    print(train.info())
    scaller = StandardScaler()
    train[train_cols] = scaller.fit_transform(train[train_cols])
    from sklearn.model_selection import ShuffleSplit
    cv_split = ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,
                            random_state=0)

    def first_dnn_model(batch_size=5,nb_epoch=50):
        in_shape = len(train_cols)
        from keras.models import Sequential
        from keras.layers import Dense
        model = Sequential()
        model.add(Dense(in_shape, input_dim=in_shape,
                        kernel_initializer='normal',
                        use_bias=True,
                        bias_initializer='normal'))

        model.add(Dense(20, input_dim=20, kernel_initializer='normal', use_bias=True))

        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score


def train_with_raw_keras(model,kfold,train,all_cols,train_cols):
    pass

def train_with_kera_reg(model,kfold,train,all_cols,train_cols):
    from keras.wrappers.scikit_learn import KerasRegressor
    est = KerasRegressor(build_fn=model,
                         nb_epoch=500, batch_size=2, verbose=1)
    # results = cross_val_score(estimator=est,X=train[train_cols],y=train[test_cols],cv=kfold)
    train_vals = train[all_cols].values
    print(train_vals.shape)
    train_x, test_x, train_y, test_y = train_test_split(train_vals[:,0:13], train_vals[:,13], test_size=0.1)
    print(len(train_cols))
    print(train_x.shape)
    print(test_x.shape)

    est.fit(train_x, train_y)
    print("\n")
    print("xxxxxxxxx",np.sqrt(metrics.mean_squared_error(est.predict(test_x), test_y)))


def gridcv_algo():
    from sklearn.model_selection import ShuffleSplit
    train,test = load_data()
    r_train_cols = [u'ID', u'crim', u'zn', u'indus', u'chas', u'nox', u'rm', u'age', u'dis',
       u'rad', u'tax', u'ptratio', u'black', u'lstat']
    train_cols = [ u'crim', u'zn', u'indus', u'chas', u'nox', u'rm', u'age', u'dis',
       u'rad', u'tax', u'ptratio', u'black', u'lstat']
    test_cols = ['medv']
    print(train.info())
    m = RandomForestRegressor()
    scaller = StandardScaler()
    train[train_cols] = scaller.fit_transform(train[train_cols])
    cv_split = ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,
                                            random_state=0)

    grid_search_models(train,train_cols,test_cols,scaller,cv_split)



def grid_search_models(train,train_cols,test_cols,scaller,cv_split):
    import common
    import time
    est, params = common.get_all_algo(type='reg')
    for clf,param in zip(est,params):
        name = clf[0]
        m = clf[1]
        print(name, m )
        start = time.time()
        rf_cv = GridSearchCV(m,param,scoring='neg_mean_squared_log_error',cv=cv_split)
        #train_x,test_x,train_y,test_y  = train_test_split(train[train_cols],train[test_cols])
        rf_cv.fit(train[train_cols],train[test_cols])
        print("mean neg_mean_squar_error",rf_cv.cv_results_['mean_train_score'].mean())

        dump_to_sub(rf_cv,train_cols,scaller,"raw_%s.csv"%name)
        end = time.time()
        print ("Cost time %f"%(end-start))


def linear_algo():
    train, test = load_data()
    r_train_cols = [u'ID', u'crim', u'zn', u'indus', u'chas', u'nox', u'rm', u'age', u'dis',
                    u'rad', u'tax', u'ptratio', u'black', u'lstat']
    train_cols = [u'crim', u'zn', u'indus', u'chas', u'nox', u'rm', u'age', u'dis',
                  u'rad', u'tax', u'ptratio', u'black', u'lstat']
    test_cols = ['medv']
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import Ridge
    m = Ridge()
    scaller = StandardScaler()
    train[train_cols] = scaller.fit_transform(train[train_cols])
    train_x, test_x, train_y, test_y = train_test_split(train[train_cols], train[test_cols])
    print(train_x.columns)
    m.fit(train_x, train_y)
    print(np.sqrt(metrics.mean_squared_error(m.predict(test_x), test_y)))
    dump_to_sub(m, train_cols, scaller, "raw_linear.csv")


def main():
    #visulize()
    base_algo()
    #gridcv_algo()
    #dnn_algo()
    #linear_algo()


if __name__ == "__main__":
    main()