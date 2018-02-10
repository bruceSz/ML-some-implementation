
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


def load_data():
    _TRAIN = "../data/boston_house_p/train.csv"
    _TEST = "../data/boston_house_p/test.csv"

    train = pd.read_csv(_TRAIN,index_col=0)
    test = pd.read_csv(_TEST,index_col=0)
    return  train, test

def pre_of_target(x):
    return np.log1p(x)

def recv_of_target(x):
    return np.expm1(x)

def raw_model_reg():
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    return {
        "ridge":Ridge(),
        'dt_reg':DecisionTreeRegressor(),
        'rf_reg':RandomForestRegressor(),
        'gd_reg':GradientBoostingRegressor()
    }

def base_algo():
    train_df, test_df = load_data()
    target_col = ['SalePrice']
    raw_cols = [u'Id', u'MSSubClass', u'MSZoning', u'LotFrontage', u'LotArea',
     u'Street', u'Alley', u'LotShape', u'LandContour', u'Utilities',
     u'LotConfig', u'LandSlope', u'Neighborhood', u'Condition1',
     u'Condition2', u'BldgType', u'HouseStyle', u'OverallQual',
     u'OverallCond', u'YearBuilt', u'YearRemodAdd', u'RoofStyle',
     u'RoofMatl', u'Exterior1st', u'Exterior2nd', u'MasVnrType',
     u'MasVnrArea', u'ExterQual', u'ExterCond', u'Foundation', u'BsmtQual',
     u'BsmtCond', u'BsmtExposure', u'BsmtFinType1', u'BsmtFinSF1',
     u'BsmtFinType2', u'BsmtFinSF2', u'BsmtUnfSF', u'TotalBsmtSF',
     u'Heating', u'HeatingQC', u'CentralAir', u'Electrical', u'1stFlrSF',
     u'2ndFlrSF', u'LowQualFinSF', u'GrLivArea', u'BsmtFullBath',
     u'BsmtHalfBath', u'FullBath', u'HalfBath', u'BedroomAbvGr',
     u'KitchenAbvGr', u'KitchenQual', u'TotRmsAbvGrd', u'Functional',
     u'Fireplaces', u'FireplaceQu', u'GarageType', u'GarageYrBlt',
     u'GarageFinish', u'GarageCars', u'GarageArea', u'GarageQual',
     u'GarageCond', u'PavedDrive', u'WoodDeckSF', u'OpenPorchSF',
     u'EnclosedPorch', u'3SsnPorch', u'ScreenPorch', u'PoolArea', u'PoolQC',
     u'Fence', u'MiscFeature', u'MiscVal', u'MoSold', u'YrSold', u'SaleType',
     u'SaleCondition', u'SalePrice']
    t_cols = [ u'MSSubClass', u'MSZoning', u'LotFrontage', u'LotArea',
       u'Street', u'Alley', u'LotShape', u'LandContour', u'Utilities',
       u'LotConfig', u'LandSlope', u'Neighborhood', u'Condition1',
       u'Condition2', u'BldgType', u'HouseStyle', u'OverallQual',
       u'OverallCond', u'YearBuilt', u'YearRemodAdd', u'RoofStyle',
       u'RoofMatl', u'Exterior1st', u'Exterior2nd', u'MasVnrType',
       u'MasVnrArea', u'ExterQual', u'ExterCond', u'Foundation', u'BsmtQual',
       u'BsmtCond', u'BsmtExposure', u'BsmtFinType1', u'BsmtFinSF1',
       u'BsmtFinType2', u'BsmtFinSF2', u'BsmtUnfSF', u'TotalBsmtSF',
       u'Heating', u'HeatingQC', u'CentralAir', u'Electrical', u'1stFlrSF',
       u'2ndFlrSF', u'LowQualFinSF', u'GrLivArea', u'BsmtFullBath',
       u'BsmtHalfBath', u'FullBath', u'HalfBath', u'BedroomAbvGr',
       u'KitchenAbvGr', u'KitchenQual', u'TotRmsAbvGrd', u'Functional',
       u'Fireplaces', u'FireplaceQu', u'GarageType', u'GarageYrBlt',
       u'GarageFinish', u'GarageCars', u'GarageArea', u'GarageQual',
       u'GarageCond', u'PavedDrive', u'WoodDeckSF', u'OpenPorchSF',
       u'EnclosedPorch', u'3SsnPorch', u'ScreenPorch', u'PoolArea', u'PoolQC',
       u'Fence', u'MiscFeature', u'MiscVal', u'MoSold', u'YrSold', u'SaleType',
       u'SaleCondition']


    target_df = train_df[target_col]
    all_df = pd.concat([train_df[t_cols],test_df[t_cols]])


    #1  deal with target dis transformation
    target_df.loc[:,target_col] = pre_of_target(target_df[target_col])


    # set MSSubClass to string to make get_dummies run automatically on all columns
    all_df['MSSubClass'] = all_df['MSSubClass'].astype('string')
    all_dummy_df = pd.get_dummies(all_df)
    # use pandas's func to get all dummies. and for MSSubClass , do it manually

    #subc_df = pd.get_dummies(dummies_df['MSSubClass'],prefix='MSSubClass')
    #all_dummy_class = pd.concat([dummies_df,subc_df],axis=1)
    #all_dummy_df = all_dummy_class.drop(['MSSubClass'],axis=1)

    # check and set na with mean
    mean_cols = all_dummy_df.mean()
    all_dummy_df = all_dummy_df.fillna(mean_cols)
    numerical_cols = (all_df.columns[all_df.dtypes!='object'])
    numerical_cols_means = all_dummy_df.loc[:,numerical_cols].mean()
    numerical_cols_std = all_dummy_df.loc[:,numerical_cols].std()

    all_dummy_df.loc[:,numerical_cols] = all_dummy_df.loc[:,numerical_cols] - numerical_cols_means
    all_dummy_df.loc[:,numerical_cols] = all_dummy_df.loc[:,numerical_cols] / numerical_cols_std

    #2 string to dummies

    #2 fillna
    d_train = all_dummy_df.loc[train_df.index]
    d_test = all_dummy_df.loc[test_df.index]

    preds = stacked_model(['rf_reg','gd_reg'])
    model_name = "rf_gd_reg"

    submit_cols = ['Id', 'SalePrice']
    sub_df = pd.DataFrame({'Id': test_df.index,
                           'SalePrice': recv_of_target((np.mean(preds)) / 2)})
    sub_df.to_csv("../data/boston_house_p/%s.csv" % model_name, index=False)

def stacked_model(model_names,d_train,target_df,d_test):
    models = raw_model_reg()
    model_name = 'gd_reg'
    model_name2 = 'rf_reg'
    val_score(models[model_name], d_train, target_df, 5)


    m = models[model_name]
    m2 = models[model_name2]
    m.fit(d_train,target_df)
    m2.fit(d_train,target_df)
    pred = (models[model_name].predict(d_test))
    pred2 = (models[model_name2].predict(d_test))



def train_with_diff_fold(f_size,m_name,df_train,target_df):
    models = raw_model_reg()
    f_size = 8
    ss = []
    for f in range(2,f_size):
        ss.append(val_score(models[m_name], df_train,target_df , f))
    plot_line(ss)




def plot_line(x):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x)
    plt.show()


def val_score(est, df,test,cv_number):
    from sklearn.model_selection import cross_val_score
    scores = -cross_val_score(est,df,test,cv=cv_number,scoring='neg_mean_squared_error')

    rmse = np.sqrt(scores)
    return rmse.mean()


def main():
    # base algo
    base_algo()


if __name__ == "__main__":
    main()