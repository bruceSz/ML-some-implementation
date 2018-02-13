
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


import common
import ensemble

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

def model_params():
    params = {
        'ridge': {
            'alpha':[0.01,0.05,0.1,0.5,1,3,5,7,10]
        },
        'bag_reg':{
            'n_estimators':[10,20,30,50,80,100],
            'max_features':[0.5,0.8,1.0]
        },
        'gd_reg':{
            'n_estimators' : [10,20,30,50,80,100,200,500],
            'subsample':[0.5,0.7,0.8,1.0],
            'learning_rate':[0.2,0.5,0.8,1.0]

        },
        'ada_reg':{
            'n_estimators': [20, 30, 50, 80, 100, 200, 500]
        },
        'xgb_reg':{
            'n_estimators': [ 80, 100, 200, 500],
            'subsample':[0.5,0.8,1.0],
            'max_depth':[2,5,6,7,9],
            'learning_rate':[0.01,0.05,0.1,0.15,0.2]
        },
        'svr':{
            "C":[0.1,1,5,10],
            'gamma':[0.1,0.2,0.3]
        }
    }
    return params

def raw_model_reg():
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.svm import SVR
    from sklearn.svm import NuSVR
    from sklearn.svm import LinearSVR
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from lightgbm import LGBMRegressor

    from  xgboost import XGBRegressor
    return {
        "pipe_ridge":make_pipeline(RobustScaler(),Ridge(10)),
        "ridge":Ridge(10),
        "k_ridge":make_pipeline(RobustScaler(),KernelRidge(alpha=0.6,kernel='polynomial')),
        'dt_reg':DecisionTreeRegressor(),
        'rf_reg':RandomForestRegressor(),
        'svr':SVR(),
        'nusvr':NuSVR(),
        'lsvr':LinearSVR(),
        'lgb_reg':LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11),
        # ada seems not work well
        'ada_reg':AdaBoostRegressor(base_estimator=Ridge(10)),
        'gd_reg':GradientBoostingRegressor(),
        'gd_reg_gs': GradientBoostingRegressor(subsample=1.0,learning_rate=0.2,n_estimators=200),
        'xgb_reg_500':XGBRegressor(n_estimators=500),
        'xgb_reg':XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1),
        'bag_reg':BaggingRegressor(base_estimator=Ridge(10),max_features=0.5,n_estimators=20),
        'bag_reg_est10': BaggingRegressor(base_estimator=Ridge(10), max_features=0.5, n_estimators=10)
    }




def fe():
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
    t_cols = [u'MSSubClass', u'MSZoning', u'LotFrontage', u'LotArea',
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
    #train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000)])
    all_df = pd.concat([train_df[t_cols], test_df[t_cols]])

    # 1  deal with target dis transformation
    target_df.loc[:, target_col] = pre_of_target(target_df[target_col])

    # for LotFrontage
    all_df['LotFrontage'] = all_df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )
    # for mszone
    all_df['MSZoning'] = all_df['MSZoning'].fillna(all_df['MSZoning'].mode()[0])
    # Functional
    all_df['Functional'] = all_df['Functional'].fillna('Typ')
    # eclectrical
    all_df['Electrical'] = all_df['Electrical'].fillna(all_df['Electrical'].mode()[0])
    all_df['KitchenQual'] = all_df['KitchenQual'].fillna(all_df['KitchenQual'].mode()[0])
    all_df['Exterior1st'] = all_df['Exterior1st'].fillna(all_df['Exterior1st'].mode()[0])
    all_df['Exterior2nd'] = all_df['Exterior2nd'].fillna(all_df['Exterior2nd'].mode()[0])
    all_df['SaleType'] = all_df['SaleType'].fillna(all_df['SaleType'].mode()[0])

    # set MSSubClass to string to make get_dummies run automatically on all columns
    all_df.loc[:, 'MSSubClass'] = all_df['MSSubClass'].astype(str)
    #all_df['OverallCond'] = all_df['OverallCond'].astype(str)
    #all_df['YrSold'] = all_df['YrSold'].astype(str)
    #all_df['MoSold'] = all_df['MoSold'].astype(str)
    #from sklearn.preprocessing import LabelEncoder
    #cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
    #        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
    #        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
    #        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
    #        'YrSold', 'MoSold')
    #for c in cols:
    #    lbl = LabelEncoder()
    #    lbl.fit(list(all_df[c].values))
    #    all_df[c] = lbl.transform(list(all_df[c].values))
    all_df['TotalSF'] = all_df['TotalBsmtSF'] + all_df['1stFlrSF'] + all_df['2ndFlrSF']
    all_dummy_df = pd.get_dummies(all_df)

    # use pandas's func to get all dummies. and for MSSubClass , do it manually

    # subc_df = pd.get_dummies(dummies_df['MSSubClass'],prefix='MSSubClass')
    # all_dummy_class = pd.concat([dummies_df,subc_df],axis=1)
    # all_dummy_df = all_dummy_class.drop(['MSSubClass'],axis=1)

    # check and set na with mean




    #all_dummy_df['MasVnrArea'] = all_dummy_df['MasVnrArea'].fillna(0)
    #for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    #    all_dummy_df[col] = all_dummy_df[col].fillna(0)
    #for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    #    all_dummy_df[col] = all_dummy_df[col].fillna(0)

    mean_cols = all_dummy_df.mean()
    all_dummy_df = all_dummy_df.fillna(mean_cols)
    print(all_dummy_df.isnull().sum())

    numerical_cols = (all_df.columns[all_df.dtypes != 'object'])
    from scipy.stats import stats
    # TODO.study boxcox1p
    from scipy.special import boxcox1p
    skewed_fts = all_dummy_df[numerical_cols].apply(lambda x:stats.skew(x)).sort_values()
    skewness = pd.DataFrame({'Skew':skewed_fts})
    skewness = skewness[abs(skewness['Skew'])>1]
    lam = 0
    # it seems below work will make ridge score getting lower.
    #for feat in skewness.index:
    #    all_dummy_df[feat] = boxcox1p(all_dummy_df[feat],lam)
    numerical_cols_means = all_dummy_df.loc[:, numerical_cols].mean()
    numerical_cols_std = all_dummy_df.loc[:, numerical_cols].std()
    #all_dummy_df.drop(all_dummy_df[all_dummy_df['GrLivArea']>4000])

    all_dummy_df.loc[:, numerical_cols] = all_dummy_df.loc[:, numerical_cols] - numerical_cols_means
    all_dummy_df.loc[:, numerical_cols] = all_dummy_df.loc[:, numerical_cols] / numerical_cols_std

    # 2 string to dummies



    #d_train = d_train.drop(d_train[(all_dummy_df['GrLivArea'] > 6) & (all_dummy_df['SalePrice'] < 12.5)])
    d_train = all_dummy_df.loc[train_df.index]

    train_cols = d_train.columns
    d_test = all_dummy_df.loc[test_df.index]
    all_out = pd.concat([d_train,target_df],axis=1)
    #all_out = all_out[(all_out['GrLivArea'] <= 6)|(all_out['SalePrice']>=12.5)]
    all_out = all_out.drop(all_out[(all_out['GrLivArea'] > 6) & (all_out['SalePrice'] < 12.5)].index)
    d_train = all_out[train_cols]
    target_df = all_out[target_col]
    return d_train,d_test,target_df,train_df,test_df

def base_algo():

    d_train,d_test,target_df,train_df,test_df = fe()
    #print((x for x in d_test.columns if x.startswith("LandSlope")))
    models = raw_model_reg()
    params = model_params()
    name = "rf_reg"
    #name = 'ridge'
    #m = models[name]
    #m_avg =  ensemble.AveragingModels([models['xgb_reg'],models['ridge']])
    #m_stack = models[name]
    m_stack = ensemble.StackModel([models['lgb_reg'],models['rf_reg'],
                                   models['ridge'],models['gd_reg_gs'],models['xgb_reg']])
    #print(val_score(m,d_train,target_df.values.ravel(),10))
    #fit_and_sub(m,"gd_reg_gs",d_test,d_train,target_df.values.ravel())

    #sm = StackModel([models['gd_reg'],models['rf_reg'],models['ridge']])
    #sm.fit(d_train,target_df)

    print(val_score(m_stack,d_train.values,target_df.values,10))
    print(d_train.shape)
    print(target_df.shape)
    fit_and_sub(m_stack, "fed_%s_best"%name, d_test, d_train.values, target_df.values)

    #t_m = common.tune_model_new(models[name],10,params[name], d_train,target_df.values.ravel(),'neg_mean_squared_error')
    #output_cv_result(t_m)

def fit_and_sub(m,name,d_test,d_train,target_df):
    #print(val_score(models['ridge'],d_train,target_df,10))
    #m = models[name]

    print(d_train.shape)
    print(target_df.shape)
    m.fit(d_train,target_df)
    create_sub(m,"%s"%name,d_test)


def create_sub(m,name,test_df):
    preds = m.predict(test_df.values)
    submit_cols = ['Id', 'SalePrice']
    sub_df = pd.DataFrame({'Id': test_df.index,
                           'SalePrice': recv_of_target(preds).ravel()})
    sub_df.to_csv("../data/boston_house_p/%s.csv" % name, index=False)



def output_cv_result(m):
    cv_ = m.cv_results_
    print("best score", np.sqrt(-m.best_score_))
    print("best param:",m.best_params_)
    print("mean train score",np.sqrt(-cv_['mean_train_score']).mean())
    print("mean test score",np.sqrt(-cv_['mean_test_score']).mean())
    print("mean test std",cv_['std_test_score'].mean())
    #scores = val_score(models['ridge'],d_train,target_df,10)
    #preds = stacked_model(['rf_reg','gd_reg'],d_train,target_df,d_test)
    #model_name = "rf_gd_reg"
    #submit_cols = ['Id', 'SalePrice']
    #sub_df = pd.DataFrame({'Id': test_df.index,
    #                       'SalePrice': recv_of_target((np.mean(preds,axis=0)))})
    #sub_df.to_csv("../data/boston_house_p/%s.csv" % model_name, index=False)



def stacked_model(model_names,d_train,target_df,d_test):
    models = raw_model_reg()
    model_name = 'gd_reg'
    model_name2 = 'rf_reg'
    val_score(models[model_name], d_train, target_df.values.ravel(), 5)

    m = models[model_name]
    m2 = models[model_name2]
    m.fit(d_train,target_df.values.ravel())
    m2.fit(d_train,target_df.values.ravel())
    pred = (models[model_name].predict(d_test))
    pred2 = (models[model_name2].predict(d_test))
    return [pred,pred2]



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


def val_score_stack(est_in,df,target,cv):
    from sklearn.model_selection import ShuffleSplit
    import copy
    x_val = df.values
    y_val = target.values
    est = copy.copy(est_in)
    ss = ShuffleSplit(n_splits=cv,test_size=1/cv,random_state=0)
    for train_idx,test_idx in ss.split(x_val):
        train_x = x_val[train_idx]
        train_y = y_val[train_idx]
        test_x = x_val[test_idx]
        test_y = y_val[test_idx]
        est.fit(train_x,train_y)
        pred = est.predict(test_x)
        mse = np.square(pred - test_y).mean()
        print(np.sqrt(mse))

def visualize():
    d_train,d_test,target_df,r_train_df, r_test_df = fe()

    # 1 outlier
    # 2 distribute target
    v=1
    if v ==1:
        fig,ax = plt.subplots()
        ax.scatter(x= d_train['GrLivArea'],y=target_df['SalePrice'])
        plt.xlabel('GrLivArea',fontsize=12)
        plt.ylabel('SalePrice', fontsize=12)
    elif v==2:
        #sns.distplot(r_train_df['SalePrice'],fit=norm)
        pass



    plt.show()


def main():
    # base algo
    base_algo()
    #visualize()


if __name__ == "__main__":
    main()