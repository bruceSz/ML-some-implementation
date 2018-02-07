#coding=utf-8

from sklearn import ensemble
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import model_selection
from sklearn import discriminant_analysis
from sklearn import naive_bayes
from sklearn import gaussian_process
import numpy as np
import pandas as pd
#
def get_all_tuned_algo():
    vote_est = [
        # Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
        ('ada', ensemble.AdaBoostClassifier()),
        ('bc', ensemble.BaggingClassifier()),
        ('etc', ensemble.ExtraTreesClassifier()),
        #('gbc', ensemble.GradientBoostingClassifier()),
        ('rfc', ensemble.RandomForestClassifier()),
        ('decision tree',tree.DecisionTreeClassifier()),

        # Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
        # ('gpc', gaussian_process.GaussianProcessClassifier()),

        # GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        # ('lr', linear_model.LogisticRegressionCV()),

        # Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
        # ('bnb', naive_bayes.BernoulliNB()),
        # ('gnb', naive_bayes.GaussianNB()),

        # Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
        ('knn', neighbors.KNeighborsClassifier()),

        # SVM: http://scikit-learn.org/stable/modules/svm.html
        ('svc', svm.SVC(probability=True))
    ]

    params = [
        # adaboost
        [{
            'n_estimators': 300,
            'learning_rate': 0.03,
            'random_state': 0
        }],
        # bagging
        [{
            'n_estimators': 300,  # default=10
            'max_samples': 0.5,  # default=1.0
            'random_state': 0
        }],
        # extratrees
        [{
            # ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
            'n_estimators': 100,  # default=10
            'criterion': 'entropy',  # default='gini'
            'max_depth': 10,  # default=None
            'random_state': 0
        }],
        # gradient boosting
        #[{
            # GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
            # 'loss': ['deviance', 'exponential'], #default=’deviance’
        #    'learning_rate': .05,
            # default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
        #    'n_estimators': 300,
            # default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            # 'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
        #    'max_depth': 2,  # default=3
        #    'subsample': 1.0,
        #    'random_state': 0
        #}],
        # rf
        [{
            # RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
            'n_estimators': 300,  # default=10
            'criterion': 'entropy',  # default=”gini”
            'max_depth': 10,  # default=None
            'oob_score': True,
            # default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
            'random_state': 0
        }],
        # dt
        [{
            'criterion':'entropy',
            'max_depth':4,
            'random_state':0
        }],
        # knn
        [{
            # KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
            'n_neighbors': 7,  # default: 5
            'weights': 'distance',  # default = ‘uniform’
            'algorithm':  'auto'
        }],
        #
        [{
            # SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
            # http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
            # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C':  2,   # default=1.0
            'gamma': 0.1,  # edfault: auto
            'decision_function_shape': 'ovo',   # default:ovr
            'probability': True,
            'random_state': 0
        }]
    ]
    for est,param in zip(vote_est,params):
        est[1].set_params(**(param[0]))
        #print param[0]

    return vote_est, params


def tune_model(model,cv_split=None,param_grid = None,da = None,fts = None,label = None):

    tune_model = model_selection.GridSearchCV(model,
                                              param_grid=param_grid, scoring='roc_auc',
                                              cv=cv_split)
    tune_model.fit(da[fts], da[label])
    dt_para = tune_model.best_params_

    #new_dt = tree.DecisionTreeClassifier(**tune_model.best_params_)
    #new_dt.fit(da[fts],da[label])
    #print("AFTER Train Accuracy Mean %f" % tune_model.cv_results_['mean_train_score'].mean())
    #print('AFTER Test Accuracy Mean %f' % tune_model.cv_results_['mean_test_score'].mean())
    # if this is a non-bias random sample, then +/-3 simtandard deviations (std) from the mean,
    # should statistically capture 99.7% of the subsets
    #print('AFTER Test Accuracy 3*STD %f' % tune_model.cv_results_['std_test_score'].std())
    return tune_model,dt_para

def tune_all_algo(da, fts, target,cv_split):
    import time
    start_total = time.time()
    vote_est,params = get_all_algo()
    for clf,param in zip(vote_est,params):
        start = time.time()
        best_search, params = tune_model(clf[1],param_grid=param,cv_split=cv_split,
                                         da=da,fts=fts,label=target)
        run = time.time()- start
        best_param = best_search.best_params_
        print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__,
                                                                                         best_param, run))
        clf[1].set_params(**best_param)
    run_total = time.time() - start_total
    print('Total optimization time was {:.2f} minutes.'.format(run_total / 60))

    print('-' * 10)


def fill_age_with1(df, groupby_df,col_n,default):
    names = groupby_df.index.names
    ret = []
    for idx_tuple in groupby_df.index:
        cond = None
        tmp_df = df.copy()
        for i in range(len(names)):
            tmp_df = tmp_df.loc[tmp_df[names[i]]==idx_tuple[i]]
        #print("Before shape:")
        #print(tmp_df.shape)
        #print(tmp_df[col_n])
        #print("Group by :")
        #print(groupby_df.loc[idx_tuple][col_n])
        if np.isnan(groupby_df.loc[idx_tuple][col_n]):
            tmp_df[col_n] = default
        else:
            tmp_df[col_n] = groupby_df.loc[idx_tuple][col_n]
        ret.append(tmp_df)
    return pd.concat(ret)

def fill_age_example():
    s1 = np.array([1,2, 4])
    s2 = np.array([1,2,7,8])
    s3 = np.array([9,10,11,12])
    df = pd.DataFrame([s1,s2,s3],columns=['a','b','c','d'])
    df_g = df.groupby(['a','b']).mean()

    print "Raw dataframe:"
    print df
    print "Dataframe group by ret"

    print("Reset index")
    #df_g = df_g.reset_index()
    #df_g = df_g[['a','b','d']]
    #df = fill_age_with(df,df_g,['a','b'],'d')
    df = fill_age_with1(df,df_g,'d')
    print "After fill age with avg"
    print(df)


def get_all_algo():
    MLA = [
        # Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),

        # Gaussian Processes
        gaussian_process.GaussianProcessClassifier(),

        # GLM
        #linear_model.LogisticRegressionCV(),
        #linear_model.PassiveAggressiveClassifier(),
        #linear_model.RidgeClassifierCV(),
        #linear_model.SGDClassifier(),
        #linear_model.Perceptron(),

        # Navies Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),

        # Nearest Neighbor
        neighbors.KNeighborsClassifier(),

        # SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        #svm.LinearSVC(),

        # Trees
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),

        # Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),

        # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        # XGBClassifier()
    ]
    vote_est = [
        # Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
        ('ada boost', ensemble.AdaBoostClassifier()),
        ('bagging', ensemble.BaggingClassifier()),
        ('extratree', ensemble.ExtraTreesClassifier()),
        #('gbc', ensemble.GradientBoostingClassifier()),
        ('random_forest', ensemble.RandomForestClassifier()),
        ('decision tree', tree.DecisionTreeClassifier()),

        # Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
        #('gpc', gaussian_process.GaussianProcessClassifier()),

        # GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        #('lr', linear_model.LogisticRegressionCV()),

        # Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
        #('bnb', naive_bayes.BernoulliNB()),
        #('gnb', naive_bayes.GaussianNB()),

        # Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
        ('knn', neighbors.KNeighborsClassifier()),

        # SVM: http://scikit-learn.org/stable/modules/svm.html
        ('svc', svm.SVC(probability=True))
    ]
    grid_n_estimator = [10, 50, 100, 300]
    grid_ratio = [.1, .25, .5, .75, 1.0]
    grid_learn = [.01, .03, .05, .1, .25]
    grid_max_depth = [2, 4, 6, 8, 10, None]
    grid_min_samples = [5, 10, .03, .05, .10]
    grid_criterion = ['gini', 'entropy']
    grid_bool = [True, False]
    grid_subsample_num = [0.1,0.2,0.5,0.8,1.0]
    grid_seed = [0]
    params = [
        # adaboost
        [{
            'n_estimators':grid_n_estimator,
            'learning_rate':grid_learn,
            'random_state':grid_seed
        }],
        # bagging
        [{
              'n_estimators': grid_n_estimator, #default=10
            'max_samples': grid_ratio, #default=1.0
            'random_state': grid_seed
        }],
        # extratrees
        [{
            # ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
            'n_estimators': grid_n_estimator, # default=10
            'criterion': grid_criterion,  # default='gini'
            'max_depth': grid_max_depth,  # default=None
            'random_state': grid_seed
        }],
        # gradient boosting
        #[{
            # GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
            # 'loss': ['deviance', 'exponential'], #default=’deviance’
        #    'learning_rate': [.05,0.08,0.1,0.3,0.5,0.8],
        # default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
        #    'n_estimators': [300,500,800,1000],
        # default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
        #    # 'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
        #    'max_depth': grid_max_depth,  # default=3
        #    'subsample':grid_subsample_num,
        #    'random_state': grid_seed
        #}],
        # rf
        [{
            # RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
            'n_estimators': grid_n_estimator,  # default=10
            'criterion': grid_criterion,  # default=”gini”
            'max_depth': grid_max_depth,  # default=None
            'oob_score': [True],
        # default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
            'random_state': grid_seed
        }],
        # pure dt
        [{
            'criterion': ['gini', 'entropy'],
             'max_depth': [2, 4, 6, 8, 10],
             'random_state': [0]
        }],
        # knn
        [{
            # KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
            'n_neighbors': [1, 2, 3, 4, 5, 6, 7],  # default: 5
            'weights': ['uniform', 'distance'],  # default = ‘uniform’
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }],
        #
        [{
            # SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
            # http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
            # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [1, 2, 3, 4, 5],  # default=1.0
            'gamma': grid_ratio,  # edfault: auto
            'decision_function_shape': ['ovo', 'ovr'],  # default:ovr
            'probability': [True],
            'random_state': grid_seed
        }]
    ]
    return vote_est,params