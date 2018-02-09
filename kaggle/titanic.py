#coding=utf-8
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import warnings
import seaborn as sns
from sklearn import model_selection
from  sklearn import ensemble
from sklearn import gaussian_process
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn import discriminant_analysis

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import scikitplot as skplt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
import warnings
warnings.warn("deprecated",DeprecationWarning)

from sklearn import ensemble

import common


# TODO 1. https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
# TODO 2. https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling

_TRAIN = "../data/titan/train.csv"
_TEST = "../data/titan/test.csv"


def explore_data():
    df_train = pd.read_csv(_TRAIN)
    #print(df_train.head())
    #print(df_train.dtypes)
    #print(df_train.info())

    fig = plt.figure()
    fig_dims = (3,2)

    plt.subplot2grid(fig_dims,(0,0))
    df_train['Survived'].value_counts().plot(kind='bar',
                                             title="Death and Survival Counts")
    plt.subplot2grid(fig_dims, (0, 1))
    df_train['Pclass'].value_counts().plot(kind='bar',
                                           title='Passenger Class Counts')
    plt.subplot2grid(fig_dims, (1, 0))
    df_train['Sex'].value_counts().plot(kind='bar',
                                        title='Gender Counts')
    plt.subplot2grid(fig_dims, (1, 1))
    df_train['Embarked'].value_counts().plot(kind='bar',
                                             title='Ports of Embarkation Counts')

    plt.subplot2grid(fig_dims, (2, 0))
    df_train['Age'].hist()
    plt.title("Age histogram")
    plt.show()

def explore_data2():
    df_train = pd.read_csv(_TRAIN)
    #print(df_train.info())
    #df_train['Survived'].value_counts().plot(kind='bar')
    #df_train["Age"].hist()
    #survival_stacked_bar(df_train,"Pclass")
    f,ax = plt.subplots(figsize=(10,10))
    sns.heatmap(df_train.corr(),annot=True,linewidths=0.5, fmt=' .2f', ax=ax)
    plt.show()

def embarked_pie(df_train):
    sizes = [sum(df_train['Embarked'] == 'C'), sum(df_train['Embarked'] == 'Q'), sum(df_train['Embarked'] == 'S')]
    colors = ['yellow', 'aqua', 'lime']
    plt.pie(sizes, labels=['c', 'q', 's'], colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.show()


def survival_stacked_bar(df, var):
    Died = df[df['Survived']==0][var].value_counts()/len(df["Survived"]==0)
    Survived = df[df['Survived']==1][var].value_counts()/len(df['Survived']==1)
    da = pd.DataFrame([Died,Survived])
    da.index = ["not survived","survived"]
    da.plot(kind='bar',stacked=True,title="Percentage")
    plt.show()
    #print(Died)


def import_opt():
    pd.set_option('display.width', 500)
    pd.set_option('display.max_columns', 100)
    #pd.set_option('display.notebook_repr_html', True)
    sns.set(style="whitegrid")
    warnings.filterwarnings('ignore')




def fe():
    df_train = pd.read_csv(_TRAIN)
    df_test = pd.read_csv(_TEST)
    traintest = pd.concat([df_train,df_test])
    sex_map = {"male":1,"female":0}

    #1 sex word to numeric
    df_train['Sex'] = df_train['Sex'].map(sex_map)
    df_test['Sex'] = df_test["Sex"].map(sex_map)
    df_train.insert(value=df_train['Name'].map(lambda name: name.split(",")[1].split(".")[0].strip()),
                    loc=12, column="Title")
    df_test.insert(value=df_test['Name'].map(lambda  x:x.split(",")[1].split(".")[0].strip()),loc=11,column="Title")
    title_map = {"Capt": "Officer",
                 "Col": "Officer",
                 "Major": "Officer",
                 "Jonkheer": "Royalty",
                 "Don": "Royalty",
                 "Sir": "Royalty",
                 "Dr": "Officer",
                 "Rev": "Officer",
                 "the Countess": "Royalty",
                 "Dona": "Royalty",
                 "Mme": "Mrs",
                 "Mlle": "Miss",
                 "Ms": "Mrs",
                 "Mr": "Mr",
                 "Mrs": "Mrs",
                 "Miss": "Miss",
                 "Master": "Master",
                 "Lady": "Royalty"
                 }

    df_train["Title"] = df_train.Title.map(title_map)
    df_test["Title"] = df_test.Title.map(title_map)
    # 2 check missing values:
    #for c in df_train.columns:
    #    print(c+" : "+str(sum(df_train[c].isnull()))+ " Missing values")

    # 2.1 add missing ages
    train_set_1 = df_train.groupby(["Pclass", "SibSp"])
    age_mean = df_train['Age'].mean()
    train_set_1_median = train_set_1.median()
    print(df_train[(df_train['Pclass']==3) &  (df_train['SibSp']==8)])
    print(train_set_1_median.loc[(3,8)])
    #train_set_1_median = train_set_1_median.reset_index()
    #train_set_1_median = train_set_1_median[["Pclass", "SibSp","Age"]]
    train = common.fill_age_with1(df_train,train_set_1_median,'Age',age_mean)
    #train = fill_age_with(df_train,train_set_1_median,["Pclass", "SibSp"],'Age')
    #train[train['Age'].isnull()]= train['Age'].mean()
    print train.shape
    test_set_1 = df_test.groupby(["Pclass", "SibSp"])
    test_set_1_median = test_set_1.median()
    test = common.fill_age_with1(df_test,test_set_1_median,'Age',age_mean)

    train["Cabin"] = train['Cabin'].fillna("U")
    test["Cabin"] = test['Cabin'].fillna("U")

    train['Cabin'] = train['Cabin'].map(lambda x: x[0])
    test['Cabin'] = test['Cabin'].map(lambda x: x[0])
    new_cabin_features(train)
    new_cabin_features(test)
    train['Embarked'] = train["Embarked"].fillna("S")
    test["Embarked"] = test['Embarked'].fillna("S")
    test["Fare"] = test["Fare"].fillna(np.mean(test["Fare"]))

    new_em_features(train)
    new_em_features(test)


    title_map_2 = {'Mr': 1,
                   'Mrs': 1,
                   'Miss': 1,
                   'Master': 2,
                   'Officer': 3,
                   'Royalty': 4}

    train["Title"] = train["Title"].map(title_map_2)
    test["Title"] = test["Title"].map(title_map_2)
    #train["FamilySize"] = train["SibSp"].astype(float) + train["Parch"].astype(float) + 1
    #test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
    train.drop(["Name", "Ticket", "PassengerId", "Embarked", "Cabin"], inplace=True, axis=1)
    test.drop(["Name", "Ticket", "Embarked", "Cabin"], inplace=True, axis=1)
    return train,test




def plot_valid_result(x_train,y_train,x_test,y_test,k_fold):
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    print("Logistic regression:")
    print("Accuracy: " + str(acc_score(log_reg, x_test, y_test, k_fold)))
    print(confusion_matrix_model(log_reg, y_test, x_test))
    plt_roc_curve("lr", log_reg, x_test, y_test)
    print("")

    print("LDA")
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    ldaA = lda.transform(x_test)
    print(ldaA.shape)
    print("Accuracy: " + str(acc_score(lda, x_test, y_test, k_fold)))
    print(confusion_matrix_model(lda, y_test, x_test))
    plt_roc_curve("lda", lda, x_test, y_test)

    print("svc linear/rbf kernel")
    svc = SVC(kernel='rbf')
    svc.fit(x_train, y_train)
    print("Accuracy: " + str(acc_score(svc, x_test, y_test, k_fold)))
    print(confusion_matrix_model(svc, y_test, x_test))
    plt_roc_curve("svc", svc, x_test, y_test, has_prob=False)

    print("dt")
    dt = DecisionTreeClassifier(max_depth=5, random_state=5)
    dt.fit(x_train, y_train)
    print("Accuracy: " + str(acc_score(dt, x_test, y_test, k_fold)))
    print(confusion_matrix_model(dt, y_test, x_test))
    plt_roc_curve("dt", dt, x_test, y_test)

    print("Random forest")
    random_f = RandomForestClassifier(n_estimators=50, max_features='sqrt', max_depth=5, random_state=5)
    random_f.fit(x_train, y_train)
    print("Accuracy: " + str(acc_score(random_f, x_test, y_test, k_fold)))
    print(confusion_matrix_model(random_f, y_test, x_test))
    plt_roc_curve("random forest", random_f, x_test, y_test)

    print("Gbdt")
    gbdt = GradientBoostingClassifier(n_estimators=50, max_features='sqrt', max_depth=5, random_state=5)
    gbdt.fit(x_train, y_train)
    print("Accuracy: " + str(acc_score(gbdt, x_test, y_test, k_fold)))
    print(confusion_matrix_model(gbdt, y_test, x_test))
    plt_roc_curve("gbdt", gbdt, x_test, y_test)

def plt_roc_curve(name,model,x_test,y_test,has_prob=True):
    if has_prob:
        fpr,tpr,thresh = skplt.metrics.roc_curve(y_test,model.predict_proba(x_test)[:,1])
    else:
        fpr,tpr,thresh = skplt.metrics.roc_curve(y_test,model.decision_function(x_test))
    x = fpr
    y = tpr
    auc = skplt.metrics.auc(x,y)

    plt.plot(x, y, label='ROC curve for %s (AUC = %0.2f)' % (name, auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

def confusion_matrix_model(model_used,y_test,x_test):
    cm=confusion_matrix(y_test,model_used.predict(x_test))
    col=["Predicted Dead","Predicted Survived"]
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Dead","Predicted Survived"]
    cm.index=["Actual Dead","Actual Survived"]
    cm[col]=np.around(cm[col].div(cm[col].sum(axis=1),axis=0),decimals=2)
    return cm


def importance_of_features(model,x_train):
    features = pd.DataFrame()
    features['feature'] = x_train.columns
    features['importance'] = model.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    return features.plot(kind='barh', figsize=(10,10))


def acc_score(model,x_train, y_train, k_fold):
    return np.mean(cross_val_score(model,x_train,y_train,cv=k_fold,scoring="accuracy"))


def new_em_features(da):
    da["Embarked S"] = np.where(da["Embarked"]=="S",1,0)
    da["Embarked C"] = np.where(da['Embarked']=="C",1,0)



def  new_cabin_features(dataset):
    dataset["Cabin A"]=np.where(dataset["Cabin"]=="A",1,0)
    dataset["Cabin B"]=np.where(dataset["Cabin"]=="B",1,0)
    dataset["Cabin C"]=np.where(dataset["Cabin"]=="C",1,0)
    dataset["Cabin D"]=np.where(dataset["Cabin"]=="D",1,0)
    dataset["Cabin E"]=np.where(dataset["Cabin"]=="E",1,0)
    dataset["Cabin F"]=np.where(dataset["Cabin"]=="F",1,0)
    dataset["Cabin G"]=np.where(dataset["Cabin"]=="G",1,0)
    dataset["Cabin T"]=np.where(dataset["Cabin"]=="T",1,0)
    #Cabin U is


def check_null_number(df,col_n):
    return sum(df[col_n].isnull())


def fill_age_with(df,groupby_df,group_by_col, col_n):
    tmp_n = "new_name"
    df = df.rename(columns={col_n:tmp_n})
    print(df.columns)
    ret = pd.merge(df,groupby_df,on=group_by_col,how='left')
    ret = ret.drop(tmp_n,axis=1)
    return ret


def train_predict(train, test):
    x = train.drop(["Survived"], axis=1)
    y = train["Survived"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

    random_f = RandomForestClassifier(n_estimators=50, max_features='sqrt', max_depth=5, random_state=5)
    random_f.fit(x_train, y_train)
    p_test = random_f.predict(test.drop("PassengerId", axis=1).copy())
    subm = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": p_test})
    subm['PassengerId'].astype('int64')

    subm.to_csv("../data/titan_subm.csv", index=None)


def select_algo(train):
    x = train.drop(["Survived"], axis=1)
    y = train["Survived"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    k_fold = KFold(n_splits=5, shuffle=True, random_state=0)


def main2():
    import_opt()
    #explore_data()
    #explore_data2()
    train, test = fe()
    algo = select_algo(train)
    train_predict()

def main_sub_0206():
    df_train = pd.read_csv(_TRAIN)
    df_test = pd.read_csv(_TEST)
    data_train_1  = df_train.copy(deep=True)
    datas = [data_train_1,df_test]
    # 4 C : Correcting, Completing,Creating,Converting.
    #
    # missing value adding
    # removing unrelated col and cols with too much missing val.
    for da in datas:
        da['Age'].fillna(da['Age'].median(), inplace=True)
        da['Embarked'].fillna(da['Embarked'].mode()[0],inplace=True)
        da['Fare'].fillna(da['Fare'].median(),inplace=True)

    drop_cols = ['PassengerId','Cabin','Ticket']
    data_train_1.drop(drop_cols,axis=1,inplace=True)

    # creating
    for da in datas:
        da['FamilySize'] = da['SibSp'] + da['Parch'] + 1
        da['IsAlone'] = 15
        mask_fm_sz = (da['FamilySize'] > 0)
        da.loc[mask_fm_sz,'IsAlone'] = 0
        da['Title'] = da['Name'].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]
        da['FareBin'] = pd.qcut(da['Fare'],4)
        da['AgeBin'] = pd.cut(da['Age'].astype(int),5)

    title_names = data_train_1['Title'].value_counts() < 10
    data_train_1['Title'] = data_train_1['Title'].apply(lambda  x: 'Misc' if title_names.loc[x] == True else x)

    # converting
    label = LabelEncoder()
    for da in datas:
        da['Sex_code'] = label.fit_transform(da['Sex'])
        da['Embarked_code'] = label.fit_transform(da['Embarked'])
        da['Title_code'] = label.fit_transform(da['Title'])
        da['AgeBin_code'] = label.fit_transform(da['AgeBin'])
        da['FareBin_code']  = label.fit_transform(da['FareBin'])

    Target = ['Survived']
    data_train_1_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']

    data_train_1_x_calc = ['Sex_code', 'Pclass', 'Embarked_code', 'Title_code', 'SibSp', 'Parch', 'Age', 'Fare']

    data_train_1_x_bin = ['Sex_code', 'Pclass', 'Embarked_code', 'Title_code', 'FamilySize', 'AgeBin_code', 'FareBin_code']

    data_train_1_dummy = pd.get_dummies(data_train_1[data_train_1_x])
    data_train_1_x_dummy = data_train_1_dummy.columns.tolist()
    data_train_1_xy_dummy = Target + data_train_1_dummy.columns.tolist()

    submit_cols = ['PassengerId','Survived']
    #print("dumy xy:",data_train_1_xy_dummy)
    train_x,train_y,test_x,test_y = model_selection.train_test_split(data_train_1[data_train_1_x_calc],data_train_1[Target],
                                                                     random_state=0)
    train_x_bin,train_y_bin,test_x_bin,test_y_bin = model_selection.train_test_split(data_train_1[data_train_1_x_bin],
                                                                                     data_train_1[Target],random_state=0)
    #train_x_dummy,train_y_dummy,test_x_dummy,test_y_dummy = model_selection.train_test_split(data_train_1[data_train_1_x_dummy],
    #                                                                                         data_train_1[Target],
    #
    #                                                                     random_state=0)
    # correlation compute.
    #for x in data_train_1_x:
    #    if data_train_1[x].dtype != 'float64':
    #        print('Survival Correlation by:',x)
    #        print(data_train_1[[x,Target[0]]].groupby(x,as_index=False).mean())
    #        print("*"*10,'\n')

    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,
                                                random_state=0)
        #all_algo_perf(data_train_1, data_train_1_x_bin, Target)
    # tune models
    # 1 dt:
    #dt = tree.DecisionTreeClassifier(random_state=0)
    #cv_results = model_selection.cross_validate(dt,data_train_1[data_train_1_x_bin],
    #                                          data_train_1[Target],cv=cv_split)
    #dt.fit(data_train_1[data_train_1_x_bin],data_train_1[Target])

    #print(dt.get_params())
    #print("MLA Train Accuracy Mean %f"%cv_results['train_score'].mean())
    #print('MLA Test Accuracy Mean %f'% cv_results['test_score'].mean())
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
    # should statistically capture 99.7% of the subsets
    #print( 'MLA Test Accuracy 3*STD %f'%(cv_results['test_score'].std()*3))


    base_dt_sub = pd.DataFrame(columns=submit_cols)
    print('-'*10)
    dt_param_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': [2, 4, 6, 8, 10],
                  'random_state': [0]}
    model,dt_param = common.tune_model(tree.DecisionTreeClassifier(),cv_split=cv_split,param_grid=dt_param_grid,
               da=data_train_1,fts=data_train_1_x_bin,label=Target)
    #base_dt_sub[''] = model.predict(df_test[data_train_1_x_bin])
    df_test['Survived'] = model.predict(df_test[data_train_1_x_bin])
    base_dt_sub[submit_cols] = df_test[submit_cols]
    base_dt_sub.to_csv("../data/titan/base_dt.csv",index=False)
    print("Base Train Accuracy Mean %f" % model.cv_results_['mean_train_score'].mean())
    print('Base Test Accuracy Mean %f' % model.cv_results_['mean_test_score'].mean())
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
    # should statistically capture 99.7% of the subsets
    print('Base Test Accuracy 3*STD %f' % model.cv_results_['std_test_score'].std())
    #tune_all_algo(data_train_1,data_train_1_x_bin,Target,cv_split)

    vote_est, params = common.get_all_tuned_algo()

    for pair in vote_est:
        name = pair[0]
        model = pair[1]
        model_cv = model_selection.cross_validate(model,data_train_1[data_train_1_x_bin],
                                                  data_train_1[Target],
                                                  cv=cv_split)
        model.fit(data_train_1[data_train_1_x_bin],data_train_1[Target])

        sub_df = pd.DataFrame(columns=submit_cols)
        df_test['Survived'] = model.predict(df_test[data_train_1_x_bin])
        sub_df[submit_cols] = df_test[submit_cols]
        sub_df.to_csv("../data/titan/%s_dt.csv"%name, index=False)

        print(" %s train Accuracy Mean %f" % (name,model_cv['train_score'].mean()))
        print('%s Test Accuracy Mean %f' % (name,model_cv['test_score'].mean()))
        # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
        # should statistically capture 99.7% of the subsets
        print('%s Test Accuracy 3*STD %f' % (name, model_cv['test_score'].std()))

    grid_hard = ensemble.VotingClassifier(estimators=vote_est,voting='hard')
    grid_hard_cv = model_selection.cross_validate(grid_hard,data_train_1[data_train_1_x_bin],
                                                  data_train_1[Target],
                                                  cv=cv_split)
    grid_hard.fit(data_train_1[data_train_1_x_bin],data_train_1[Target])
    sub_df = pd.DataFrame(columns=submit_cols)
    df_test['Survived'] = grid_hard.predict(df_test[data_train_1_x_bin])
    sub_df[submit_cols] = df_test[submit_cols]
    sub_df.to_csv("../data/titan/grid_hard_dt.csv"  , index=False)
    print("Hard voting Train Accuracy Mean %f" % grid_hard_cv['train_score'].mean())
    print('Hard voting Test Accuracy Mean %f' % grid_hard_cv['test_score'].mean())
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
    # should statistically capture 99.7% of the subsets
    print('Hard voting Test Accuracy 3*STD %f' % grid_hard_cv['test_score'].std())

    grid_soft = ensemble.VotingClassifier(estimators=vote_est,voting='soft')
    grid_soft_cv = model_selection.cross_validate(grid_soft,data_train_1[data_train_1_x_bin],
                                                  data_train_1[Target],
                                                  cv=cv_split)
    grid_soft.fit(data_train_1[data_train_1_x_bin],data_train_1[Target])

    sub_df = pd.DataFrame(columns=submit_cols)

    df_test['Survived'] = grid_soft.predict(df_test[data_train_1_x_bin])
    sub_df[submit_cols] = df_test[submit_cols]
    sub_df.to_csv("../data/titan/grid_soft_dt.csv", index=False)

    print("Soft voting Train Accuracy Mean %f" % grid_soft_cv['train_score'].mean())
    print('Soft voting Test Accuracy Mean %f' % grid_soft_cv['test_score'].mean())
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
    # should statistically capture 99.7% of the subsets
    print('Soft voting Test Accuracy 3*STD %f' % grid_soft_cv['test_score'].std())




def fs(dt_param,da,fts,Target,cv_split):
    from  sklearn import feature_selection
    base_model = tree.DecisionTreeClassifier(**dt_param)
    dtree_rfe = feature_selection.RFECV(base_model,step=1,scoring='accuracy',cv=cv_split)
    dtree_rfe.fit(da[fts],da[Target])
    x_rfe = da[fts].columns.values[dtree_rfe.get_support()]
    print(dtree_rfe.get_support())
    print(dtree_rfe)







def all_algo_perf(data_train_1, data_train_1_x_bin, Target):

    MLA = common.get_all_algo()
    # split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
    # note: this is an alternative to train_test_split
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,
                                            random_state=0)  # run model 10x with 60/30 split intentionally leaving out 10%

    # print(data_train_1.isnull().sum())
    # print("*"*10)
    # print(df_test.isnull().sum())
    MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean',
                   'MLA Test Accuracy 3*STD', 'MLA Time']
    MLA_compare = pd.DataFrame(columns=MLA_columns)
    MLA_predict = data_train_1[Target]

    row_index = 0
    for alg in MLA:
        # set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

        # score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        cv_results = model_selection.cross_validate(alg, data_train_1[data_train_1_x_bin],
                                                    data_train_1[Target], cv=cv_split)

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
        # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
        # should statistically capture 99.7% of the subsets
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results[
                                                                    'test_score'].std() * 3
        # let's know the worst that can happen!

        # save MLA predictions - see section 6 for usage
        alg.fit(data_train_1[data_train_1_x_bin], data_train_1[Target])
        MLA_predict[MLA_name] = alg.predict(data_train_1[data_train_1_x_bin])

        row_index += 1
    MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)
    print(MLA_compare[['MLA Name', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean']])


def main():

    df_train_raw = pd.read_csv(_TRAIN)
    df_test = pd.read_csv(_TEST)
    df_train = df_train_raw.copy(deep=True)
    all_datas = [df_train, df_test]
    #print(df_train.shape)
    #print(df_test.shape)
    from  sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    raw_names = [u'PassengerId', u'Pclass', u'Name', u'Sex', u'Age',
       u'SibSp', u'Parch', u'Ticket', u'Fare', u'Cabin', u'Embarked']
    train_names = [u'PassengerId', u'Pclass', u'Name', u'Sex', u'Age',
    u'SibSp', u'Parch', u'Ticket', u'Fare', u'Cabin', u'Embarked']
    target_col = 'Survived'

    # age fillna
    df_train['Age']=df_train.groupby(["Pclass","SibSp","Parch","Sex"])['Age']\
        .transform(lambda  x:x.fillna(x.mean()))
    df_train['Age'].fillna((df_train['Age'].value_counts().idxmax()),inplace=True)

    df_test['Age'] = df_test.groupby(["Pclass","SibSp","Parch","Sex"])['Age']\
        .transform(lambda x: x.fillna(x.mean()))
    df_test['Age'].fillna(df_test['Age'].value_counts().idxmax(),inplace=True)

    # cabin fillna
    df_train['Cabin'].fillna((df_train['Cabin'].value_counts().idxmax()), inplace=True)
    df_test['Cabin'].fillna(df_test['Cabin'].value_counts().idxmax(), inplace=True)

    # train embark fillna
    #df_train['Embarked'].fillna(df_train['Embarked'].value_counts().idxmax(),inplace=True)
    df_train['Embarked'].fillna(df_train['Embarked'].mode()[0],inplace=True)

    # test fare fillna
    #df_test['Fare'].fillna(df_test['Fare'].value_counts().idxmax(),inplace=True)
    df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)

    # fe transformation


    print(df_train.head(2))
    print(df_train.info())
    print(df_test.info())

    # sex to dummies
    for idx,da in enumerate(all_datas):
        all_datas[idx] = pd.get_dummies(da,columns=['Sex'],
                                         prefix=['sex_'],
                                         prefix_sep="",
                                         dummy_na=False,
                                         drop_first=False)
    raw_names.extend(['sex_female', 'sex_male'])
    train_names.extend(['sex_female', 'sex_male'])
    train_names.remove('Sex')
    df_train[['sex_female','sex_male']] = all_datas[0][['sex_female','sex_male']]
    df_test[['sex_female','sex_male']] = all_datas[1][['sex_female','sex_male']]

    # embarked to dummies
    for idx,da in enumerate(all_datas):
        all_datas[idx] = pd.get_dummies(da,columns=['Embarked'],
                                        prefix=['embarked_'],
                                        prefix_sep="",
                                        dummy_na=False,
                                        drop_first=False)
    t_names = [col_name for col_name in all_datas[0].columns if col_name.startswith("embarked_")]
    raw_names.extend(t_names)
    train_names.extend(t_names)
    train_names.remove('Embarked')
    df_train[t_names] = all_datas[0][t_names]
    df_test[t_names] = all_datas[1][t_names]

    #cabin to one-hot
    train_v_set = compute_val_set(df_train,'Cabin')
    df_train = compute_dummies(df_train,'Cabin','Cabin_',train_v_set)
    df_test = compute_dummies(df_test,'Cabin','Cabin_',train_v_set)
    t_names = [col_name for col_name in df_train.columns if col_name.startswith("Cabin_")]
    raw_names.extend(t_names)
    train_names.extend(t_names)
    train_names.remove("Cabin")

    # name to title and then to category val
    df_train['title'] = df_train['Name'].map(lambda name: name.split(",")[1].split(".")[0].strip())
    df_train = title_short(df_train)

    df_test['title'] = df_train['Name'].map(lambda name: name.split(",")[1].split(".")[0].strip())
    df_test = title_short(df_test)
    train_title_set = compute_val_set(df_train,'title')
    df_train = compute_dummies(df_train,'title','title_',train_title_set)
    df_test = compute_dummies(df_test,'title','title_',train_title_set)
    t_names = [col_name for col_name in df_train.columns if col_name.startswith("title_")]
    raw_names.extend(t_names)
    train_names.extend(t_names)
    train_names.remove("Name")

    # add more fts(by combining)
    for idx,da in enumerate(all_datas):
        all_datas[idx]['FamilySize'] = da['SibSp'] + da['Parch'] + 1
        all_datas[idx]['IsAlone'] = 1
        mask_fm_sz = (all_datas[idx]['FamilySize'] > 0)
        all_datas[idx].loc[mask_fm_sz, 'IsAlone'] = 0
    #train_names.remove("title")
    raw_names.extend(["FamilySize","IsAlone"])
    train_names.extend(["FamilySize","IsAlone"])
    df_train[["FamilySize","IsAlone"]] = all_datas[0][["FamilySize","IsAlone"]]
    df_test[["FamilySize","IsAlone"]] = all_datas[1][["FamilySize","IsAlone"]]

    train_names.remove('PassengerId')
    train_names.remove('Ticket')


    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,
                                            random_state=0)

    submit_cols = ['PassengerId', 'Survived']
    dt = ensemble.GradientBoostingClassifier(random_state=0)

    param_test2 = {
                    'max_depth': range(3, 14,2),
                   'min_samples_split': range(50, 201, 20),
                   'n_estimators':[30,60,80,100,200],
                   'min_samples_leaf' :[20,10,5,30],
                   'max_features' :['sqrt','auto']
                   }
    gsearch2 = GridSearchCV(estimator=dt.set_params(
                                                             random_state=10),
                            param_grid=param_test2, scoring='accuracy', iid=False, cv=5)
    gsearch2.fit(df_train[train_names], df_train[target_col])
    print gsearch2.grid_scores_
    print gsearch2.best_params_
    print gsearch2.best_score_
    #cv_results = model_selection.cross_validate(dt,df_train[train_names],
    #                                         df_train[target_col],cv=cv_split)
    dt = gsearch2
    #dt.fit(df_train[train_names],df_train[target_col])

    sub_df = pd.DataFrame(columns=submit_cols)
    df_test['Survived'] = dt.predict(df_test[train_names])
    sub_df[submit_cols] = df_test[submit_cols]
    sub_df.to_csv("../data/titan/raw_rf.csv", index=False)




def grid_voting(df,fts,target,cv,submit_cols,df_test):
    vote_est, param = common.get_all_tuned_algo()
    for v in vote_est:
        print(v)
    grid_hard = ensemble.VotingClassifier(estimators=vote_est, voting='hard')
    grid_hard_cv = model_selection.cross_validate(grid_hard, df[fts],
                                                  df[target],
                                                  cv=cv)
    grid_hard.fit(df[fts], df[target])
    sub_df = pd.DataFrame(columns=submit_cols)
    df_test['Survived'] = grid_hard.predict(df_test[fts])
    sub_df[submit_cols] = df_test[submit_cols]
    sub_df.to_csv("../data/titan/grid_hard_dt.csv", index=False)
    print("Hard voting Train Accuracy Mean %f" % grid_hard_cv['train_score'].mean())
    print('Hard voting Test Accuracy Mean %f' % grid_hard_cv['test_score'].mean())
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
    # should statistically capture 99.7% of the subsets
    print('Hard voting Test Accuracy 3*STD %f' % grid_hard_cv['test_score'].std())

    grid_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
    grid_soft_cv = model_selection.cross_validate(grid_soft, df[fts],
                                                  df[target],
                                                  cv=cv)
    grid_soft.fit(df[fts], df[target])

    sub_df = pd.DataFrame(columns=submit_cols)

    df_test['Survived'] = grid_soft.predict(df_test[fts])
    sub_df[submit_cols] = df_test[submit_cols]
    sub_df.to_csv("../data/titan/grid_soft_dt.csv", index=False)

    print("Soft voting Train Accuracy Mean %f" % grid_soft_cv['train_score'].mean())
    print('Soft voting Test Accuracy Mean %f' % grid_soft_cv['test_score'].mean())
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
    # should statistically capture 99.7% of the subsets
    print('Soft voting Test Accuracy 3*STD %f' % grid_soft_cv['test_score'].std())


    #vote_est, params = common.get_all_tuned_algo()

    #for pair in vote_est:
    #    name = pair[0]
    #    model = pair[1]
        #model_cv = model_selection.cross_validate(model, df_train[train_names],
        #                                          df_train[target_col],
        #                                          cv=cv_split)
    #    model.fit(df_train[train_names], df_train[target_col])

    #   sub_df = pd.DataFrame(columns=submit_cols)
    #    df_test['Survived'] = model.predict(df_test[train_names])
    #    sub_df[submit_cols] = df_test[submit_cols]
    #    sub_df.to_csv("../data/titan/%s_dt.csv" % name, index=False)

        #print(" %s train Accuracy Mean %f" % (name, model_cv['train_score'].mean()))
        #print('%s Test Accuracy Mean %f' % (name, model_cv['test_score'].mean()))
        # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
        # should statistically capture 99.7% of the subsets
        #print('%s Test Accuracy 3*STD %f' % (name, model_cv['test_score'].std()))

    #train_random_forest(df_train,train_names,target_col,cv_split,df_test)
    #rfc = ensemble.RandomForestClassifier(max_features='auto',n_estimators=100,bootstrap=100,criterion='entropy')
    #plot_cv_with_best_param(rfc,df_train,train_names,target_col)

def plot_cv_with_best_param(model,df,fts,target):
    for i in range(10,11):
        cv_results = model_selection.cross_validate(model, df[fts],
                                       df[target],
                                       cv=i)
        print("%d fold"%i)
        print("fit time:",cv_results['fit_time'].mean())
        print("train score mean:",cv_results['train_score'].mean())
        print('test score mean',cv_results['test_score'].mean())
        # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
        # should statistically capture 99.7% of the subsets
        print('test score std',cv_results['test_score'].std() * 3)


def train_random_forest(df,fts,target,cv,df_test):
    submit_cols = ['PassengerId', 'Survived']
    param = {
        'n_estimators':[100,300,500,800,100],
        'max_features':['sqrt','auto'],

        'criterion':['gini','entropy'],
        'bootstrap':[True,False]
    }
    rfc = ensemble.RandomForestClassifier()
    import time
    from sklearn.model_selection import GridSearchCV
    start = time.time()
    rfc_cv,param = common.tune_model(rfc,cv_split=cv,param_grid=param,da=df,fts=fts,label=target)
    model = rfc_cv
    print(model.best_params_)
    sub_df = pd.DataFrame(columns=submit_cols)
    df_test['Survived'] = model.predict(df_test[fts])
    sub_df[submit_cols] = df_test[submit_cols]
    sub_df.to_csv("../data/titan/tuned_randomforest_dt.csv", index=False)
    print("Base Train Accuracy Mean %f" % model.cv_results_['mean_train_score'].mean())
    print('Base Test Accuracy Mean %f' % model.cv_results_['mean_test_score'].mean())
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
    # should statistically capture 99.7% of the subsets
    print('BaseTest Accuracy 3*STD %f' % model.cv_results_['std_test_score'].std())
    end = time.time()
    print("Cost time: %f secs"%(end-start))


    plot_cv_with_best_param(model,df,fts,target)


def train_gbdt(df,fts,target,cv):

    #vote_est, param = common.get_all_tuned_algo()
    submit_cols = ['PassengerId', 'Survived']
    gbdt = ensemble.GradientBoostingClassifier()
    param ={
        # GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
         #'loss': ['deviance', 'exponential'], #default=’deviance’
         'n_estimators':[300,500,1000],
         'learning_rate': [0.1,0.5,0.8],
        # default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
        #    'n_estimators': 300,
        # default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
        # 'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
            'max_depth': [2,4,6,10],  # default=3
            'subsample': [1.0,0.6]
     }

    model, dt_param = common.tune_model(gbdt, cv_split=cv, param_grid=param,
                                              da=df, fts=fts, label=target)
    model.fit(df[fts],df[target])
    print(model.best_params_)
    print("Base Train Accuracy Mean %f" % model.cv_results_['mean_train_score'].mean())
    print('Base Test Accuracy Mean %f' % model.cv_results_['mean_test_score'].mean())
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
    # should statistically capture 99.7% of the subsets
    print('Base Test Accuracy 3*STD %f' % model.cv_results_['std_test_score'].std())
    #perf_model(vote_est,df_train,train_names,target_col,cv_split,submit_cols,df_test)

    #sns.heatmap(df_train.corr(),annot=True)
    #plt.show()


def perf_model(vote_est,df,fts,target,cv,submit_cols,df_test):
    #vote_est, param = common.get_all_tuned_algo()

    #perf_model(vote_est, df, fts, target, cv, submit_cols, df_test)
    for v in vote_est:
        print(v)
    grid_hard = ensemble.VotingClassifier(estimators=vote_est, voting='hard')
    grid_hard_cv = model_selection.cross_validate(grid_hard, df[fts],
                                                  df[target],
                                                  cv=cv)
    grid_hard.fit(df[fts], df[target])
    sub_df = pd.DataFrame(columns=submit_cols)
    df_test['Survived'] = grid_hard.predict(df_test[fts])
    sub_df[submit_cols] = df_test[submit_cols]
    sub_df.to_csv("../data/titan/grid_hard_dt.csv", index=False)
    print("Hard voting Train Accuracy Mean %f" % grid_hard_cv['train_score'].mean())
    print('Hard voting Test Accuracy Mean %f' % grid_hard_cv['test_score'].mean())
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
    # should statistically capture 99.7% of the subsets
    print('Hard voting Test Accuracy 3*STD %f' % grid_hard_cv['test_score'].std())

    grid_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
    grid_soft_cv = model_selection.cross_validate(grid_soft, df[fts],
                                                  df[target],
                                                  cv=cv)
    grid_soft.fit(df[fts], df[target])

    sub_df = pd.DataFrame(columns=submit_cols)

    df_test['Survived'] = grid_soft.predict(df_test[fts])
    sub_df[submit_cols] = df_test[submit_cols]
    sub_df.to_csv("../data/titan/grid_soft_dt.csv", index=False)

    print("Soft voting Train Accuracy Mean %f" % grid_soft_cv['train_score'].mean())
    print('Soft voting Test Accuracy Mean %f' % grid_soft_cv['test_score'].mean())
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
    # should statistically capture 99.7% of the subsets
    print('Soft voting Test Accuracy 3*STD %f' % grid_soft_cv['test_score'].std())

def gen_best_algo(df_train,train_names,target_col):

    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,
                                                random_state=0)
    vote_est,params = common.get_all_algo()
    import time
    for clf,param in zip(vote_est,params):
        model = clf[1]
        name = model.__class__.__name__
        start = time.time()
        print("start tuning %s"%name)
        tuned_model, dt_param = common.tune_model(model, cv_split=cv_split, param_grid=param,
                                            da=df_train, fts=train_names, label=target_col)

        print(dt_param)
        print("%s Train Accuracy Mean %f" % (name,tuned_model.cv_results_['mean_train_score'].mean()))
        print('%s Test Accuracy Mean %f' % (name,tuned_model.cv_results_['mean_test_score'].mean()))
        # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
        # should statistically capture 99.7% of the subsets
        print('%s Test Accuracy 3*STD %f' % (name, tuned_model.cv_results_['std_test_score'].std()))
        end = time.time()
        print("cost %f secs" %(end-start))
    #train_x,test_x,train_y,test_y = train_test_split(df_train[train_names],df_train[target_col])
    #cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,
    #                                        random_state=0)

    #lr = LogisticRegression()
    #lr.fit(train_x,train_y)
    #name='lr'
    #model = lr
    #print(" %s train Accuracy Mean %f" % (name, model.score(train_x,train_y).mean()))
    #print('%s Test Accuracy Mean %f' % (name, model.score(train_x,train_y).mean()))
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
    # should statistically capture 99.7% of the subsets
    #print('%s Test Accuracy 3*STD %f' % (name, model.score(train_x,train_y).std()))


def title_short(df):
    title_map = {"Capt": "Officer",
                 "Col": "Officer",
                 "Major": "Officer",
                 "Jonkheer": "Royalty",
                 "Don": "Royalty",
                 "Sir": "Royalty",
                 "Dr": "Officer",
                 "Rev": "Officer",
                 "the Countess": "Royalty",
                 "Dona": "Royalty",
                 "Mme": "Mrs",
                 "Mlle": "Miss",
                 "Ms": "Mrs",
                 "Mr": "Mr",
                 "Mrs": "Mrs",
                 "Miss": "Miss",
                 "Master": "Master",
                 "Lady": "Royalty"
                 }
    df['title'] = df['title'].map(title_map)
    return df


def compute_val_set(df,col):
    cabin_set = set()
    for cabin in df[col].values:
        names = cabin.split(" ")
        names = [name.strip() for name in names]
        for n in names:
            cabin_set.add(n)
    return cabin_set

def compute_dummies(df,col,col_prefix,col_val_set):
    # cabin encoding
    cabin_set = set()


    n_prefix = col_prefix
    cabin_set = col_val_set


    cabin_dict = dict()
    for idx,cabin in enumerate(cabin_set):
        cabin_dict[cabin] = idx

    cols = [n_prefix+str(i) for i in range(len(cabin_set)+1)]
    tmp_df = pd.DataFrame(columns=cols)

    def val_to_encoding(val):
        values = [0 for i in range(len(cabin_set)+1)]
        names = [ n.strip() for n in val.split(" ")]
        names = filter(lambda  x:len(x)>0,names)
        for n in names :
            if n in cabin_dict:
                values[cabin_dict[n]] = 1
            else:
                values[-1] = 1
        return values
    idx = 0
    for enc in df[col].apply(lambda  x:val_to_encoding(x)):
        tmp_df.loc[idx,cols] = enc
        idx += 1
    df[cols] = tmp_df[cols]
    return df


def cabin_dummies_test():
    df = pd.DataFrame(columns=['Cabin'])
    df.loc[0, 'Cabin'] = "e12"
    df.loc[1, 'Cabin'] = 'e12 e13'
    df.loc[2, 'Cabin'] = 'e13'
    df = compute_dummies(df)
    print(df)

if __name__ == "__main__":
    #main_sub_0206()
    main()

    #get_all_tuned_algo()
    #fill_age_example()