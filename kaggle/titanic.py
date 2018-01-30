
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
    train = fill_age_with1(df_train,train_set_1_median,'Age',age_mean)
    #train = fill_age_with(df_train,train_set_1_median,["Pclass", "SibSp"],'Age')
    #train[train['Age'].isnull()]= train['Age'].mean()
    print train.shape
    test_set_1 = df_test.groupby(["Pclass", "SibSp"])
    test_set_1_median = test_set_1.median()
    test = fill_age_with1(df_test,test_set_1_median,'Age',age_mean)

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

def main():
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
    print(data_train_1_x_dummy)
    data_train_1_xy_dummy = Target + data_train_1_dummy.columns.tolist()

    #print("dumy xy:",data_train_1_xy_dummy)
    train_x,train_y,test_x,test_y = model_selection.train_test_split(data_train_1[data_train_1_x_calc],data_train_1[Target],
                                                                     random_state=0)
    train_x_bin,train_y_bin,test_x_bin,test_y_bin = model_selection.train_test_split(data_train_1[data_train_1_x_bin],
                                                                                     data_train_1[Target],random_state=0)
    #train_x_dummy,train_y_dummy,test_x_dummy,test_y_dummy = model_selection.train_test_split(data_train_1[data_train_1_x_dummy],
    #                                                                                         data_train_1[Target],
    #                                                                                         random_state=0)

    for x in data_train_1_x:
        if data_train_1[x].dtype != 'float64':
            print('Survival Correlation by:',x)
            print(data_train_1[[x,Target[0]]].groupby(x,as_index=False).mean())
            print("*"*10,'\n')
    #print(data_train_1.isnull().sum())
    #print("*"*10)
    #print(df_test.isnull().sum())


if __name__ == "__main__":
    main()
    #fill_age_example()