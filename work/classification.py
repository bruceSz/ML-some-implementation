
import pandas as pd
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']



def main():
    pima = pd.read_csv(url, header=None,names = col_names)
    ft_cols = ['pregnant','insulin','bmi','age']
    X = pima[ft_cols]
    y = pima.label

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train,y_train)

    y_pred_class = logreg.predict(X_test)
    print(metrics.accuracy_score(y_test,y_pred_class))
    print(y_test.mean())
    print(y_test.value_counts())
    confusion = metrics.confusion_matrix(y_test,y_pred_class)
    TP = confusion[1,1]
    TN = confusion[0,0]
    FP = confusion[0,1]
    FN = confusion[1,0]
    print((TP+TN)/float(TP+TN+FP+FN))
    print(metrics.accuracy_score(y_test,y_pred_class))

    print(TP/float(TP+FN))
    print(metrics.recall_score(y_test,y_pred_class))

    print(TP/float(TP+FP))
    print(metrics.precision_score(y_test,y_pred_class))


    print("predict ret and proba:")
    print(logreg.predict(X_test)[0:10])
    print(logreg.predict_proba(X_test)[0:10,0])
    y_pred_prob = logreg.predict_proba(X_test)[:,1]

    plt.rcParams['font.size'] = 14

    #plt.hist(y_pred_prob,bins=8)
    #plt.xlim(0,1)
    #plt.title("Hist of prediction proba")
    #plt.xlabel('Predicted proba')
    #plt.ylabel('Frequency')
    #plt.show()
    y_pred_class = binarize([y_pred_prob],0.3)[0]
    print(confusion)
    print(metrics.confusion_matrix(y_test,y_pred_class))

    print(y_pred_class)

    fpr,tpr,thresholds = metrics.roc_curve(y_test,y_pred_prob)
    plt.plot(fpr,tpr)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.title('Roc curve for diabetes classifier')
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()