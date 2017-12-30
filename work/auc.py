
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def main():
    y = np.array([1,1,2,2])
    pred = np.array([0.1,0.4,0.35,0.8])
    fpr,tpr,thresholds = metrics.roc_curve(y,pred,pos_label=2)
    print(thresholds)
    print(fpr)
    print(tpr)
    roc_auc = metrics.auc(fpr,tpr)
    plt.clf()
    plt.plot(fpr,tpr, label='ROC curve (area = %0.2f)'%roc_auc)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()