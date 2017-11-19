
import numpy as np
import matplotlib.pyplot as plt

import random

test_file="../data/lr_test_set.txt"
def loadDataSet():
    dataMat = []
    labelMat = []
    fr= open(test_file)
    for l in fr.readlines():
        l_arr = l.strip().split()
        dataMat.append([1.0,float(l_arr[0]),float(l_arr[1])])
        labelMat.append(int(l_arr[2]))

    return dataMat, labelMat

def selectJrand(i,m):
    j = i
    while (j == i):
        j = int(random.uniform(0,m))
    return j


def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLables, C, toler, maxIter):
    dataMtrix = np.mat(dataMatIn)
    labelMat = np.mat(classLables).transpose()
    b = 0
    m,n = np.shape(dataMtrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # outer loop
            fXi = float(np.multiply(alphas,labelMat).T*(dataMtrix*dataMtrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i]<C)
                or ((labelMat[i]*Ei>toler) and (alphas[i]>0))):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T*(dataMtrix*dataMtrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                if (labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j]-alphas[j])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[i]+alphas[j] -C)
                    H = min(C,alphas[j]+alphas[i])
                eta = 2.0* dataMtrix[i,:] * dataMtrix[j,:].T
                -dataMtrix[i,:]*dataMtrix[i,:].T-dataMtrix[j,:]*dataMtrix[j,:].T
                if eta >= 0:
                    print "eta>=0"
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)

                alphas[i] += labelMat[i]*labelMat[j]*(alphaJold-alphas[j])

                b1 = b-Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMtrix[i,:]*dataMtrix[i,:].T- labelMat[j]*(alphas[j]-alphaJold)*dataMtrix[j,:]*dataMtrix[i,:].T
                b2 = b-Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMtrix[i,:]*dataMtrix[j,:].T- labelMat[j]*(alphas[j]-alphaJold)*dataMtrix[j,:]*dataMtrix[j,:].T
                if (0<alphas[i]) and (C>alphas[i]):
                    b = b1
                elif (0<alphas[j]) and (alphas[j]<C):
                    b= b2
                else:
                    b = (b1+b2)/2.0
                alphaPairsChanged+=1
        if (alphaPairsChanged == 0):
            iter +=1
        else:
            iter = 0
    return b, alphas




def matplot(dataM, labelM):
    xcord1 = []
    ycord1 = []

    xcord2 = []
    ycord2 = []

    xcord3 = []
    ycord3 = []

    for i in range(100):
        if labelM[i] == 1:
            xcord1.append(dataM[i][0])
            ycord1.append(dataM[i][1])
        else:
            xcord2.append(dataM[i][0])
            ycord2.append(dataM[i][1])

    b,alphas = smoSimple(dataM,labelM,0.6,0.001,40)

    for j in range(100):
        if alphas[j]>0:
            xcord3.append(dataM[j][0])
            ycord3.append(dataM[j][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xcord1,ycord1, s= 30, c='red', marker='s')
    ax.scatter(xcord2,ycord2, s=30, c='green')
    ax.scatter(xcord3,ycord3, s=80, c='blue')
    ax.plot()
    plt.xlabel('X1')
    plt.ylabel('X1')
    plt.show()


if __name__ == '__main__':
    data_file = "../data/lr_test_set.txt"
    dataM, labelM = loadDataSet()
    matplot(dataM, labelM)
