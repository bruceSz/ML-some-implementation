
from numpy import *
import numpy as np
import pandas as pd
from math import log
import operator


class oldCart(object):

    def calcGini(dataSet):
        numEntries = len(dataSet)
        labelsC = {}



    def cart_class():
        pass


    def cart_regre():
        pass

def loadData(filename):
    ''' '''
    dm = []
    with open(filename) as fr:
        for l in fr.readlines():
            cutL = l.strip().split("\t")
            floatL = map(float,cutL)
            dm.append(floatL)
    return dm


def binarySplitDataSet(data, feature, val):
    matL = data[np.nonzero(data[:,feature] <= val)[0],:]
    matR = data[np.nonzero(data[:,feature] > val)[0],:]
    return matL, matR


def regressLeaf(da):
    """ """

    return np.mean(da[:,-1])


def regressErr(da):
    # mean as the output, then var*no_node = mse
    return np.var(da[:,-1])*np.shape(da)[0]


def regressDa(filename):
    fr = open(filename)
    return pickle.load(fr)


def chooseBestSplit(da, leafType=regressLeaf,errType =regressErr, threshold=(1,4)):
    thresholdErr = threshold[0]
    thresholdSamples = threshold[1]
    if len(set(da[:,-1].T.tolist()[0])) == 1:
        return None,leafType(da)

    m, n = np.shape(da)
    Err = errType(da)

    bestErr = np.inf
    bestFtIndex = 0
    bestFtVal = 0
    for ft_idx in range(n-1):
        for ft_val in da[:,ft_idx]:
            matL, matR = binarySplitDataSet(da, ft_idx, ft_val)
            if (np.shape(matL)[0]<thresholdSamples or
                np.shape(matR)[0]<thresholdSamples):
                continue

            temErr = errType(matL) + errType(matR)
            if temErr < bestErr:
                bestErr = temErr
                bestFtIndex = ft_idx
                bestFtVal = ft_val

    # wether the diff between thisErr and Err is smaller than that of threshold.
    if (Err - bestErr) < thresholdErr:
        return None,leafType(da)

    matL, matR = binarySplitDataSet(da,bestFtIndex,bestFtVal)
    if (np.shape(matL)[0] < thresholdSamples or
                np.shape(matR)[0] < thresholdSamples):
        return None,leafType(da)

    return bestFtIndex,bestFtVal


def createCartTree(da,leafType=regressLeaf,errType=regressErr,threshold=(1,4)):
    """ """
    ft, val = chooseBestSplit(da,leafType,errType,threshold)
    if ft == None:
        return val

    returnTree = {}
    returnTree['bestSplitFt'] = ft
    returnTree['bestSplitVal'] = val
    leftSet,rightSet = binarySplitDataSet(da,ft,val)
    returnTree['left'] = createCartTree(leftSet,leafType,errType,threshold)
    returnTree['right'] = createCartTree(rightSet,leafType,errType,threshold)
    return returnTree


def isTree(obj):
    return(type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right'])/2.0

def prune(tree, testDa):
    if np.shape(testDa)[0] == 0:
        return getMean(tree)
    if isTree(tree['left']) or isTree(tree['right']):
        leftTestD, rightTestD = \
            binarySplitDataSet(testDa,tree['bestSplitFt'],tree['bestSplitVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],leftTestD)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rightTestD)
    if not isTree(tree['left']) and not isTree(tree['right']):
        leftTestD, rightTestD =\
            binarySplitDataSet(testDa, tree['bestSplitFt'], tree['bestSplitVal'])
        errorNoMerge = sum(np.power(leftTestD[:,-1]-tree['left'],2)) + \
            sum(np.power(rightTestD[:,-1]-tree['right'],2))
        errorMerge = sum(np.power(testDa[:,-1] -getMean(tree),2))
        if errorMerge < errorNoMerge:
            print 'Merging'
            return getMean(tree)
        else: return tree
    else:
        return tree

def linearSolve(da):
    m,n = np.shape(da)
    X = np.mat(np.ones((m,n)))
    Y = np.mat(np.ones((m,1)))
    # TODO. explain this.
    X[:,1:n] = da[:,0:n-1]
    Y = da[:,-1]
    xtx = X.T*X
    if np.linalg.det(xtx) == 0:
        raise NameError('This matrix is singular, cannot do inverse'
                        'try increase the second value of the threshold')
    ws = xtx.I * (X.T*Y)
    return ws,X,Y


def modelLeaf(da):
    wx,X,Y = linearSolve(da)
    return wx

def modelErr(da):
    wx,X,Y = linearSolve(da)
    yHat = X * wx
    return sum(np.power(Y-yHat,2))

def regressEval(tree, inputData):
    return float(tree)

def modelTreeEval(model,inputData):
    n = np.shape(inputData)
    # TODO. explain this: why need 1:n+1
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1] = inputData
    return float(X*model)

def treeForeCast(tree, inputData, modelEval = regressEval):
    if not isTree(tree):
        return modelEval(tree,inputData)
    if inputData[tree['bestSplitFt']] < tree['bestSplitVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inputData,modelEval)
        else:
            return modelEval(tree['left'],inputData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inputData,modelEval)
        else:
            modelEval(tree['right'],inputData)

def createForeCast(tree, testDa, modelEval = regressEval):
    m = len(testDa)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i] = treeForeCast(tree,testDa[i],modelEval)
    return yHat


def main():
    a = np.array([[1,2],[0,1],[2,0]])
    b = np.nonzero(a[:,1]<=1)
    print(type(a))
    print(a[:,-1].T.tolist())



if __name__ == "__main__":
    main()