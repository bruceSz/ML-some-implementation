#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math

def loadSimpleData():
    dataMat = np.matrix([[1.,2.1],
                         [2.,1.1],
                         [1.3, 1.],
                         [1.,1.],
                         [2.,1.]])
    classL = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classL


def stumpClassify(dataM, dimen, thresholdV, threshIneq):
    retArr = np.ones((np.shape(dataM)[0],1))
    if threshIneq == 'lt':
        retArr[dataM[:,dimen]<=thresholdV] = -1.0
    else:
        retArr[dataM[:,dimen]>thresholdV] = -1.0
    return retArr


def buildStump(dataArr, classL, D):
    dataM = np.mat(dataArr)
    labelM = np.mat(classL).T
    m,n = np.shape(dataM)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    # traverse the nth ft.
    for i in range(n):
        rangeMin = dataM[:,i].min()
        rangeMax = dataM[:,i].max()
        stepS = (rangeMax-rangeMin)/numSteps
        # curr ft's all step
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshV = (rangeMin + float(j)*stepS)
                predictVals = stumpClassify(dataM,i,threshV,inequal)
                errorArr = np.mat(np.ones((m,1)))
                errorArr[predictVals == labelM] = 0
                weightedError = D.T * errorArr
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump["thresh"] = threshV
                    bestStump["ineq"] = inequal
    return bestStump,minError,bestClassEst


def adaBoostTrainDS(dataArr, classL, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump, error,classEst = buildStump(dataArr,classL,D)
        print 'D:',D.T
        alpha = float(0.5*math.log((1-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        expon = np.multiply(-1*alpha*np.mat(classL).T,classEst)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()

        aggClassEst += alpha*classEst

        aggErrors = np.multiply(np.sign(aggClassEst)!=np.mat(classL).T,np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print 'total error:',errorRate
        bestStump['error rate'] = errorRate
        weakClassArr.append(bestStump)
        if errorRate == 0.0:
            break
    return weakClassArr,aggClassEst




def main():
    data,classL = loadSimpleData()
    adaBoostTrainDS(data,classL)

if __name__ == "__main__":
    main()