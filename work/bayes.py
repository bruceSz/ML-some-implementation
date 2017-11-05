#!/usr/bin/python
import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_v = [0,1,0,1,0,1]
    return postingList, class_v


def createVList(dataSet):
    vSet = set([])
    for doc in dataSet:
        vSet = vSet|set(doc)
    return list(vSet)


def setOfWordsVec(vList,InputSet):
    retVlist = [0]*len(vList)
    for w in InputSet:
        if w in vList:
            retVlist[vList.index(w)] = 1
        else:
            print 'word ', w, "not in dict"
    return retVlist


def bagOfW2Vec(vList, InputSet):
    retV  = [0] * len(vList)
    for w in InputSet:
        if w in vList:
            retV[vList.index(w)] +=1
    return retV


def trainBN0(trainM, trainC):
    numDoc = len(trainM)
    numWord = len(trainM[0])
    pAbusive = sum(trainC)/float(numDoc)
    p0Num = np.ones(numWord)
    p1Num = np.ones(numWord)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numDoc):
        if trainC[i] == 1:
            p1Num += trainM[i]
            p1Denom += sum(trainM[i])
        else:
            p0Num += trainM[i]
            p0Denom += sum(trainM[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2C, p0V, p1V, pClass):
    p1 = sum(vec2C * p1V) + np.log(pClass)
    p0 = sum(vec2C * p0V) + np.log(1.0-pClass)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listPosts, listC = loadDataSet()
    myVList = createVList(listPosts)
    trainM = []
    for post in listPosts:
        trainM.append(setOfWordsVec(myVList, post))
    p0v, p1v, pAb = trainBN0(np.array(trainM),np.array(listC))
    testE = ['love','my','dalmation']
    thisdoc = np.array(setOfWordsVec(myVList,testE))
    print testE, 'classified as : ',classifyNB(thisdoc, p0v,p1v,pAb)
    testE = ['stupid','garbage']
    thisdoc = np.array(setOfWordsVec(myVList,testE))
    print testE,"classfied as :", classifyNB(thisdoc, p0v, p1v, pAb )


def main():
    testingNB()

if __name__ == "__main__":
    main()