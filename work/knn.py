#!/bin/python

import numpy as np
import os

import operator

class KNN:

    def createDateSet(self):
        group = np.array([[1.0,0.9],[1.0,1.0],[0.1,0.2],[0.0,0.1]])
        labels = ['A','A','B','B']
        return group,labels

    def kNNClassify(self,newInput,dataSet,labels, k):
        numSamples = dataSet.shape[0]
        diff = np.tile(newInput,(numSamples,1)) - dataSet
        squaredDiff = diff ** 2
        squaredDist = np.sum(squaredDiff,axis=1)
        distance = squaredDist ** 0.5

        sortedDisIndices = np.argsort(distance)

        classCount = {}
        for i in range(k):
            voteLabel = labels[sortedDisIndices[i]]
            classCount[voteLabel] = classCount.get(voteLabel,0) + 1


        maxCount = 0
        for k, v in classCount.items():
            if v > maxCount:
                maxCount = v
                mIdx = k
        return mIdx


    def img2vec(self,filename):
        rows = 32
        cols = 32
        imgvec = np.zeros((1,rows*cols))
        fileIn = open(filename)
        for row in range(rows):
            lineStr = fileIn.readline()
            for col in range(cols):

                imgvec[0,row*32+col] = int(lineStr[col])

        return imgvec

    def loadDateSet(self):
        print("getting data set")
        dir = "///"
        files = os.listdir(dir)
        numSamples = len(files)

        trainx = np.zeros((numSamples,1024))
        trainy = []
        for i in range(numSamples):

            filename = files[i]
            trainx[i,:] = self.img2vec(dir + "/"+ filename)
            label = int(filename.split('_')[0])
            trainy.append(label)

        print('getting testing data')


        test_dir = "./test"
        t_files = os.listdir(test_dir)
        numSamples_t = len(t_files)
        testx_t = np.zeros((numSamples_t,1024))
        test_y_t = []

        for i in xrange(numSamples_t):

            filename = t_files[i]
            testx_t[i,:] = self.img2vec(test_dir)
            label = int(filename.split('_')[0])
            test_y_t.append(label)

        return trainx,trainy,testx_t,test_y_t


def mk_test2():
    trainx,trainy,test_x,test_y = KNN().loadDateSet()

    match = 0
    for i in range(test_x.shape[0]):

        predict = KNN.kNNClassify(test_x[i],trainx,trainy,3)
        if predict == test_y[i]:
            match += 1
    accuracy = float(match) / test_x.shape[0]
    print('accuracy is :',accuracy)


def mk_test1():
    dataSet, labels = KNN().createDateSet()
    testX = np.array([1.2,1.0])
    k = 3
    outLabel = KNN().kNNClassify(testX,dataSet,labels,3)

    print('your input is ',testX,'and class should be:',outLabel)

    testX = np.array([0.1,0.3])
    outLabel = KNN().kNNClassify(testX,dataSet,labels,3)
    print('your input is ', testX, 'and class should be:', outLabel)


def main():
    mk_test1()



if __name__ == "__main__":
    main()