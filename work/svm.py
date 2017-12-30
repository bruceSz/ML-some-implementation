
import random

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i,m):
    j = i
    while(j == i):
        i = int(random.uniform(0,m))
    return j


def clipAppha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    pass

def calcKernelVal(mat, sample_x, kernelOp):
    kernelType = kernelOp[0]()

def main():
    pass

if __name__ == "__main__":
    main()