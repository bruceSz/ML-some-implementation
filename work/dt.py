#!/user/bin/python

import math
import time

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
                [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no','surfaceing','flippers']
    return dataSet,labels

def calcShannonEnt(dataSet):
    num_e = len(dataSet)
    labelCounts = {}

    for fv in dataSet:
        currL = fv[-1]
        if currL not in labelCounts:
            labelCounts[currL] = 0
        labelCounts[currL]  += 1
    shannonEnt = 0.0
    for k in labelCounts:
        prob = float(labelCounts[k])/num_e
        shannonEnt -= prob * math.log(prob,2)
    return shannonEnt

def splitDataSet(dataSet, axis, val):
    retDataSet = []
    for fv in dataSet:
        if fv[axis] == val:
            r_fv = fv[:axis]
            r_fv.extend(fv[axis+1:])
            retDataSet.append(r_fv)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    # the last fv is class label.
    nf = len(dataSet[0]) -1
    base_e = calcShannonEnt(dataSet)
    bestGain = 0.0
    best_f = -1
    for i in range(nf):
        # ith feature list
        feat_list = [example[i] for example in dataSet]
        unique_vals = set(feat_list)
        new_e = 0.0
        for v in unique_vals:
            sub_ds = splitDataSet(dataSet, i, v)
            prob = len(sub_ds) / float(len(dataSet))
            new_e += prob*calcShannonEnt(sub_ds)
        info_g = base_e-new_e
        print 'info gain:',info_g,"at :",i
        if info_g > bestGain:
            bestGain = info_g
            best_f = i
    return best_f


def majorityCnt(class_list):
    class_count = {}
    for v in class_list:
        if v not in class_count.keys():
            class_count[v]  = 0
        class_count[v] += 1
    # TODO. fix this.
    return  max(class_count)


def createTree(dataSet, labels):
    classL = [ex[-1] for ex in dataSet]

    # all class are same.
    if classL.count(classL[0]) == len(classL):
        return majorityCnt(classL)

    # only 1 fv left?, just use majority
    if len(dataSet[0]) == 1:
        return majorityCnt(classL)

    best_fv = chooseBestFeatureToSplit(dataSet)
    best_fv_l = labels[best_fv]
    mytree = {best_fv_l:{}}
    del (labels[best_fv])
    fv = [ex[best_fv] for ex in dataSet]
    unique_v = set(fv)
    for v in unique_v:
        sub_l = labels[:]
        #print best_fv
        mytree[best_fv_l][v] = createTree(splitDataSet(dataSet,best_fv,v),sub_l)
    return mytree


def classify(inputTree, featLabels,  testVec):
    firstStr = inputTree.keys()[0]
    sec_dict = inputTree[firstStr]
    feat_idx = featLabels.index(firstStr)
    for k in sec_dict.keys():
        if testVec[feat_idx] == k:
            if type(sec_dict[k]).__name__ == "dict":
                classL = classify(sec_dict[k], featLabels,testVec)
            else:
                classL = sec_dict[k]

    return classL




def storeTree(inputT, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputT, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


def main():
    data, label = createDataSet()
    t1 = time.clock()
    myTree = createTree(data,label)
    t2 = time.clock()
    print myTree
    print 'execute for  ', t2-t1


if __name__ == "__main__":
    main()