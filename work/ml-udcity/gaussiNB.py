

def classify(label,train):
    from sklearn.naive_bayes import GaussianNB
    cls = GaussianNB()
    cls.fit(train,label)
    return cls

def main():
    l1 = [1,2,3]
    l2 = [1,2,4]
    acc = sum([l1[i] == l2[i] for i in range(len(l1))])/len(l1)
    print(acc)

if __name__ == "__main__":
    main()