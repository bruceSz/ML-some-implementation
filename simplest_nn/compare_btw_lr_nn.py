

import random
import string
import hashlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib
import sklearn.linear_model

import data_generator

def random_str(length):
    return ''.join(random.choice(string.ascii_letters) for x in range(length))


def gen_md5(tobe_md5,length):

    m1 = hashlib.md5()
    m1.update(tobe_md5)
    return m1.hexdigest()[:length]


def lr_model_moon_data():

    dg = data_generator.MakeMoonDataGenerator()
    X, y = dg.gen()
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=cm.spectral)
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    s_x = sorted(X,cmp=lambda x,y : cmp(x[0], y[0]))
    print clf.predict_proba(s_x[0])
    l = [(s_x[0],s_x[-1]),(clf.predict(s_x[0]), clf.predict(s_x[-1]))]
    (line1_xs, line1_ys) = zip(*l)
    #y_p = [clf.predict(x) for x in X]
    figure, ax = plt.subplots()
    ax.add_line(Line2D(line1_xs, line1_ys))
    plt.plot()
    plt.show()


def main():
    lr_model_moon_data()
    #print gen_md5("this is a md5 test", 10)

if __name__ == "__main__":
    main()