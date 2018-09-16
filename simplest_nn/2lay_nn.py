
import numpy as np
import data_generator

def nonlin(x, deriv=False):
    if(deriv == True):
        return x(1-x)
    return 1/(1+np.exp(-x))


def twoLayer_nn():
    tdg = data_generator.TestDataGenerator()
    X,y = tdg.gen()
    np.random.seed(1)
    # -1 ~ 1 random value
    syn0 = 2*np.random.random((3,1)) - 1
    # number of iter num
    for iter in xrange(1000):
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))
        l1_error = y-l1
        l1_delta = l1_error * nonlin(l1,True)


def main():
    twoLayer_nn()

if __name__ == "__main__":
    main()