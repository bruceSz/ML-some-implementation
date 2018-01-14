# coding=utf-8

import numpy as np
import random
import os
import struct
from numpy import append,array,int8,uint8,zeros


from matplotlib import pyplot as plt

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def plot_sigmoid():
    x = np.linspace(-8.0,8.0,2000)
    y = sigmoid(x)
    plt.plot(x,y)
    plt.show()


class NeuralNet(object):

    def __init__(self,sizes):
        self.sizes_ = sizes
        self.num_layers_ = len(sizes)
        self.w_ = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        self.b_ = [np.random.randn(y,1) for x,y in zip(sizes[:-1],sizes[:-1])]


    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    # 导数 of sigmoid
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1-self.sigmoid(z))

    def feedforward(self, x):
        for b,w in zip(self.b_,self.w_):
            x = self.sigmoid(np.dot(w,x) + b)
        return x

    def SGD(self,training_data,epochs, mini_batch_size,eta,test_data = None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0,n,mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {0} complete".format(j))


    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.b_, self.w_):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1],y) * \
            self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2,self.num_layers_):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.w_[-l+1].transpose(), delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            # TODO.
        return (nabla_b,nabla_w)


    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]
        # 1 for each x, y in mini_batch
        # 2 compute all delta_b, delta_w
        # 3 accumulate all this delta_b, delta_w as nabla_b,nabla_w
        # 4 update w and b with nb and nw
        # 5 note: eta is divided by length of batch.
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [np+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]

        self.w_ = [w-(eta/len(mini_batch)) * nw for w,nw in zip(self.w_,nabla_w)]
        self.b_ = [b-(eta/len(mini_batch)) * nb for w,bn in zip(self.b_,nabla_b)]


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)),y) for x,y in test_data]
        return sum(int(x == y) for x,y in test_results)

    def predict(self, data):
        value = self.feedforward(data)
        return value.tolist().index(max(value))














    def cost_derivative(self, output_a, y):
        return (output_a - y)


def load_mnist(dataset = "training_data", digits = np.arange(10),path="."):
    if dataset == "training_data":
        fname_image = os.path.join(path, "train-images-idx3-ubyte")
        fname_label = os.path.join(path, "train-labels-idx1-ubyte")
    elif dataset == "testing_data":
        fname_image = os.path.join(path,'t10k-images-idx3-ubyte')
        fname_label = os.path.join(path,'t10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data' ")

    flbl = open(fname_label,'rb')
    magic_nr, size = struct.unpack(">II",flbl.read(8))
    lbl = pyarray("b",flbl.read())
    flbl.close()

    fimg = open(fname_image,"rb")
    magic_nr, size,rows,cols = struct.unpack(">IIII", fimg.read(16))
    img  = pyarray("B",fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N,rows,cols), dtype=uint8)
    lables = zeros((N,1),dtype=uint8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i]*rows*cols]:(ind[i]+1)*rows*cols]).reshape((rows,cols))
        labels[i] = lbl[ind[i]]
    return images, lables

def load_samples(dataset="training_data"):
    image,label = load_mnist(dataset)
    X = [np.reshape(x,(28*28,1)) for x in image]
    X = [x/255.0 for x in X]
    def vectorized_Y(y):
        e= np.zeros((10,1))
        e[y] = 1.0
        return e
    if dataset == "training_data":
        Y = [vectorized_Y(y) for y in label]
        pair = list(zip(X,Y))
        return pair
    elif dataset == "testing_data":
        pair = list(zip(X,label))
        return pair
    else :
        print("Something wrong")


def neutran_net():
    net = NeuralNet([3,4,2])
    print('weight: ', net.w_)
    print(' biases: ',net.b_)


def main():
    #neutran_net()
    plot_sigmoid()

if __name__ == "__main__":
    main()