
import numpy as np
from sklearn import datasets

class DataGenerator(object):
    def __init__(self):
        pass

    def gen(self):
        raise NotImplementedError("DataGenerator must be override.")


class TestDataGenerator(DataGenerator):
    def __init__(self):
        DataGenerator.__init__(self)

    def gen(self):
        x = np.array([
            [0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]
        ])
        y = np.array([[1,0,1,1]]).T
        return x, y

class MakeMoonDataGenerator(DataGenerator):
    def __init__(self):
        DataGenerator.__init__(self)

    def gen(self):
        return datasets.make_moons(200,noise=0.2)


