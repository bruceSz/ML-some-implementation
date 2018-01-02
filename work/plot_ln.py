import random
import matplotlib.pyplot as plt
import math

def main():
    r = 20
    a = [ math.log(random.uniform(0,4)) for i in range(100)]
    a.sort()
    a_i = [i for i in range(100)]
    plt.scatter(a_i,a)
    plt.show()



if __name__ == "__main__":
    main()