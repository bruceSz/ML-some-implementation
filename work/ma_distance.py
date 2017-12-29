
import numpy as np

def ma_dis():
    x = np.random.random(10)
    y = np.random.random(10)

    X = np.vstack((x, y))
    XT = X.T
    a = X
    b = XT

    S = np.cov(a)
    print(S)
    SI = np.linalg.inv(S)
    n = b.shape[0]
    print("Number of train samples" + str(n))
    print()
    d1 = []
    for i in range(0, n):
        tmp = []
        for j in range(i + 1, n):
            delta = b[i] - b[j]
            fh = np.dot(delta, SI)
            d = np.sqrt(np.dot(fh, delta.T))
            tmp.append(d)
        d1.append(tmp)
    print("Together dis number: %d" % len(d1))
    from scipy.spatial.distance import pdist
    print(d1)
    d2 = pdist(b, 'mahalanobis')
    print(d2)

def cov(da):
    mean_ = np.mean(da,axis=0)
    da_diff = da - mean_
    cov_mat = da_diff.T.dot(da_diff)/(da_diff.shape[0]-1)
    return cov_mat


def cov_man():
    a = np.random.random((10,2))
    print(a)
    cov_mat_np = np.cov(a,rowvar=False)
    print("Matrix by using Numpy:")
    print(cov_mat_np)
    my_cov_m = cov(a)
    print("Matrix by me")
    print(my_cov_m)


def main():
    cov_man()

if __name__ == "__main__":
    main()