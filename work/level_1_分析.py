import pandas as pd
import numpy as np

def main():
    da = pd.read_csv("/Users/brucesz/Documents/level_1_stat.out")
    print da.head(1)
    print da.shape
    #da.convert_objects(convert_numeric=True)
    da["cv_ratio"] = da["cv_ratio"].astype(np.float64)
    print "after filter with cv"
    print 0.2*0.04
    print da[da["cv_ratio"]>0.007].shape


if __name__ == "__main__":
    main()