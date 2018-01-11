
import time
import pandas as pd
from enum import Enum

class SQLPAN_OP(Enum):
    AVG = 1
    COUNT = 2
    MAX = 3
    MIN = 4
    STD = 4

class SQLPAN_HAVING_OP(Enum):
    EQ = 1
    GT = 2
    LT = 3

class SQLPAN(object):
    def __init__(self, da):
        #if indexes is None:
        #    self._da = pd.DataFrame(columns = columns)
        #else:
        #    self._da = pd.DataFrame(index=indexes, columns=columns)
        #    self._da.append()
        self._da = da

    def get(self):
        return self._da

    def group_by(self,group_by_cols, op=None,op_column=None):
        if not isinstance(op,SQLPAN_OP):
            raise RuntimeError("Error: group by op:%s is not supported."%(op))
        else:
            if op == SQLPAN_OP.AVG:
                return self._da.group_by(group_by_cols)[op_column].mean()
            elif op == SQLPAN_OP.COUNT:
                return self._da.group_by(group_by_cols)[op_column].count()
            elif op == SQLPAN_OP.MAX:
                return self._da.group_by(group_by_cols)[op_column].max()
            elif op == SQLPAN_OP.MIN:
                return self._da.group_by(group_by_cols)[op_column].min()
            elif op == SQLPAN_OP.STD:
                return self._da.group_by(group_by_cols)[op_column].std()
            else:
                raise RuntimeError("Unsupported SQLPN op type:%s"%op)

    def group_having(self, group_cols, having_cols,having_vals, having_ops):
        pass
    
    def limit(self,number):
        return SQLPAN(self._da.head(number))



    def order_by(self, target_columns  = None):
        return SQLPAN(self._da.sort_values(by=target_columns))

    def distinct(self, target_columns = None):
        return SQLPAN(self._da.drop_duplicates(subset=target_columns))

    def union(self,other):
        return SQLPAN(pd.concat([self._da,other.get()]))

    def join(self, other):
        return SQLPAN(pd.concat([self._da,other.get()],axis=1))

def compare_append():
    da1 = pd.DataFrame(columns=['a', 'b', 'c'])
    #da1.sort_values
    total = 3000
    start_t = time.time()
    for i in range(total):
        da1 = da1.append({'a': 1, 'b': 2, 'c': 3}, ignore_index=True)
    print("len after append:%d"%(len(da1)))
    end_t = time.time()
    print("append 1000 times Cost %f"%(end_t-start_t))

    start_t = time.time()
    da2 = pd.DataFrame(columns=['a','b','c'],index=[i for i in range(total)])
    for i in range(total):
        da2.iloc[i] = pd.Series({'a':1,'b':2,'c':3})
    end_t = time.time()
    print("iloc set 1000 times Cost %f" % (end_t - start_t))
def main():
    #1. join
    da1 = pd.DataFrame(columns=['a','b','c'])
    #da1.sort_values
    da1 = da1.append({'a':1,'b':2,'c':3},ignore_index=True)
    da2 = pd.DataFrame([[2,1,4]],columns=['a','b','c'])
    #da1.iloc[da1['a'].count()] = pd.Series([2,3,4])
    s1 = SQLPAN(da1)
    s2 = SQLPAN(da2)
    # join
    other = s1.union(s2)
    #print(other.distinct(target_columns=['a']).get())
    print(other.limit(1).get())


if __name__ == "__main__":
    main()
    #compare_append()