
import pandas as pd
data_dir = "/Users/brucesz/Desktop/fresh_comp_offline/tianchi_fresh_comp_train_user.csv.10000"

# 1 browser, 2 collect, 3 addcart, 4 purchase
action_type=[1,2,3,4]


def main_cv_ratio():
    data = pd.read_csv(data_dir)
    #print  data.head()
    #print  data.columns
    #print data.index
    da1 = data.set_index(['user_id','item_id'])
    #print da1.index
    df = pd.DataFrame()
    s_total = []
    #groupby_user_product = da1.groupby(['user_id','item_id'])
    for user , group in  da1.groupby(['user_id']):
        total = group.count()['behavior_type']
        purchase = group['behavior_type'].value_counts()
        if 4 in purchase:
            purchase = purchase[4]
            s_t = [user,float(purchase)/total]
            s_total.append(s_t)

    print max(s_total,key=lambda x:x[1])
    print min(s_total,key=lambda  x:x[1])
    def xx(x, y):
        return [0,x[1]+y[1]]
    print reduce(xx, s_total,[0,0])[1]/len(s_total)


def main():
    data = pd.read_csv(data_dir)
    print  data.head()


if __name__ == "__main__":
    main()
