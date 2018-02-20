
import pandas as pd
import datetime

#user_data_dir = "../../data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv.10000"
user_data_dir = "../../data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv"
item_data_dir = "../../data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv"

# 1 browser, 2 collect, 3 addcart, 4 purchase
action_type=[1,2,3,4]


def main_cv_ratio():
    data = pd.read_csv(data_dir,index_col=['user_id','item_id'])
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

def load_data():
    u_data = pd.read_csv(user_data_dir)
    i_data = pd.read_csv(item_data_dir)
    u_data['time'] = pd.to_datetime(u_data['time'])
    return u_data,i_data


def fe_gen():
    u_df, i_df = load_data()

    # train and test split, raw data to fe data
    # cols : user_id    item_id  behavior_type user_geohash  item_category, time
    u_df, i_df = load_data()
    u_df.set_index('time')
    td = datetime.timedelta(weeks=1)
    td_day = datetime.timedelta(days=1)
    fe_data_gap = [(i-td,i) for i  in pd.date_range(start='2014-11-25',end='2014-12-18')]
    gap = fe_data_gap[0]

    x_start,x_end = gap[0],gap[1]-td_day
    print("x start date:",x_start)
    print("x end date:",x_end)
    y_date= gap[1]
    fe_df = pd.DataFrame(columns=['visit_sku_times','add_cart_times','collect_times',
                                  'last_visit_times_by_hour'])
    fe_df = u_df.groupby([''])

    #for gap in fe_data_gap:
    #    x_start,x_end = gap[0],gap[1]-td_day
    #    print("x start date:",x_start)
    #    print("x end date:",x_end)
    #    y_date= gap[1]
    #    print("y date: ", y_date)


    print(u_df[u_df['time']<'2014-11-19'].shape)

    #

    #

def simplest_data_check():
    #
    #  check null exist column

    data = pd.read_csv(user_data_dir)
    item_data = pd.read_csv(item_data_dir)
    print("item data size:",item_data.shape)
    data['time'] = pd.to_datetime(data['time'])
    #data = pd.read_csv(item_data_dir)
    print("total size:",data.shape)
    print(data.isnull().sum())
    print  data.head()

    print("Check user number:",data['user_id'].unique().shape[0])
    print("Check item number:",data['item_id'].unique().shape[0])
    print("Check item category number:",data['item_category'].unique().shape[0])
    data['time'] = pd.to_datetime(data['time'])
    print("min time:",data['time'].min(),"max time:",data['time'].max())





if __name__ == "__main__":
    fe_gen()
