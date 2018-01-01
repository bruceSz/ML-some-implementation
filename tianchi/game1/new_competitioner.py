
import pandas as pd
from sklearn.linear_model import LogisticRegression

_OP = [1,2,3,4]
# visit,collect,add_cart,purchase

_USER_TRAIN_DATA = "../../data/fresh_comp_offline/tianchi_fresh_comp_train_user100000.csv"
_user_cols = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']

_ITEM_DATA = "../../data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv"
_item_cols = ['item_id', 'item_geohash', 'item_category']

_fts = ['user_id','item_id','visit_times','cart_times','collect_times','latest_visit_interval','is_purchase']

_P_TIME_LEN = 14

def collect_visit_times_before_purchase(da_usr):
    indexes = da_usr[da_usr['behavior_type']==4].index



def ft_gen(da_usr):
    # 1. bt:4 as target , and agg user bt features .
    ft_visit_times  = collect_visit_times_before_purchase(da_usr)
    pass


def usr_da_discovery():
    da = pd.read_csv(_USER_TRAIN_DATA)
    print(da.head())


def usr_item(raw_usr_da):
    usr_item_df_d = raw_usr_da[['user_id','item_id']]
    usr_item_df = usr_item_df_d.drop_duplicates()
    return usr_item_df


def groupby_ui_df_dict(usr_da):
    ui_dict = {}
    usr_item_df = usr_item(usr_da)
    for idx,row in usr_item_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        target_df1 = usr_da[usr_da['user_id'] == user_id]
        target_df = target_df1[target_df1['item_id'] == item_id]
        #target_df = usr_da[usr_da['user_id'] == user_id and usr_da['item_id'] == item_id]
        ui_dict[(user_id,item_id)] = target_df
    return ui_dict


#def get_trained_usr_item_pair(usr_da):
#    purchase_df = usr_da[usr_da['behavior_type']==4 ]
#    usr_item_df = purchase_df[['user_id','item_id']]
#    ret = set([])
#    # TODO. only considered the latest purchase here.
#    for idx, row in usr_item_df.iterrows():
#        u_id = row['user_id']
#        i_id = row['item_id']
#        ret.add((u_id,i_id))#
#
#    return ret

def extract_train_raw_df(usr_item_df_dict_all, trained_usr_item_pair_set):
    ret = {}
    #ret = pd.DataFrame(columns=['user_id','item_id','visit_times'])
    for usr_item_p, df in usr_item_df_dict_all.iteritems():
        #user_id = usr_item_p[0]
        #item_id = usr_item_p[1]
        if usr_item_p in trained_usr_item_pair_set:
            time_sorted = df.sort_values(by='time',ascending=False)
            has_met = False
            start_t = None
            val_df = pd.DataFrame(columns=['behavior_type', 'user_geohash', 'item_category', 'time'])
            # TODO. should consider there is a case when multi purchase are there.
            for iidex , row in time_sorted.iterrows():
                if not has_met and row['behavior_type'] == 4:
                    has_met = True
                    val_df = val_df.append(row[['behavior_type', 'user_geohash', 'item_category', 'time']],ignore_index=True)
                    start_t = row['time']

                # already met these
                elif has_met and row['time'] < start_t - pd.Timedelta(days=14):
                    break
                elif not has_met:
                    continue
                else:
                    val_df = val_df.append(row[['behavior_type', 'user_geohash', 'item_category', 'time']],ignore_index=True)
            #print("length of his for %s, is : %d"%(usr_item_p,len(val_df)))
            if len(val_df)  > 1:
                ret[usr_item_p] = val_df
    return ret


def compute_trained_ft(usr_item_df_dict):
    ret_df = pd.DataFrame(
        columns=['user_id','item_id','visit_times','cart_times','collect_times','latest_visit_interval','is_purchase'])

    # 1. extract all positive ft
    for usr_item_p,df in usr_item_df_dict.iteritems():
        visit_t = 0
        cart_t = 0
        collect_t = 0
        latest_visit_interval_days = 0
        # TODO. days? price,etc?

        has_purchase_times = df[df['behavior_type']==4].count()['behavior_type']
        # no purchase record for this item
        if has_purchase_times <=0 :
            continue
        else:
            purchase_his = df[df['behavior_type']==4]
            for idx, row in  purchase_his.iterrows():
                curr_t = row['time']
                begin_t = curr_t- pd.Timedelta(days=14)
                tmp_df = df[df['time']>begin_t & df['time']<=curr_t]
                tmp_df = tmp_df.sort_values(by='time',ascending=False)
                tmp_ret = pd.DataFrame(columns=_fts)
                for idx,row in tmp_df.iterrows():
                    if row['behavior_type']

        # visit:1
        visit_t = df[df['behavior_type']==1].count()['behavior_type']
        # collect:2
        collect_t = df[df['behavior_type']==2].count()['behavior_type']
        # cart:3
        cart_t = df[df['behavior_type'] ==3 ].count()['behavior_type']

        if visit_t > 0:

            # latest_visit_interval
            visit_df = df[df['behavior_type']==1].sort_values(by='time',ascending=False)
            last_visit_t = visit_df.iloc[0]['time']

            purchase_df = df[df['behavior_type']==4].sort_values(by='time',ascending=False)
            purchase_t = purchase_df.iloc[0]['time']
            latest_visit_interval_days = (purchase_t - last_visit_t).days
        else:
            # 0 is valid for all models?
            latest_visit_interval_days = 0
        tmp_d = {
            'user_id':usr_item_p[0],
            'item_id':usr_item_p[1],
            'visit_times':visit_t,
            'cart_times':cart_t,
            'collect_times':collect_t,
            'latest_visit_interval':latest_visit_interval_days,
            'is_purchase':1
        }
        tmp_s = pd.Series(tmp_d)
        ret_df.append(tmp_s,ignore_index=True)
    # 2. compute all negative ft.

    # 3. concat them together.
    return ret_df


def compute_test_usr_item(usr_da):
    max_time = usr_da['time'].max()
    tl = pd.Timedelta(days=14)
    end_t = max_time - tl

    test_df = usr_da[usr_da['time']>end_t]

    usr_item_df = test_df[['user_id', 'item_id']]
    usr_item_df = usr_item_df.drop_duplicates()
    # TODO. only considered the latest purchase here.
    return usr_item_df

def get_test_usr_item_pair(test_usr_raw_data):
    usr_item_df = test_usr_raw_data[['user_id','item_id']].drop_duplicates()
    ret = set()
    for idex,row in usr_item_df.iterrows():
        u_id = row['user_id']
        i_id = row['item_id']
        ret.add((u_id,i_id))
    return ret


def extract_test_raw_df(usr_item_df_dict_all, test_usr_item_pair_set):
    pass


def raw_process(usr_da, item_da):
    # 1. compute history stat before op:4(purchase)
    #   1.1 compute ft.

    usr_item_df_dict_all = groupby_ui_df_dict(usr_da)
    # 1.
    train_ft = compute_trained_ft(usr_item_df_dict_all)
    # 2.
    test_ft = compute_test_ft(usr_item_df_dict_all)

    train_ft_with_item_ft = concat_item(train_ft,item_da)
    test_ft_with_item_ft = concat_item(test_ft,item_da)

    return  train_ft_with_item_ft,test_ft_with_item_ft
    #test_usr_raw_data = compute_test_usr_item(usr_da)

    #test_usr_item_pair_set = get_test_usr_item_pair(test_usr_raw_data)

    #cleaned_trained_df = extract_train_raw_df(usr_item_df_dict_all)
    #cleaned_test_df = extract_test_raw_df(usr_item_df_dict_all,test_usr_item_pair_set)
    #visit_ft_df = compute_trained_ft(cleaned_trained_df)


    #print(usr_da.head())
    #ft = ft_gen(usr_da)

    return None, None
    # 2. compute all candidate for user A.
    #   2.1 each as a test
    #   2.2 each compute corresponding ft.

    # 3. concat the usr_da and item_da

    # 4. return train and test


def model_train_cv(model = 'dt'):
    # 1.
    # 2.
    pass


def main():
    #user_da = pd.read_csv(_ITEM_DATA)
    usr_da = pd.read_csv(_USER_TRAIN_DATA,parse_dates=['time']
                         ,date_parser=lambda date:pd.datetime.strptime(date,'%Y-%m-%d %H'))
    item_da = pd.read_csv(_ITEM_DATA)
    train_ft,test_ft = raw_process(usr_da,item_da)

    train_ft_arr = train_ft[].values
    train_ft_tar = train_ft[].values

    test_ft_arr = test_ft[].values
    test_ft_tar = test_ft[].values

    #
    lr = LogisticRegression(random_state=1)
    lr.fit(train_ft_arr,train_ft_tar)
    y_pred = lr.predict(test_ft_arr)
    recall = compute_recall(y_pred,test_ft_tar)
    precision = compute_precision(y_pred,test_ft_tar)




if __name__ == "__main__":
    main()