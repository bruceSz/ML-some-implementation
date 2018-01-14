# -*- coding=utf-8 -*-

from collections import OrderedDict
import Levenshtein

_F_PATH = "../data/can_cate3.csv"

_FOOD_PATH = "../../tag_crawler/data/tag_crawler_food.csv"
#food_name = "tag_crawler_food.csv"

def gen_cate_map():
    cates_arr = []
    cate_name2id = {}
    with open(_F_PATH, "r") as f:
        for line in f:
            tmp_l = line.split(",")
            tmp_l = [item.strip() for item in tmp_l]
            cate_name2id[tmp_l[1].decode('utf-8')] = long(tmp_l[0],10)
            cates_arr.append(tmp_l)
    return cate_name2id


def gen_food_map():
    f_name2id = {}
    with open(_FOOD_PATH,'r') as f:
        # skip title
        next(f)
        for line in f:
            tmp_l = line.split(",")
            tmp_l = [item.strip() for item in tmp_l]
            f_name2id[tmp_l[1].decode('utf-8')] = long(tmp_l[0],base=10)
    return f_name2id


def main():
    cols = ['cate3_id','cate3_name']
    can_map = gen_cate_map()
    food_map = gen_food_map()
    can_list = sorted(can_map.keys())
    food_2canlist = {}
    # food_map: food_name: food_id( kudu/impala table `ref`)
    for k,v in food_map.items():
        for i in range(len(can_list)):
            can_name = can_list[i]
            dis = Levenshtein.distance(k,can_name)

            if dis < len(k):
                print can_name, "dis is :", dis, "length of org val is :", len(k),"org:",k
                if k in food_2canlist:
                    food_2canlist[k].append((can_name,dis))
                else:
                    food_2canlist[k] = []
                    food_2canlist[k].append((can_name,dis))



    for k,can_list in food_2canlist.iteritems():
        print(k)
        for i in range(len(can_list)):
            print can_list[i][0],"#",
        break


if __name__ == "__main__":
    main()