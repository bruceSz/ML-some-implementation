
import copy

def combine(src_arr,dst_arr,t_len):
    if len(dst_arr) == t_len :
        print(dst_arr)
        return
    if (len(src_arr) < t_len - len(dst_arr)):
        return
    item = src_arr[0]
    dst_arr.append(item)
    combine(src_arr[1:len(src_arr)],copy.copy(dst_arr),t_len)
    dst_arr.remove(item)
    combine(src_arr[1:len(src_arr)],copy.copy(dst_arr),t_len)


def main():
    src_arr = [i for i in range(5)]
    combine(src_arr,[],4)


if __name__ == "__main__":
    main()