import os
import shutil
import random
oripath='./'
def namefilter(prefix,name,last): #need to complete
    name=prefix+name+last
    return name

def rename_move_file(src_path, dst_path, picfile,newname_filter):
    print('from : ',src_path)
    print('to : ',dst_path)
    try:
        # cmd = 'chmod -R +x ' + src_path
        # os.popen(cmd)
        f_src = os.path.join(src_path, picfile)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        f_dst = os.path.join(dst_path, picfile)
        shutil.copyfile(f_src, f_dst)
        os.rename(f_dst,os.path.join(dst_path, namefilter(picfile)))
    except Exception as e:
        print('move_picfile ERROR: ',e)


def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2



