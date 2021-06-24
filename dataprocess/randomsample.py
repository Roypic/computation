import shutil
import os
import random
from os.path import join as pathjoin
# random sample with ratio from dataset

resourcedir="D:\\Gitrepository\\file_for_test\\images"
labeldir="D:\\Gitrepository\\file_for_test\\masks"
resourcedirnew="D:\\Gitrepository\\file_for_test\\imagesnew"
labeldirnew="D:\\Gitrepository\\file_for_test\\masksnew"


def moveFile(resourcedir,labeldir,resourcedirnew,labeldirnew):
    if not (os.path.exists(resourcedirnew)):
        os.makedirs(resourcedirnew)
    if not (os.path.exists(labeldirnew)):
        os.makedirs(labeldirnew)
    pathDir = os.listdir(resourcedir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        if(os.path.exists(pathjoin(resourcedir , name)) and os.path.exists(pathjoin(labeldir , name))):
            shutil.move(pathjoin(resourcedir ,name), pathjoin(resourcedirnew , name))
            shutil.move(pathjoin(labeldir , name), pathjoin(labeldirnew , name))

if __name__ == "__main__":
    moveFile(resourcedir, labeldir, resourcedirnew, labeldirnew)