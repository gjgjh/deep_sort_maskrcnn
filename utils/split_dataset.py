import os
import glob
from random import shuffle
from shutil import copyfile

dir = '/Users/huiguo/Code/ros_workspace/data/omd/*.json'
train_dir = '/Users/huiguo/Code/ros_workspace/data/omd_train'
val_dir = '/Users/huiguo/Code/ros_workspace/data/omd_val'
split = 0.8

filepaths = glob.glob(dir)
shuffle(filepaths)
train_num = split * len(filepaths)

count = 0
for filepath in filepaths:
    _, filename = os.path.split(filepath)
    if (count < train_num):
        dst = os.path.join(train_dir, filename)
    else:
        dst = os.path.join(val_dir, filename)
    copyfile(filepath, dst)

    count = count + 1

print('Done!')
