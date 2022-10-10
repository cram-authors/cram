# From https://github.com/papers-submission/structured_transposable_masks


# Generates a calibration set of 1000 ImageNet training samples (1 per class) 
# To be used for Batch Norm tuning, e.g. after one-shot pruning

import os
import shutil

basepath = 'PATH/TO/IMAGENET/TRAINSET'
basepath_calib = 'PATH/TO/CALIBRATION/SET'

directory = os.fsencode(basepath)
os.mkdir(basepath_calib)
for d in os.listdir(directory):
    dir_name = os.fsdecode(d)
    dir_path = os.path.join(basepath,dir_name)
    dir_copy_path = os.path.join(basepath_calib,dir_name)
    os.mkdir(dir_copy_path)
    for i, f in enumerate(os.listdir(dir_path)):
        if i==1:
            break
        file_path = os.path.join(dir_path,f)
        copy_file_path = os.path.join(dir_copy_path,f)
        shutil.copyfile(file_path, copy_file_path)
