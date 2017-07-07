import numpy as np
#import pandas as pd

def read():
    x_size = 784
    
    print('### Reading training data file...')

    feature_train= []
    label_train= []
    feature_test= []
    label_test= []
    with open('../data/train.csv') as IFILE_TRAIN:
        for i, line in enumerate(IFILE_TRAIN):
            s= line.strip().split(',')
            feature_train.append(s[1:])
            label_train.append(s[0])
    with open('../data/test.csv') as IFILE_TEST:
        for i, line in enumerate(IFILE_TEST):
            s= line.strip().split(',')
            feature_test.append(s[1:])
            label_test.append(s[0])
    return feature_train, label_train, feature_test, label_test 
