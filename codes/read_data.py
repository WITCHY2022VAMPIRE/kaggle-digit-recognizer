import numpy as np
#import pandas as pd

def read():
    features_train= []
    labels_train= []
    features_test= []
    labels_test= []
    
    print('### Reading training data file...')
    with open('../data/train.csv') as IFILE_TRAIN:
        for i, line in enumerate(IFILE_TRAIN):
            if i==0: continue
            data= line.strip().split(',')
            features_train.append( [ int(d) for d in data[1:] ] )
            labels_train.append( int(data[0]) )
    print("Reading complete! Totally %d data" % len(labels_train))
    
    print('### Reading test data file...')
    with open('../data/test.csv') as IFILE_TEST:
        for i, line in enumerate(IFILE_TEST):
            if i==0: continue
            data= line.strip().split(',')
            features_test.append( [ int(d) for d in data ] )
            #labels_test.append( int(data[0]) )
    print("Reading complete! Totally %d data" % len(features_test))
    
    return np.array(features_train), np.array(labels_train), np.array(features_test)
