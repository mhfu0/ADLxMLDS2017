#!/usr/bin/env python3

import numpy as np
import pandas as pd

import sys
import pickle
import itertools

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction=0.333
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=1

set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import SimpleRNN, LSTM
from keras.layers.wrappers import TimeDistributed

from keras import initializers
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam

from keras.models import load_model

# For reproducibilty
np.random.seed(7)
tf.set_random_seed(7)
import os, random
os.environ['PYTHONHASHSEED'] = '0'
random.seed(7)

# General parameter settings
data_path=sys.argv[1]
read_raw_data=False

# Create phone mapping
sys.stderr.write('Create phone mappings...\n')
mapping={}
with open(data_path+'phones/48_39.map','r') as m:
    for line in m:
        line=line.strip('\n').split('\t')
        mapping[line[0]]=line[1]

char_mapping={}
with open(data_path+'48phone_char.map','r') as m:
    for line in m:
        line=line.strip('\n').split('\t')
        char_mapping[line[0]]=line[2]

# Create index mappings
phone_index={}
index_phone=[]
with open('phone_index.map','r') as m:
    for line in m:
        line=line.strip('\n').split('\t')
        phone_index[line[0]]=line[1]
        index_phone.append(line[0])

# Read test data
sys.stderr.write('Load mfcc/test.ark...\n')
mfcc_df=pd.read_csv(data_path+'mfcc/test.ark',
                     delim_whitespace=True,
                     header=None,
                     index_col=0)

sys.stderr.write('Load fbank/test.ark...\n')
fbank_df=pd.read_csv(data_path+'fbank/test.ark',
                     delim_whitespace=True,
                     header=None,
                     index_col=0)

# list names of instances (sentences)
sent_name=mfcc_df.index[mfcc_df.index.str.endswith('_1')].values
sent_name=list(sent_name)
for i in range(len(sent_name)):
    sent_name[i] = sent_name[i][:-2]
    
x_test=[]
i=0
for name in sent_name:
    sys.stderr.write('Processing instance #%d...\n' % i)
    i+=1
    if i==5:
        break
    
    mfcc=mfcc_df.iloc[mfcc_df.index.str.startswith(name)]
    mfcc=mfcc.as_matrix().astype(np.float32)
    
    fbank=fbank_df.iloc[fbank_df.index.str.startswith(name)]
    fbank=fbank.as_matrix().astype(np.float32)
    
    # Concatenate two features directly
    sent=np.concatenate((mfcc,fbank), axis=1)
    sys.stderr.write('shape=(%d,%d)...\n' % (sent.shape[0],sent.shape[1]))
    
    x_test.append(sent)

num_sent=len(x_test)

# Free DataFrame memory
del mfcc_df
del fbank_df

# Model parameter settings
frame_size=400      # padding size
batch_size =16

data_dim=69         # take only fbank feature into consideration
dummy_class=39
num_classes=39+1    # +1 for dummy class

# Pad 0 / Split x_test into timesteps=400
sys.stderr.write('Processing x_test...\n')
num_split=[]        # number of splits for each sentence (len=num_sent)
x_test_split=[]     # x_test after splitting; should have length 688 here
for i in range(num_sent):
    # Split data if len(x_test[i]) > frame_size
    num_split.append(len(x_test[i])//frame_size+1)
    
    for k in range(num_split[i]):
        split = x_test[i][frame_size*k:frame_size*(k+1)]
        print(split.shape)
        print(num_split)
        padding=np.zeros((frame_size,data_dim))
        index=-min(frame_size,split.shape[0])
        
        padding[index:] = (split[i][index:])[:,-data_dim:]
        # take only fbank feature
        #x_test[i] = padding.copy()

x_test_split=np.array(x_test_split)

'''
# Pad the sequence by zero to proceed sliding window
sys.stderr.write('Paddind zeros...\n')
x_test_r=[]
for i in range(len(x_test)):
    x_test[i] = x_test[i].astype(np.float32)
    for p in range(padding_size):
        x_test[i] = np.insert(x_test[i],0,0.0,axis=0)
        #x_test[i] = np.insert(x_test[i],x_test[i].shape[0],0.0,axis=0)
        
    # Reshape data by sliding window (shape=(timesteps,data_dim))
    # Expect x_test.shape=(num_data, timesteps, data_dim)
    
    r=[]  # reshaped x_train
    sent = x_test[i]
    #num_windows=sent.shape[0]-2*padding_size
    num_windows=sent.shape[0]-padding_size
    for i in range(num_windows):
        window=sent[i:i+timesteps,:]
        r.append(window)
        
    x_test_r.append(np.array(r))
'''

print(x_test_split.shape)
print(x_test_split[0:5])

# Load trained model
sys.stderr.write('Load trained model...\n')
model = load_model(sys.argv[2])
#model.summary()

# Prdiction
sys.stderr.write('Predicting...\n' )
print('id,phone_sequence')
y_test_split = model.predict_classes(x_test_split,batch_size=batch_size,verbose=0)


k=0
y_test=[]
for i in range(len(num_split)):
    tmp = []
    for j in range(num_split[i]):
        y = y_test.tolist[k]
        tmp = tmp + y_test_split
        k += 1
    y_test.append(tmp)

print(y_test)

for i in range(len(y_test)):
    seq = ''
    for l in y:
        try:
            seq += char_mapping[index_phone[l]]
        except:
            seq += ''
    # Trimming
    seq = ''.join(i for i, _ in itertools.groupby(seq))
    seq = seq.strip(char_mapping['sil'])
    
    print(sent_name[i]+','+seq)
'''
for i, x_test in enumerate(x_test_r):
    
    res = model.predict_classes(x_test,verbose=0)
    res = res.tolist()
    seq = ''
    for l in res:
        seq += char_mapping[index_phone[l]]
    
    # Trimming
    seq = ''.join(i for i, _ in itertools.groupby(seq))
    seq = seq.strip(char_mapping['sil'])
    
    print(sent_name[i]+','+seq)
'''