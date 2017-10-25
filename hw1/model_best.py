#!/usr/bin/env python3

import numpy as np
import pandas as pd

import sys
import pickle

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction=0.333
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=2

set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout, Reshape, Masking
from keras.layers import SimpleRNN, LSTM, GRU
from keras.layers import Conv1D
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization

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
mapping={}  # 48 -> 39 label mapping
with open(data_path+'phones/48_39.map','r') as m:
    for line in m:
        line=line.strip('\n').split('\t')
        mapping[line[0]]=line[1]

char_mapping={}  # label -> char mapping
with open(data_path+'48phone_char.map','r') as m:
    for line in m:
        line=line.strip('\n').split('\t')
        char_mapping[line[0]]=line[2]

# Create index mappings
phone_index={}  # label -> 48
index_phone=[]  # 48 -> label
'''
# using 39 phones
with open('phone_index.map','r') as m:
    for line in m:
        line=line.strip('\n').split('\t')
        phone_index[line[0]]=line[1]
        index_phone.append(line[0])
'''
# using 48 phones
with open(data_path+'48phone_char.map','r') as m:
    for line in m:
        line=line.strip('\n').split('\t')
        phone_index[line[0]]=line[1]
        index_phone.append(line[0])

# Read raw data if needed
if read_raw_data:
    # Should have smarter approach here?
    sys.stderr.write('Load mfcc/train.ark...\n')
    mfcc_df=pd.read_csv(data_path+'mfcc/train.ark',
                         delim_whitespace=True,
                         header=None,
                         index_col=0)
    
    sys.stderr.write('Load fbank/train.ark...\n')
    fbank_df=pd.read_csv(data_path+'fbank/train.ark',
                         delim_whitespace=True,
                         header=None,
                         index_col=0)
    
    sys.stderr.write('Load label/train.lab...\n')
    label_df=pd.read_csv(data_path+'label/train.lab',
                         header=None,
                         index_col=0)
    #label_df=label_df.replace(mapping)
    
    # pd.concat will align data automatically yet losing original order
    #mfcc_df=pd.concat([label_df,mfcc_df],axis=1,index=None)
    
    # list names of instances (sentences)
    sent_name=mfcc_df.index[mfcc_df.index.str.endswith('_1')].values
    sent_name=list(sent_name)
    for i in range(len(sent_name)):
        sent_name[i] = sent_name[i][:-2]
        
    sys.stderr.write('%d sentences to process\n' % len(sent_name))
    
    # Preparing x_train, y_train
    # len(x_train)==len(y_train)== #sentences
    # x_train[0].shape would be (#frames, #dims==39+69)
    # y_train[0].shape would be (#frames,)
    
    x_train=[]
    y_train=[]
    i=0
    for name in sent_name:
        sys.stderr.write('Processing instance #%d...\n' % i)
        i+=1
        
        mfcc=mfcc_df.iloc[mfcc_df.index.str.startswith(name)]
        mfcc=mfcc.as_matrix().astype(np.float32)
        
        fbank=fbank_df.iloc[fbank_df.index.str.startswith(name)]
        fbank=fbank.as_matrix().astype(np.float32)
        
        # Concatenate two features directly
        sent=np.concatenate((mfcc,fbank), axis=1)
        sys.stderr.write('shape=(%d,%d)...\n' % (sent.shape[0],sent.shape[1]))
        
        x_train.append(sent)
        
        label=label_df.iloc[label_df.index.str.startswith(name)]
        label=label.as_matrix()
        y_train.append(label)
        
    # Free DataFrame memory
    del mfcc_df
    del label_df
    del fbank_df
    
    # Save data with pickle
    sys.stderr.write('Saving data...\n')
    with open(data_path+'x_train.pickle', 'wb') as x_f,\
         open(data_path+'y_train.pickle', 'wb') as y_f:
        pickle.dump(x_train, x_f)
        pickle.dump(y_train, y_f)

else:
    # Load data with pickle
    sys.stderr.write('Loading data...\n')
    with open(data_path+'x_train.pickle', 'rb') as x_f,\
         open(data_path+'y_train.pickle', 'rb') as y_f:
        x_train = pickle.load(x_f)
        y_train = pickle.load(y_f)

# len(x_train)==len(y_train)==num_sentences
if len(x_train)!=len(y_train):
    raise Exception
num_sent = len(x_train)

# x_train[0].shape would be (num_frames, data_dim==39+69)
# y_train[0].shape would be (num_frames,)

# Model parameter settings
frame_size=400    # padding size
epochs=2000

data_dim=108        # take both features into consideration
dummy_class=48
num_classes=48+1    # +1 for dummy class

# CNN settings
num_filters=512
kernel_size=15
kernel_size2=7

# Pad 0. / Truncate x_train into timesteps=400
sys.stderr.write('Processing x_train...\n')
for i in range(num_sent):
    padding=np.zeros((frame_size,data_dim))
    index=-min(frame_size,x_train[i].shape[0])
    
    padding[index:] = (x_train[i][index:])[:,-data_dim:]
    # take only fbank feature
    x_train[i] = padding.copy()

# x_train.shape should be (3696,400,69)
x_train=np.array(x_train)
#print(x_train.shape)

#print(x_train[0], x_train[1])
#print(len(x_train), len(y_train))

# Pad / Truncate y_train to confirm dim.
# use 48th as the index of "dummy class"
sys.stderr.write('Processing y_train...\n')
for i in range(num_sent):
    sent = y_train[i].flatten().tolist()
    labels = [phone_index[l] for l in sent]
    labels = np.array(labels)
    
    padding = np.full(frame_size,dummy_class)
    index=-min(frame_size,len(sent))
    padding[index:] = labels[index:]
    y_train[i] = np_utils.to_categorical(padding, num_classes=num_classes)

y_train=np.array(y_train)

# y_train should be list of (400,49) with len=3692
#print(y_train[0].tolist(), y_train[1].tolist())
#print(y_train.shape)
#print(len(y_train))

# Build models
sys.stderr.write('Building NN model...\n')

#optimizer = RMSprop(clipnorm=1.)
optimizer = Adam(clipnorm=1.)
batch_size = 64

model = Sequential()
model.add(TimeDistributed(Dense(512,activation='relu'),
          input_shape=(frame_size, data_dim)))
model.add(Dropout(0.3))

model.add(Conv1D(num_filters,
                 kernel_size,
                 padding='same',
                 activation='relu',
                 strides=1))
model.add(BatchNormalization())
model.add(Dense(num_filters, activation='relu'))
model.add(Dropout(0.3))

model.add(Conv1D(num_filters,
                 kernel_size2,
                 padding='same',
                 activation='relu',
                 strides=1))
model.add(BatchNormalization())
model.add(Dense(num_filters, activation='linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.3))

model.add(Bidirectional(GRU(256,
          return_sequences=True,
          activation='relu')))
model.add(Dense(256, activation='linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

try:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1)
except:
    model.save(sys.argv[2])
              
model.save(sys.argv[2])
'''
# Pad the sequence by zero to proceed sliding window
sys.stderr.write('Paddind zeros...\n')
for i in range(len(x_train)):
    x_train[i] = x_train[i].astype(np.float32)
    for p in range(padding_size):
        x_train[i] = np.insert(x_train[i],0,0.0,axis=0)
        #x_train[i] = np.insert(x_train[i],x_train[i].shape[0],0.0,axis=0)

# Reshape data by sliding window (shape=(timesteps,data_dim))
# Expect x_train_r.shape=(num_data, timesteps, data_dim)
sys.stderr.write('Reshaping x_train...\n')
x_train_r=[]  # reshaped x_train
for sent in x_train:
    #num_windows=sent.shape[0]-2*padding_size
    num_windows=sent.shape[0]-padding_size
    for i in range(num_windows):
        window=sent[i:i+timesteps,:]
        x_train_r.append(window)
    
x_train_r=np.array(x_train_r)

# Expect y_train_r.shape=(num_data, num_classes)
# Map phone labels to int (Should have done in DataFrame structure...)
sys.stderr.write('Reshaping y_train...\n')
y_train_r=[]
for sent in y_train:
    sent = sent.flatten().tolist()
    for i in range(len(sent)):
        y_train_r.append(phone_index[mapping[sent[i]]])

y_train_r=np.array(y_train_r)
y_train_r=np_utils.to_categorical(y_train_r)

print(x_train_r.shape, y_train_r.shape)
print(x_train_r[:10,:10,:].tolist())
print(y_train_r[:10,:].tolist())

# Build LSTM models
sys.stderr.write('Building LSTM model...\n')
model = Sequential()
model.add(SimpleRNN(512, return_sequences=True,
          input_shape=(timesteps, data_dim)))
model.add(Dropout(0.2))
model.add(SimpleRNN(256, return_sequences=True)) 
model.add(GRU(128))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

model.fit(x_train_r[:200000,:,:], y_train_r[:200000,:],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.5)
          
model.save(sys.argv[2])

loss, accuracy = model.evaluate(x_train_r[900000:,:,:],y_train_r[900000:,:],
                            batch_size=batch_size)                     
sys.stderr.write('End training with loss=%f and accuracy=%f\n' % (loss,accuracy))
'''
