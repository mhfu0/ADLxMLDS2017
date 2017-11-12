# coding: utf-8
import numpy as np
import pandas as pd

import sys, os
import collections
import json
import pickle

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction=0.333
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=2
tf.set_random_seed(7)

set_session(tf.Session(config=config))

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation
from keras.layers import Dropout, Reshape, Masking
from keras.layers import SimpleRNN, LSTM, GRU
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import regularizers, initializers
from keras.optimizers import RMSprop, Adam

# For reproducibilty
np.random.seed(7)
import os, random
os.environ['PYTHONHASHSEED'] = '0'
random.seed(7)

def load_data(feat_dir_path, label_path):
  # Read training features
  feats = []
  for feat_name in os.listdir(feat_dir_path):
    if feat_name.endswith('.npy'):
      feat_data = np.load(feat_dir_path+feat_name)
      feats.append([feat_name,feat_data])
  feats.sort(key=lambda x: x[0])
  idx = [f[0][:-4] for f in feats]
  feats = np.array([f[1] for f in feats])
  
  # Read training labels
  label_file = open(label_path, 'r')
  labels = json.load(label_file)
  label_file.close()
  labels.sort(key=lambda x:x['id'])

  return idx, feats, labels

def extract_label(raw_labels):
  # Pick 3rd-long sentences (not sure)
  labels = []
  for i in raw_labels:
    sents = i['caption']
    sents.sort(key=lambda x: len(x))
    labels.append(sents[-3].rstrip('.\n').lower())
  
  return labels

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
  # borrowed this function from NeuralTalk
  sys.stderr.write('preprocessing word counts and creating vocab based on word count threshold %d\n' % (word_count_threshold))
  word_counts = collections.OrderedDict()
  nsents = 0
  for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
         word_counts[w] = word_counts.get(w, 0) + 1
  vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and len(w)>0]
  sys.stderr.write('filtered words from %d to %d' % (len(word_counts), len(vocab)))
  vocab_size = len(vocab) + 4

  ixtoword = {}
  ixtoword[0] = '<pad>'
  ixtoword[1] = '<bos>'
  ixtoword[2] = '<eos>'
  ixtoword[3] = '<unk>'

  wordtoix = {}
  wordtoix['<pad>'] = 0
  wordtoix['<bos>'] = 1
  wordtoix['<eos>'] = 2
  wordtoix['<unk>'] = 3

  for idx, w in enumerate(vocab):
      wordtoix[w] = idx+4
      ixtoword[idx+4] = w

  word_counts['<pad>'] = nsents
  word_counts['<bos>'] = nsents
  word_counts['<eos>'] = nsents
  word_counts['<unk>'] = nsents

  bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
  bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
  bias_init_vector = np.log(bias_init_vector)
  bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

  return wordtoix, ixtoword, vocab_size, bias_init_vector

def remove_special_chara(labels):
  labels = map(lambda x: x.replace('.', ''), labels)
  labels = map(lambda x: x.replace(',', ''), labels)
  labels = map(lambda x: x.replace('"', ''), labels)
  labels = map(lambda x: x.replace('\n', ''), labels)
  labels = map(lambda x: x.replace('?', ''), labels)
  labels = map(lambda x: x.replace('!', ''), labels)
  labels = map(lambda x: x.replace('\\', ''), labels)
  labels = map(lambda x: x.replace('/', ''), labels)
  labels = map(lambda x: x.replace('-', ''), labels)
  labels = map(lambda x: x.replace('\(', ''), labels)
  labels = map(lambda x: x.replace('\)', ''), labels)  
  
  return list(labels)

# General parameter settings
#data_dir_path='/nfs/nas-5.1/mhfu/MLDS_hw2_data/'
data_dir_path=sys.argv[1]
feat_dir_path=data_dir_path+'training_data/feat/'
label_path=data_dir_path+'training_label.json'

# Load training data
# idx: names of training instances, len=1450
# feat: features of data, (1450, 80, 4096)
# labels: json style label data
idx, feats, labels = load_data(feat_dir_path, label_path)
data_size = feats.shape[0]  # size of data
feat_len = feats.shape[1]  # timesteps
feat_dim = feats.shape[2]  # data dimension

# Pick one sentence for each instance
labels = extract_label(labels)
labels = remove_special_chara(labels)

# Create word-index mapping
word2id, id2word, num_classes, _ = preProBuildWordVocab(labels, word_count_threshold=1)

# Pad sentences to fixed length
labels = [s.split(' ') for s in labels]
max_sent_len = max([len(s) for s in labels])

padded_sents=[]
#pad_len = feat_len+max_sent_len+1
pad_len = (feat_len+max_sent_len+1)+1  # for encoder-decoder model
for raw_label in labels:
  sent = np.zeros(pad_len, dtype=np.int)
  sent[(feat_len-1)-1] = word2id['<bos>']
  for i in range(len(raw_label)):
    if raw_label[i] in word2id:
      sent[(feat_len-1)+i] = word2id[raw_label[i]]
    else:
      sent[(feat_len-1)+i] = word2id['<unk>']
  sent[feat_len+i] = word2id['<eos>']
  padded_sents.append(sent)

padded_sents = np.asarray(padded_sents, dtype=np.int)

# Pad video features
padded_feats = np.zeros((data_size,pad_len,feat_dim))
padded_feats[:,0:feat_len,:] = feats[:,0:feat_len,:]
x_train = padded_feats

# Change labels into one-hot encoding
y_train = []
for i in range(len(padded_sents)):
  tmp = np_utils.to_categorical(padded_sents[i], num_classes=num_classes)
  y_train.append(tmp)
y_train = np.asarray(y_train, dtype=np.int)

# Data modification for encoder-decoder model
encoder_input_data = x_train[:,:80,:]
decoder_input_data = y_train[:,78:-1,:]
decoder_target_data = y_train[:,79:,:]
max_decoder_seq_length = decoder_target_data.shape[1]

# Build seq2seq model
# Borrowing code from https://github.com/fchollet/keras/blob/master/examples/lstm_seq2seq.py
batch_size = 16
epochs = 15
latent_dim = 1024

inputs = Input(shape=(None, feat_dim))
lstm1 = LSTM(latent_dim, return_sequences=True, activation='relu')(inputs)
encoder_inputs = Dropout(0.5)(lstm1)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_classes))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_classes, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1)

from keras import backend as K

# Define sampling models
encoder_model = Model(inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

from collections import Counter

def decode_sequence(input_seq):
  # Encode the input as state vectors.
  states_value = encoder_model.predict(input_seq)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1, num_classes))
  target_seq[0, 0, word2id['<bos>']] = 1.

  # Sampling loop for each sequences
  stop_condition = False
  decoded_sentence = []
  while not stop_condition:
    output_tokens, h, c = decoder_model.predict(
        [target_seq] + states_value)

    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    np.random.seed(7)
    '''
    sampled_token_index_list = np.random.choice(num_classes,3,p=output_tokens[0, -1, :])
    counter = Counter(sampled_token_index_list)
    sampled_token_index = counter.most_common(1)[0][0]
    '''
    sampled_token = id2word[sampled_token_index]
    decoded_sentence.append(sampled_token)

    if (sampled_token_index == word2id['<eos>'] or
       len(decoded_sentence) > max_decoder_seq_length):
        stop_condition = True

    target_seq = np.zeros((1, 1, num_classes))
    target_seq[0, 0, sampled_token_index] = 1.

    states_value = [h, c]

  return decoded_sentence

test_feat_dir_path=data_dir_path+'testing_data/feat/'
test_label_path=data_dir_path+'testing_label.json'

test_idx, test_feats, test_labels = load_data(test_feat_dir_path, test_label_path)
test_data_size = test_feats.shape[0]  # size of data

predictions=[]
for seq_index in range(test_data_size):
  # Take one sequence (part of the training test)
  # for trying out decoding.
  input_seq = test_feats[seq_index: seq_index + 1]
  decoded_sentence = decode_sequence(input_seq)
  predictions.append(decoded_sentence)

trimmed_pred=[]
for p in predictions:
  p = list(map(lambda x: '' if x=='<eos>' else x, p))
  p = list(map(lambda x: '' if x=='<pad>' else x, p))
  p = list(map(lambda x: '' if x=='<unk>' else x, p))
  p = list(map(lambda x: '' if x=='<bos>' else x, p))
  s = ' '.join(p).strip(' ') + '.'
  s = s.capitalize()
  trimmed_pred.append(s)

output_path=sys.argv[2]

id_dict={}
for pair in zip(test_idx, trimmed_pred):
  id_dict[pair[0]] = pair[1]

sp_idx=['klteYv1Uv9A_27_33.avi','5YJaS2Eswg0_22_26.avi','UbmZAe5u5FI_132_141.avi','JntMAcTlOF0_50_70.avi','tJHUH9tpqPg_113_118.avi']
'''
with open(output_path, 'w') as out, open(data_dir_path+'testing_id.txt','r') as f:
  for line in f:
    idx = line.strip('\n')
    out.write('%s,%s\n' % (idx,id_dict[idx]))
'''
with open(output_path, 'w') as out:
  for idx in sp_idx:
    out.write('%s,%s\n' % (idx,id_dict[idx]))