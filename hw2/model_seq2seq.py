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
#config.gpu_options.allow_growth=True
#config.intra_op_parallelism_threads=1
#config.inter_op_parallelism_threads=2
#config.gpu_options.per_process_gpu_memory_fraction=0.333
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
import random
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
    # Pick 4th-longest sentences (not sure)
    labels = []
    for i in raw_labels:
        sents = i['caption']
        sents.sort(key=lambda x: len(x))
        labels.append(sents[-4].rstrip('.\n').lower())
    
    return labels

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    # Borrowed this from https://github.com/chenxinpeng/S2VT/blob/master/model_RGB.py
    # Original source: NeuralTalk
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
    # Borrowed this from https://github.com/chenxinpeng/S2VT/blob/master/model_RGB.py
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

def shuffle(X, Y):
    order = np.arange(len(X))
    np.random.shuffle(order)
    return (X[order], Y[order])

# General parameter settings
#data_dir_path='/nfs/nas-5.1/mhfu/MLDS_hw2_data/'
data_dir_path = sys.argv[1]
feat_dir_path=os.path.join(data_dir_path, 'training_data/feat/')
label_path=os.path.join(data_dir_path, 'training_label.json')

TRAIN = False

if TRAIN:
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
    with open('word_dict.pickle', 'wb') as f:
        pickle.dump([word2id, id2word, num_classes], f, protocol=pickle.HIGHEST_PROTOCOL)

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
    padded_sents = padded_sents[:,78:-1]
    
    from keras.layers import RepeatVector, Merge
    from keras.layers import Input, concatenate
    
    # Build seq2seq model
    batch_size = 32
    epochs = 10
    latent_dim = 256
    max_cap_len = padded_sents.shape[1]
    
    feat_inputs = Input(shape=(80,4096))
    lstm = LSTM(latent_dim, return_sequences=False)(feat_inputs)
    feat_output = RepeatVector(max_cap_len)(lstm)
    
    text_inputs = Input(shape=(max_cap_len, num_classes))
    masked_inputs = Masking(mask_value=0.)(text_inputs)
    lstm2 = LSTM(latent_dim, return_sequences=True)(masked_inputs)
    text_output = TimeDistributed(Dense(latent_dim))(lstm2)
    
    merge = concatenate([feat_output, text_output])
    x = LSTM(1000,return_sequences=False)(merge)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model([feat_inputs, text_inputs], [output])
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    
    def data_generator(feats, padded_sents, batch_size=32):
        partial_caps = []
        next_words = []
        feat_tile = []
        print("Generating data...")
        gen_count = 0
        nb_samples = feats.shape[0]
        max_cap_len = padded_sents.shape[1]
    
        total_count = 0
        while 1:
            feats, padded_sents = shuffle(feats, padded_sents)
            feat_counter = -1
            for text in padded_sents:
                feat_counter+=1
                current_feat = feats[feat_counter]
                
                #for i in range(max_cap_len-1):
                for i in range(list(text).index(0.)):
                    total_count+=1
                    partial = np.zeros(max_cap_len, dtype='int')
                    partial[:i+1] = text[:i+1]
                    partial_onehot = np_utils.to_categorical(partial, num_classes=num_classes)  # one-hot
                    partial_caps.append(partial_onehot)
                    next = np_utils.to_categorical(text[i+1], num_classes=num_classes)  # one-hot
                    next = np.squeeze(next)
                    next_words.append(next)
                    feat_tile.append(current_feat)
    
                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        feat_tile = np.asarray(feat_tile)
                        partial_caps = np.asarray(partial_caps)
                        total_count = 0
                        gen_count+=1
                        #print("yielding count: "+str(gen_count))
                        yield [[feat_tile, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        feat_tile = []
    
    model.fit_generator(data_generator(feats, padded_sents),
                        steps_per_epoch=(data_size*max_cap_len)//32,
                        epochs=10)
    
    model.save('model.h5')

else:
    # Load testing data
    test_feat_dir_path=os.path.join(data_dir_path, 'testing_data/feat/')
    test_label_path=os.path.join(data_dir_path, 'testing_label.json')
    test_idx, test_feats, test_labels = load_data(test_feat_dir_path, test_label_path)
    test_data_size = test_feats.shape[0]
    
    # Load word index dictionary
    dict_f = open('word_dict.pickle', 'rb')
    dict_data = pickle.load(dict_f)
    word2id = dict_data[0]
    id2word = dict_data[1]
    num_classes = dict_data[2]
    
    # Load seq2seq model
    model = load_model('./model.h5')
    max_cap_len = model.input_shape[1][1]
    
    cap_list = []
    for k in range(test_data_size):
        feat = test_feats[k:k+1]
        cap = np.zeros((max_cap_len),dtype='int')
        cap[0] = word2id['<bos>']
        cap_word_list = []
        for i in range(max_cap_len-1):
            cap_o = np_utils.to_categorical(cap, num_classes=num_classes)
            cap_o = cap_o.reshape(1,max_cap_len,num_classes)
            cap[i+1]=np.argmax(model.predict([feat,cap_o]))
            if cap[i+1] == word2id['<eos>']:
                break
            cap_word_list.append(id2word[cap[i+1]])
        sent = ' '.join(cap_word_list)
        cap_list.append(sent)
    
    id2cap = {}
    for pair in zip(test_idx, cap_list):
        #id2cap[pair[0]] = pair[1]
        arraged_cap = pair[1]
        id2cap[pair[0]] = arraged_cap
    
    test_id_path = os.path.join(data_dir_path, 'testing_id.txt')
    output_path = sys.argv[2]
    
    with open(test_id_path, 'r') as id_f, open(output_path, 'w') as f:
        for line in id_f:
            idx = line.strip('\n')
            f.write('%s,%s\n' % (idx, id2cap[idx].capitalize()))
    
    # Peer review
    peer_feat_dir_path=os.path.join(data_dir_path, 'peer_review/feat/')
    #peer_label_path=os.path.join(data_dir_path, 'testing_label.json')
    peer_idx, peer_feats, _ = load_data(peer_feat_dir_path, test_label_path)
    peer_data_size = peer_feats.shape[0]
    
    cap_list = []
    for k in range(peer_data_size):
        feat = peer_feats[k:k+1]
        cap = np.zeros((max_cap_len),dtype='int')
        cap[0] = word2id['<bos>']
        cap_word_list = []
        for i in range(max_cap_len-1):
            cap_o = np_utils.to_categorical(cap, num_classes=num_classes)
            cap_o = cap_o.reshape(1,max_cap_len,num_classes)
            cap[i+1]=np.argmax(model.predict([feat,cap_o]))
            if cap[i+1] == word2id['<eos>']:
                break
            cap_word_list.append(id2word[cap[i+1]])
        sent = ' '.join(cap_word_list)
        cap_list.append(sent)
        #print(sent)
    
    id2cap = {}
    for pair in zip(peer_idx, cap_list):
        #id2cap[pair[0]] = pair[1]
        arraged_cap = pair[1]
        id2cap[pair[0]] = arraged_cap
    
    peer_id_path = os.path.join(data_dir_path, 'peer_review_id.txt')
    output_path = sys.argv[3]
    
    with open(peer_id_path, 'r') as id_f, open(output_path, 'w') as f:
        for line in id_f:
            idx = line.strip('\n')
            f.write('%s,%s\n' % (idx, id2cap[idx].capitalize()))
    