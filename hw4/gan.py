import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib as tc

#import matplotlib.pyplot as plt

import skimage
import skimage.io, skimage.transform

import os, sys
import pickle
import itertools
import collections

import scipy.misc

# Create feature indices mapping
hair_colors = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 
               'green hair', 'red hair', 'purple hair', 'pink hair',
               'blue hair', 'black hair', 'brown hair', 'blonde hair']
eye_colors = ['gray eyes', 'black eyes', 'orange eyes', 'pink eyes',
              'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes', 
              'brown eyes', 'red eyes', 'blue eyes', 'bicolored eyes']

index2hair = dict(list(enumerate(hair_colors)))
index2eye = dict(list(enumerate(eye_colors)))

hair2index = dict(list(zip(index2hair.values(), index2hair.keys())))
eye2index = dict(list(zip(index2eye.values(), index2eye.keys())))

########## Load tag data ########## 
print('Load tag data')

file = open('tags_clean.csv', 'r')
text_labels = []

MANUAL_TAGGING = False

for line in file:
    line = line.rstrip('\n\t').split(',')
    img_id = line[0]

    line = line[1]
    raw_tags = line.split('\t')
    raw_tags = list(map(lambda x: x.split(':')[0], raw_tags)) 

    pair = [None, None] 
    for t in raw_tags:
        if t in hair_colors:
            pair[0] = t
        if t in eye_colors: 
            pair[1] = t

    '''
    MANUAL_TAGGING = True
    if not pair[0] or not pair[1]:
        tmp_img = skimage.io.imread('faces/%s.jpg' % img_id)
        
        plt.imshow(tmp_img)
        plt.show()
        
        if not pair[0]:
            pair[0] = str(input()) + ' hair'
        if not pair[1]:
            pair[1] = str(input()) + ' eyes'
    
    print('faces/%s.jpg' % img_id)
    print(pair)
    
    if int(img_id) > 0 and int(img_id) % 10 == 0:
        clear_output()
    '''
    text_labels.append(pair)
    
file.close()
with open('text_labels.pkl', 'wb') as p:
    pickle.dump(text_labels, p)

##########  End loading tag data ##########
print('End loading %d tag data' % len(text_labels))

idx_label_tuples = []
if not MANUAL_TAGGING:
    for idx, label in enumerate(text_labels):
        if label[0] in hair_colors and label[1] in eye_colors:
            idx_label_tuples.append((idx, label))
        else:
            pass

print('Pick %d data' % len(idx_label_tuples))

########## Load image data ########## 
print('Load image data')
try:
    with open('img_data.pkl', 'rb') as f:
        img_data = pickle.load(f)
    with open('label_data.pkl', 'rb') as f:
        label_data = pickle.load(f)
        
except:
    print('Load raw image data')
    idx_list = [tup[0] for tup in idx_label_tuples]

    img_list = []
    label_list = []

    for tup in idx_label_tuples:
        # Read image
        idx = tup[0]
        img = skimage.io.imread('faces/%d.jpg' % idx)
        img_r = skimage.transform.resize(img, (64,64))

        # encode labels
        hair_idx = hair2index[tup[1][0]]
        hair_one_hot = np.zeros(len(hair2index))
        hair_one_hot[hair_idx] = 1.

        eye_idx = eye2index[tup[1][1]]
        eye_one_hot = np.zeros(len(eye2index))
        eye_one_hot[eye_idx] = 1.

        label = np.array(list(hair_one_hot) + list(eye_one_hot))
        img_list.append(img_r)
        label_list.append(label)

    img_data = np.array(img_list)
    label_data = np.array(label_list)

    with open('img_data.pkl', 'wb') as f:
        pickle.dump(img_data, f)
    with open('label_data.pkl', 'wb') as f:
        pickle.dump(label_data, f)

print(img_data.shape, label_data.shape)
print('End loading %d image data' % len(img_data))

########## End loading image data ##########

def sample_batch(img_data, label_data, batch_size=64):
    assert(len(img_data)==len(label_data))
    
    order = np.random.choice(np.arange(len(img_data)), batch_size)
    return img_data[order], label_data[order]

def shuffle(X, Y):
    order = np.arange(len(X))
    np.random.shuffle(order)
    return (X[order], Y[order])

########## NN Settings ##########

BATCH_SIZE = 64
EPOCHS = 100

def LeakyReLU(input, leak=0.2, name='lrelu'):
    return tf.maximum(input, leak * input)
    
def Conv2d(input, output_dim=64, kernel=5, strides=2, stddev=0.2, name='conv_2d'):

    with tf.variable_scope(name):
        W = tf.get_variable('Conv2dW', [kernel, kernel, input.get_shape()[-1], output_dim],
                           #initializer=tf.truncated_normal_initializer(stddev=stddev))
                           initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('Conv2db', [output_dim], initializer=tf.zeros_initializer())
        
        return tf.nn.conv2d(input, W, strides=[1, strides, strides, 1], padding='SAME') + b
    
def Deconv2d(input, output_dim, batch_size, kernel=5, strides=2, stddev=0.2, name='deconv_2d'):
    
    with tf.variable_scope(name):
        W = tf.get_variable('Deconv2dW', [kernel, kernel, output_dim, input.get_shape()[-1]],
                           #initializer=tf.truncated_normal_initializer(stddev=stddev))
                           initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('Deconv2db', [output_dim], initializer=tf.zeros_initializer())

        input_shape = input.get_shape().as_list()
        output_shape = [batch_size, int(input_shape[1] * strides),
                        int(input_shape[2] * strides), output_dim]
        deconv = tf.nn.conv2d_transpose(input, W, output_shape=output_shape,
                                        strides=[1, strides, strides, 1])
        return deconv + b

def Dense(input, output_dim, stddev=0.02, name='dense'):
    
    with tf.variable_scope(name):
        W = tf.get_variable('DenseW', [input.get_shape()[1], output_dim],
                           initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('Denseb', [output_dim], initializer=tf.zeros_initializer())
        return tf.matmul(input, W) + b

def BatchNormalization(input, name='bn'):
    
    with tf.variable_scope(name):
    
        output_dim = input.get_shape()[-1]
        beta = tf.get_variable('BnBeta', [output_dim], initializer=tf.zeros_initializer())
        gamma = tf.get_variable('BnGamma', [output_dim], initializer=tf.ones_initializer())
    
        if len(input.get_shape()) == 2:
            mean, var = tf.nn.moments(input, [0])
        else:
            mean, var = tf.nn.moments(input, [0, 1, 2])
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-5)


def Discriminator(X, reuse=True, name='D_net'):

    with tf.variable_scope(name, reuse=reuse):
        
        batch_size = X.get_shape().as_list()[0]
        
        if len(X.get_shape()) > 2:
            D_conv1 = Conv2d(X, output_dim=64, name='D_conv1')
        else:
            D_reshaped = tf.reshape(X, [-1, 64, 64, 3])
            D_conv1 = Conv2d(D_reshaped, output_dim=64, name='D_conv1')
        D_h1 = LeakyReLU(D_conv1) 
        
        D_conv2 = Conv2d(D_h1, output_dim=128, name='D_conv2')
        D_h2 = LeakyReLU(D_conv2) 
        D_conv3 = Conv2d(D_h2, output_dim=128, name='D_conv3')
        D_h3 = LeakyReLU(D_conv3) 
        D_r2 = tc.layers.flatten(D_h3)
        
        D_h3 = LeakyReLU(D_r2) 
        D_h4 = tf.nn.dropout(D_h3, 0.1)
        D_h5 = Dense(D_h4, output_dim=1, name='D_h5')
        
        return tf.nn.sigmoid(D_h5)

        
def Generator(z, reuse=False, name='G_net'):

    with tf.variable_scope(name, reuse=reuse):
        
        batch_size = z.get_shape().as_list()[0]
        
        G_1 = Dense(z, output_dim=1024, name='G_1')
        G_bn1 = BatchNormalization(G_1, name='G_bn1')
        G_h1 = tf.nn.relu(G_bn1)
        
        G_2 = Dense(G_h1, output_dim=4*4*256, name='G_2')
        G_bn2 = BatchNormalization(G_2, name='G_bn2')        
        G_h2 = tf.nn.relu(G_bn2)
        G_r2 = tf.reshape(G_h2, [-1, 4, 4, 256])
        
        G_conv3 = Deconv2d(G_r2, output_dim=64, batch_size=BATCH_SIZE, name='G_conv3')
        G_bn3 = BatchNormalization(G_conv3, name='G_bn3')    
        G_h3 = tf.nn.relu(G_bn3)
        print(G_h3.get_shape())  # [8, 8]
        
        G_conv4 = Deconv2d(G_h3, output_dim=128, batch_size=BATCH_SIZE, name='G_conv4')
        G_bn4 = BatchNormalization(G_conv4, name='G_bn4')    
        G_h4 = tf.nn.relu(G_bn4)
        print(G_h4.get_shape())  # [16, 16]
        
        G_conv5 = Deconv2d(G_h4, output_dim=256, batch_size=BATCH_SIZE, name='G_conv5')
        G_bn5 = BatchNormalization(G_conv5, name='G_bn5')    
        G_h5 = tf.nn.relu(G_bn5)
        print(G_h5.get_shape())  # [32, 32]
        
        G_conv6 = Deconv2d(G_h5, output_dim=3, batch_size=BATCH_SIZE, name='G_conv6')
        G_r6 = tf.reshape(G_conv6, [-1, 64, 64, 3])
        
        return tf.nn.sigmoid(G_r6)
        
        
print('Build NN')

X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
z = tf.placeholder(tf.float32, shape=[None, 100])

G = Generator(z, False, 'G_net')
D_real = Discriminator(X, False, 'D_net')
D_fake = Discriminator(G, True, 'D_net')

#D_loss = - tf.reduce_mean(tf.log(D_real)) - tf.reduce_mean(
#                                            tf.log(np.ones_like(D_fake)-D_fake))
#G_loss = tf.reduce_mean(tf.log(np.ones_like(D_fake) - D_fake))
D_loss = -tf.reduce_mean(tf.log(D_real) - tf.log(D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

LR_D = 1e-5
LR_G = 2e-5

vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('D_net/')]
g_params = [v for v in vars if v.name.startswith('G_net/')]

#d_params = [var for var in tf.global_variables() if 'D_net' in var.name]
#g_params = [var for var in tf.global_variables() if 'G_net' in var.name]

D_solver = tf.train.AdamOptimizer(
           learning_rate=LR_D, beta1=0.1).minimize(D_loss, var_list=d_params)
G_solver = tf.train.AdamOptimizer(
           learning_rate=LR_G, beta1=0.3).minimize(G_loss, var_list=g_params)

# TODO: clip_op = tf.assign(x, tf.clip(x, 0, np.infty))

# Start tf session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())

    D_loss_vals = []
    G_loss_vals = []

    iteration = int(img_data.shape[0] / BATCH_SIZE)
    
    for e in range(EPOCHS):

        for i in range(iteration):
            x, _ = sample_batch(img_data, label_data)
            rand = np.random.uniform(0., 1., size=[BATCH_SIZE, 100])
            _, D_loss_curr = sess.run([D_solver, D_loss], {X: x, z: rand})
            if i % 4 == 0:
                rand = np.random.uniform(0., 1., size=[BATCH_SIZE, 100])
                _, G_loss_curr = sess.run([G_solver, G_loss], {z: rand})

            D_loss_vals.append(D_loss_curr)
            G_loss_vals.append(G_loss_curr)

            print("\r%d / %d: %e, %e" % (i, iteration, D_loss_curr, G_loss_curr))

        data = sess.run(G, {z: rand})
        scipy.misc.imsave('result/outfile_%d.jpg' % e, np.squeeze(data[e]))
        print('save image %d' % e)
        
        #plot(data, D_loss_vals, G_loss_vals, e, EPOCHS * iteration)


