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
               'blue hair', 'black hair', 'brown hair', 'blonde hair', 'unk']
eye_colors = ['gray eyes', 'black eyes', 'orange eyes', 'pink eyes',
              'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes', 
              'brown eyes', 'red eyes', 'blue eyes', 'unk']

index2hair = dict(list(enumerate(hair_colors)))
index2eye = dict(list(enumerate(eye_colors)))

hair2index = dict(list(zip(index2hair.values(), index2hair.keys())))
eye2index = dict(list(zip(index2eye.values(), index2eye.keys())))

y_dim = len(hair_colors) + len(eye_colors)  # y_dim = 25
z_dim = y_dim*4  # z_dim = 100

########## Load data ########## 
try:
    with open('img_data_unk.pkl', 'rb') as f:
        img_data = pickle.load(f)
    with open('label_data_unk.pkl', 'rb') as f:
        label_data = pickle.load(f)
    print('Preprocessed data loaded')
    
except:
    MANUAL_TAGGING = False
    
    print('Load tag data')
    file = open('tags_clean.csv', 'r')
    text_labels = []
    
    for line in file:
        line = line.rstrip('\n\t').split(',')
        img_id = line[0]
    
        line = line[1]
        raw_tags = line.split('\t')
        raw_tags = list(map(lambda x: x.split(':')[0], raw_tags)) 
    
        pair = ([], []) 
        for t in raw_tags:
            if t in hair_colors: pair[0].append(t)
            if t in eye_colors:  pair[1].append(t)
    
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
    
    ##########  End loading tag data ##########
    print('End loading %d tag data' % len(text_labels))
    
    '''
    idx_label_tuples = []
    for idx, label in enumerate(text_labels):
        if label[0] in hair_colors and label[1] in eye_colors:
            idx_label_tuples.append((idx, label))
        else:
            pass
    
    print('Pick %d data' % len(idx_label_tuples))
    '''
    
    ########## Load image data ##########         
    print('Load raw image data')
    img_list = []
    label_list = []

    for idx, pair in enumerate(text_labels):
        # Read image
        print('Read image faces/%d.jpg' % idx)
        img = skimage.io.imread('faces/%d.jpg' % idx)
        img_r = skimage.transform.resize(img, (64,64))
        
        # encode labels
        hair_idx_list = pair[0]
        eye_idx_list = pair[1]
        
        # skip if both labels not valid
        if (not hair_idx_list) and (not eye_idx_list):
            sys.stdout.write('neither\n')
            continue
        
        if hair_idx_list and eye_idx_list:
            sys.stdout.write('both ')
            hair_idx = hair2index[hair_idx_list[0]]
            hair_one_hot = np.zeros(len(hair2index))
            hair_one_hot[hair_idx] = 1.
            
            eye_idx = eye2index[eye_idx_list[0]]
            eye_one_hot = np.zeros(len(eye2index))
            eye_one_hot[eye_idx] = 1.
    
            label = np.array(list(hair_one_hot) + list(eye_one_hot))
            img_list.append(img_r)
            label_list.append(label)
        
        
        elif hair_idx_list:
            sys.stdout.write('hair ')
            hair_idx = hair2index[hair_idx_list[0]]
            hair_one_hot = np.zeros(len(hair2index))
            hair_one_hot[hair_idx] = 1.
            eye_one_hot = np.zeros(len(eye2index))
            eye_one_hot[eye2index['unk']] = 1.

            label = np.array(list(hair_one_hot) + list(eye_one_hot))
            img_list.append(img_r)
            label_list.append(label)
            
        elif eye_idx_list:
            sys.stdout.write('eyes ')
            eye_idx = eye2index[eye_idx_list[0]]
            eye_one_hot = np.zeros(len(eye2index))
            eye_one_hot[eye_idx] = 1.
            hair_one_hot = np.zeros(len(hair2index))
            hair_one_hot[hair2index['unk']] = 1.

            label = np.array(list(hair_one_hot) + list(eye_one_hot))
            img_list.append(img_r)
            label_list.append(label)
  
        print('')
        
    img_data = np.array(img_list)
    label_data = np.array(label_list)

    with open('img_data_unk.pkl', 'wb') as f:
        pickle.dump(img_data, f, protocol=4)
    with open('label_data_unk.pkl', 'wb') as f:
        pickle.dump(label_data, f)

print(img_data.shape, label_data.shape)
print('End creating %d data' % len(img_data))

print(label_data[0])
########## End loading data ##########

def sample_batch(img_data, label_data, batch_size=64):
    assert(len(img_data)==len(label_data))
    
    order = np.random.choice(np.arange(len(img_data)), batch_size)
    
    img_batch = img_data[order]
    label_batch = label_data[order]
    
    # data augmentation
    # choose random data to flip/rotate
    aug_idx = np.random.choice(np.arange(batch_size), 16)
    for idx in aug_idx:
        img_batch[idx] = np.fliplr(img_batch[idx])
    
    '''  
    aug_idx = np.random.choice(np.arange(batch_size), 32)
    for idx in aug_idx[:16]:
        #img_batch[idx] = scipy.ndimage.rotate(img_batch[idx], 5, cval=1.)
        img_batch[idx] = scipy.misc.imrotate(img_batch[idx], 5)
    for idx in aug_idx[16:]:
        #img_batch[idx] = scipy.ndimage.rotate(img_batch[idx], -5, cval=1.)
        img_batch[idx] = scipy.misc.imrotate(img_batch[idx], -5)
    '''
    return img_batch, label_batch

def create_wrong_label(labels):
    label_batch = labels.copy()
    
    for i in range(len(label_batch)):
        sh = label_batch[i].copy()
        while np.array_equal(sh, label_batch[i]):
            np.random.shuffle(sh)
        label_batch[i] = sh
    
    return label_batch

def shuffle(X, Y):
    order = np.arange(len(X))
    np.random.shuffle(order)
    return (X[order], Y[order])

########## NN Settings ##########
'''
def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def Generator(z, y, reuse=False, train=True):
    bs = z.get_shape().as_list()[0]
    
    with tf.variable_scope('g_net') as sc:
        if reuse:
            sc.reuse_variables()
    
        z_y = tf.concat([z, y], axis=1)
        
        fc = tc.layers.fully_connected(z_y, 4*4*256, weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        fc = tf.layers.batch_normalization(fc, training=train)
        fc = tf.reshape(fc, [-1, 4, 4, 256])
        fc = tf.nn.relu(fc)

        conv1 = tc.layers.convolution2d_transpose(fc, 128, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02),activation_fn=None)
        conv1 = tf.layers.batch_normalization(conv1, training=train)
        conv1 = tf.nn.relu(conv1)

        conv2 = tc.layers.convolution2d_transpose(conv1, 64, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02),activation_fn=None)
        conv2 = tf.layers.batch_normalization(conv2, training=train)
        conv2 = tf.nn.relu(conv2)

        conv3 = tc.layers.convolution2d_transpose(conv2, 32, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02),activation_fn=None)
        conv3 = tf.layers.batch_normalization(conv3, training=train)
        conv3 = tf.nn.relu(conv3)
        
        conv4 = tc.layers.convolution2d_transpose(conv3, 3, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        conv4 = tf.nn.tanh(conv4)
        
        return conv4

def Discriminator(x, y, reuse=False):
    bs = x.get_shape().as_list()[0]
    
    with tf.variable_scope('d_net') as sc:
        if reuse:
            sc.reuse_variables()
        
        conv1 = tc.layers.convolution2d(x, 32, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        conv1 = tf.layers.batch_normalization(conv1, training=True)
        conv1 = leaky_relu(conv1)
        #conv1 = tc.layers.convolution2d(conv1, 32, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        #conv1 = tf.layers.batch_normalization(conv1, training=True)
        #conv1 = leaky_relu(conv1)
        
        conv2 = tc.layers.convolution2d(conv1, 64, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        conv2 = leaky_relu(conv2)
        
        conv3 = tc.layers.convolution2d(conv2, 128, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        conv3 = tf.layers.batch_normalization(conv3, training=True)
        conv3 = leaky_relu(conv3)
        
        ##################################
        flat = tc.layers.flatten(conv3)
        
        y = tc.layers.fully_connected(y, 200, weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        y = tf.layers.batch_normalization(y, training=True)
        y = leaky_relu(y)
        
        fc_y = tf.concat([flat, y], axis=1)
        fc = tc.layers.fully_connected(fc_y, z_dim+y_dim, weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        fc = leaky_relu(fc)
        fc = tf.nn.dropout(fc, 0.3)
        
        #fc_y = tf.concat([fc, y], axis=1)
        
        fc2 = tc.layers.fully_connected(fc, 1, weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        
        return tf.nn.sigmoid(fc2)
        ##################################
        #y = tc.layers.fully_connected(y, 10, weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        #y = tf.layers.batch_normalization(y, training=True)
        #y = leaky_relu(y)
        
        y = tf.expand_dims(tf.expand_dims(y, 1), 2)
        y = tf.tile(y, [1, 8, 8, 1])
        conv3_y = tf.concat([conv3, y], axis=-1)
        
        conv4 = tc.layers.convolution2d(conv3_y, 128, [1, 1], [1, 1], padding='same', activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
        conv4 = tf.layers.batch_normalization(conv4, training=True)
        conv4 = leaky_relu(conv4)
        
        conv5 = tc.layers.convolution2d(conv4, 1, [8, 8], [1, 1], padding='valid',activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
        output = tf.squeeze(conv5, [1, 2, 3])
        
        return output
        
def plot_result_list(result_list, path):
    # combine 8*8=64 images
    # input shape=(64, 64, 64, 3)
    comb_list = []
    for i in range(len(result_list)):
        comb_list.append(np.hstack(result_list[i]))
    comb_img = np.vstack(comb_list)
    
    scipy.misc.imsave(path, comb_img)
    print('Saving Image in %s' % path)

x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

w = tf.placeholder(tf.float32, shape=[None, y_dim]) # wrong label
xw = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

G = Generator(z, y, reuse=False)
D_real = Discriminator(x, y, reuse=False)
D_fake = Discriminator(G, y, reuse=True)
D_real_wrong = Discriminator(x, w, reuse=True)
D_wrong_real = Discriminator(xw, y, reuse=True)

sampler = Generator(z, y, reuse=True, train=False)

#d_loss = -tf.reduce_mean(tf.log(D_real) - tf.log(D_fake))
#g_loss = -tf.reduce_mean(tf.log(D_fake))

# d_loss for DCGAN
#d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))) + #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))

#d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))) + (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_real_wrong))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake_wrong))))/3
d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))*.33 + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_wrong, labels=tf.zeros_like(D_real_wrong)))*.33 + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_wrong_real, labels=tf.zeros_like(D_wrong_real)))*.33

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

LR_D = 2e-4
LR_G = 1e-4

BATCH_SIZE = 64

d_vars = [var for var in tf.global_variables() if 'd_net' in var.name]
g_vars = [var for var in tf.global_variables() if 'g_net' in var.name]

d_updates = tf.train.AdamOptimizer(LR_D, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_vars)
g_updates = tf.train.AdamOptimizer(LR_G, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_vars)

TRAIN = True

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    checkpoint_path = os.path.join('cdcgan/', 'network.ckpt')
    
    if not TRAIN:
        print('Loading model parameters...')
        saver.restore(sess, checkpoint_path)
        test_rand = np.random.uniform(0., 1., size=[5, z_dim])
        test_label = np.zeros(y_dim)
        test_label[np.array([5,19])] = 1.
        test_label = np.tile(test_label, (5,1))
        
        data = sess.run(G, {z: test_rand, y: test_label})
        for i in range(5):
            scipy.misc.imsave('early/sample_1_%d.jpg' % (i+1), np.squeeze(data[i]))
        
    else:
        d_loss_vals = []
        g_loss_vals = []

        test_rand = np.random.uniform(0., 1., size=[8, z_dim])
        
        for ep in range(200):
            for i in range(200):
                imgs, labels = sample_batch(img_data, label_data)
                w_imgs, w_labels = sample_batch(img_data, label_data)
                
                labels += np.random.uniform(0., .3, size=[BATCH_SIZE, y_dim])
                labels *= 1. / labels.max()
                w_labels += np.random.uniform(0., .3, size=[BATCH_SIZE, y_dim])
                w_labels *= 1. / w_labels.max()
                

                rand = np.random.uniform(0., 1., size=[BATCH_SIZE, z_dim])
                _, d_loss_curr = sess.run([d_updates, d_loss], {x: imgs, z: rand, y: labels, w: w_labels, xw:w_imgs})
                
                gen_train_step = 1
                if i % gen_train_step == 0:
                    rand = np.random.uniform(0., 1., size=[BATCH_SIZE, z_dim])
                    _, g_loss_curr = sess.run([g_updates, g_loss], {z: rand, y: labels})

                d_loss_vals.append(d_loss_curr)
                g_loss_vals.append(g_loss_curr)            

                print("%d / %d: %e, %e" % (i, 200, d_loss_curr, g_loss_curr))
            
            # show result: 8 images for each condition
            condition_list = []
            condition_list.append(np.array([5,19]))
            for i in range(7):
                condition_list.append(np.array([0+i,12+i]))
            
            result_list = []
            for i in range(8):
                
                condition = np.zeros(y_dim)
                condition[condition_list[i]] = 1.
                condition = np.tile(condition, (8,1))
            
                data = sess.run(G, {z: test_rand, y: condition})
                result_list.append(data)
            
            result_path = 'dump/train_output_%d.jpg' % ep
            plot_result_list(result_list, result_path)
            
            print('Saving model parameters...')
            saver.save(sess, checkpoint_path, global_step=ep*200)
            # TODO: dump loss
'''
from model import Generator, Discriminator

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

g_net = Generator()
d_net = Discriminator()

seq = tf.placeholder(tf.float32, [None, y_dim], name='seq')
img = tf.placeholder(tf.float32, [None, 64, 64, 3], name='img')
z = tf.placeholder(tf.float32, [None, z_dim])

w_seq = tf.placeholder(tf.float32, [None, y_dim], name='w_seq')
w_img = tf.placeholder(tf.float32, [None, 64, 64, 3], name='w_img')

r_img, r_seq = img, seq
f_img = g_net(r_seq, z)

sampler = tf.identity(g_net(r_seq, z, reuse=True, train=False), name='sampler') 

d = d_net(r_seq, r_img, reuse=False)
d_1 = d_net(r_seq, f_img)
d_2 = d_net(w_seq, img)
d_3 = d_net(r_seq, w_img)
   
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_1, labels=tf.ones_like(d_1)))
d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d, labels=tf.ones_like(d))) + (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_1, labels=tf.zeros_like(d_1))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_2, labels=tf.zeros_like(d_2))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_3, labels=tf.zeros_like(d_3))) ) / 3

global_step = tf.Variable(0, name='g_global_step', trainable=False)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    d_updates = tf.train.AdamOptimizer(2e-4, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_net.vars)
    g_updates = tf.train.AdamOptimizer(2e-4, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_net.vars, global_step=global_step)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())

def next_noise_batch(size, dim):
    z_sampler = scipy.stats.truncnorm((-1 - 0.) / 1., (1 - 0.) / 1., loc=0., scale=1)
    return z_sampler.rvs([size, dim])

img_dir = 'dump_/'

def plot_result(result_list, path):
    # combine 8*8=64 images
    # input shape=(64, 64, 64, 3)
    comb_list = []
    for i in range(len(result_list)):
        comb_list.append(np.hstack(result_list[i]))
    comb_img = np.vstack(comb_list)
    
    scipy.misc.imsave(path, comb_img)
    print('Saving Image in %s' % path)

BATCH_SIZE = 64
d_epoch = 1
rand_t = next_noise_batch(8, z_dim)

for t in range(20000):
    d_cost = 0
    g_cost = 0
    
    for d_ep in range(d_epoch):
        img_b, tags_b = sample_batch(img_data, label_data)
        w_img_b, w_tags_b = sample_batch(img_data, label_data)
        rand = next_noise_batch(BATCH_SIZE, z_dim)
        
        feed_dict={seq:tags_b,img:img_b,z:rand,w_seq:w_tags_b,w_img:w_img_b}
        _, loss = sess.run([d_updates, d_loss], feed_dict=feed_dict)
        d_cost += loss/d_epoch
        
        rand = next_noise_batch(BATCH_SIZE, z_dim)
        feed_dict={seq:tags_b,img:img_b,z:rand,w_seq:w_tags_b,w_img:w_img_b}
        
        _, loss, step = sess.run([g_updates, g_loss, global_step], feed_dict=feed_dict)
        current_step = tf.train.global_step(sess, global_step)
        g_cost = loss
        
        if current_step % 20 == 0:
            print("Current_step {}".format(current_step))
            print("Discriminator loss :{}".format(d_cost))
            print("Generator loss     :{}".format(g_cost))
            print("---------------------------------")
        if current_step % 500 == 0:
            checkpoint_path = os.path.join('cdcgan/', 'network.ckpt')
            path = saver.save(sess, checkpoint_path, global_step=current_step)
            print("\nSaved model checkpoint to {}\n".format(path))
        if current_step % 500 == 0:
            result_list = []
            
            condition = np.zeros(y_dim)
            condition[np.array([5,19])] = 1.
            condition = np.tile(condition, (8,1))
            
            feed_dict={z:rand_t, seq:condition}
            f_imgs = sess.run(sampler, feed_dict=feed_dict)
            
            for i in range(7):
                condition = np.zeros(y_dim)
                condition[np.array([i,13+i])] = 1.
                condition = np.tile(condition, (8,1))
                
                feed_dict={z:rand_t, seq:condition}
                f_imgs = sess.run(sampler, feed_dict=feed_dict)
                
                result_list.append(f_imgs)
            
            result_path = 'dump_/train_output_%d.jpg' % (t+1)
            plot_result(result_list, result_path)
            print("Dump test image")

sess.close()