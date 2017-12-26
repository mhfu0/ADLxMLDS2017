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
              'brown eyes', 'red eyes', 'blue eyes']

index2hair = dict(list(enumerate(hair_colors)))
index2eye = dict(list(enumerate(eye_colors)))

hair2index = dict(list(zip(index2hair.values(), index2hair.keys())))
eye2index = dict(list(zip(index2eye.values(), index2eye.keys())))

y_dim = len(hair_colors) + len(eye_colors)  # y_dim = 23
z_dim = y_dim*4  # z_dim = 23*4

########## Load data ########## 
try:
    with open('img_data.pkl', 'rb') as f:
        img_data = pickle.load(f)
    with open('label_data.pkl', 'rb') as f:
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
        
        if hair_idx_list:
            sys.stdout.write('hair ')
            hair_idx = hair2index[hair_idx_list[0]]
            hair_one_hot = np.zeros(len(hair2index))
            hair_one_hot[hair_idx] = 1.
            eye_one_hot = np.zeros(len(eye2index))

            label = np.array(list(hair_one_hot) + list(eye_one_hot))
            img_list.append(img_r)
            label_list.append(label)
            
        if eye_idx_list:
            sys.stdout.write('eyes ')
            eye_idx = eye2index[eye_idx_list[0]]
            eye_one_hot = np.zeros(len(eye2index))
            eye_one_hot[eye_idx] = 1.
            hair_one_hot = np.zeros(len(hair2index))

            label = np.array(list(hair_one_hot) + list(eye_one_hot))
            img_list.append(img_r)
            label_list.append(label)
            
        print('')

    img_data = np.array(img_list)
    label_data = np.array(label_list)

    with open('img_data.pkl', 'wb') as f:
        pickle.dump(img_data, f, protocol=4)
    with open('label_data.pkl', 'wb') as f:
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
def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def Generator(z, y, reuse=False):
    bs = z.get_shape().as_list()[0]
    
    with tf.variable_scope('g_net') as sc:
        if reuse:
            sc.reuse_variables()
    
        z_y = tf.concat([z, y], axis=1)
        
        fc = tc.layers.fully_connected(z_y, 4*4*256, weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        fc = tf.layers.batch_normalization(fc, training=True)
        fc = tf.reshape(fc, [-1, 4, 4, 256])
        fc = tf.nn.relu(fc)

        conv1 = tc.layers.convolution2d_transpose(fc, 128, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02),activation_fn=None)
        conv1 = tf.layers.batch_normalization(conv1, training=True)
        conv1 = tf.nn.relu(conv1)

        conv2 = tc.layers.convolution2d_transpose(conv1, 64, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02),activation_fn=None)
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        conv2 = tf.nn.relu(conv2)

        conv3 = tc.layers.convolution2d_transpose(conv2, 32, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02),activation_fn=None)
        conv3 = tf.layers.batch_normalization(conv3, training=True)
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
        conv1 = tc.layers.convolution2d(conv1, 32, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        conv1 = tf.layers.batch_normalization(conv1, training=True)
        conv1 = leaky_relu(conv1)
        
        conv2 = tc.layers.convolution2d(conv1, 64, [5, 5], [2, 2], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        conv2 = leaky_relu(conv2)
        
        conv3 = tc.layers.convolution2d(
                conv2, 128, [5, 5], [2, 2],
                padding='same',
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None)
        conv3 = tf.layers.batch_normalization(conv3, training=True)
        conv3 = leaky_relu(conv3)
        
        flat = tc.layers.flatten(conv3)
        fc_y = tf.concat([flat, y], axis=1)
        fc = tc.layers.fully_connected(fc_y, z_dim+y_dim, weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        fc = leaky_relu(fc)
        fc = tf.nn.dropout(fc, 0.3)
        
        #fc_y = tf.concat([fc, y], axis=1)
        
        fc2 = tc.layers.fully_connected(fc, 1, weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        
        return tf.nn.sigmoid(fc2)

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

G = Generator(z, y, reuse=False)
D_real = Discriminator(x, y, reuse=False)
D_fake = Discriminator(G, y, reuse=True)
D_real_wrong = Discriminator(x, w, reuse=True)
D_fake_wrong = Discriminator(G, w, reuse=True)

#d_loss = -tf.reduce_mean(tf.log(D_real) - tf.log(D_fake))
#g_loss = -tf.reduce_mean(tf.log(D_fake))

# d_loss for DCGAN
#d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))) + #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))

#d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))) + (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_real_wrong))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake_wrong))))/3
d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))) + (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_real_wrong))))/2

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

LR_D = 1e-4
LR_G = 1e-4

BATCH_SIZE = 64

d_vars = [var for var in tf.global_variables() if "d_net" in var.name]
g_vars = [var for var in tf.global_variables() if "g_net" in var.name]

d_updates = tf.train.AdamOptimizer(LR_D, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_vars)
g_updates = tf.train.AdamOptimizer(LR_G, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_vars)

TRAIN = True

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    checkpoint_path = os.path.join('dcgan/', 'network.ckpt')
    
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

        for ep in range(200):
            for i in range(200):
                imgs, labels = sample_batch(img_data, label_data)
                
                w_labels = create_wrong_label(labels)

                rand = np.random.uniform(0., 1., size=[BATCH_SIZE, z_dim])
                _, d_loss_curr = sess.run([d_updates, d_loss], {x: imgs, z: rand, y: labels, w: w_labels})
                
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
                test_rand = np.random.uniform(0., 1., size=[8, z_dim])
                condition = np.zeros(y_dim)
                condition[condition_list[i]] = 1.
                condition = np.tile(condition, (8,1))
            
                data = sess.run(G, {z: test_rand, y: condition})
                result_list.append(data)
            
            result_path = 'result/train_output_%d.jpg' % ep
            plot_result_list(result_list, result_path)
            
            print('Saving model parameters...')
            saver.save(sess, checkpoint_path)
            # TODO: dump loss
            