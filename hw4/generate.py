import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

import os, sys
import pickle
import itertools
import collections

import skimage
import skimage.io
import skimage.transform
import scipy.misc

from model import Generator, Discriminator

def plot_result(result, path_pre):
    for i in range(len(result)):
        path = '%s_%d.jpg' % (path_pre, i+1)
        scipy.misc.imsave(path, result[i])
        print('Saving Image in %s' % path)

def next_noise_batch(size, dim):
    z_sampler = scipy.stats.truncnorm((-1 - 0.) / 1., (1 - 0.) / 1., loc=0., scale=1)
    return z_sampler.rvs([size, dim])

# Create feature indices mapping
hair_colors = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 'pink hair',
               'blue hair', 'black hair', 'brown hair', 'blonde hair', 'unk']
eye_colors = ['gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes', 
              'brown eyes', 'red eyes', 'blue eyes', 'unk']
              
index2hair = dict(list(enumerate(hair_colors)))
index2eye = dict(list(enumerate(eye_colors)))

hair2index = dict(list(zip(index2hair.values(), index2hair.keys())))
eye2index = dict(list(zip(index2eye.values(), index2eye.keys())))

y_dim = len(hair_colors) + len(eye_colors)  # y_dim = 25
z_dim = y_dim*4  # z_dim = 100

BATCH_SIZE = 64
dump_dir = 'samples/'
model_path = 'trained_model/'

# Build DCGAN model
g_net = Generator()
d_net = Discriminator()

z = tf.placeholder(tf.float32, [None, z_dim])

label = tf.placeholder(tf.float32, [None, y_dim], name='label')
img = tf.placeholder(tf.float32, [None, 64, 64, 3], name='img')
w_label = tf.placeholder(tf.float32, [None, y_dim], name='w_label')
w_img = tf.placeholder(tf.float32, [None, 64, 64, 3], name='w_img')

r_img, r_label = img, label
f_img = g_net(r_label, z)
sampler = tf.identity(g_net(r_label, z, reuse=True, train=False), name='sampler') 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

checkpoint_path = os.path.join(model_path, 'network.ckpt-16500')
saver.restore(sess, checkpoint_path)
print('Load model checkpoint')

# create random noise to observe output
try:
    rand_t = pickle.load(open(os.path.join(model_path, 'fixed.pkl'), 'rb'))
except:
    rand_t = next_noise_batch(5, z_dim)
    pickle.dump(rand_t, open(os.path.join(model_path, 'fixed.pkl'), 'wb'))
    
with open(sys.argv[1], 'r') as f:
    for line in f:
        pair = line.strip('\n').split(',')
        idx = int(pair[0])
        text = pair[1]
        
        hair_onehot = np.zeros(len(hair2index))
        for h in hair_colors:
            if h in text:
                hair_onehot[hair2index[h]] = 1.
                break
            hair_onehot[hair2index['unk']] = 1.
        
        eye_onehot = np.zeros(len(eye2index))
        for e in eye_colors:
            if e in text:
                eye_onehot[eye2index[e]] = 1.
                break
            eye_onehot[eye2index['unk']] = 1.
            
        condition = np.array(list(hair_onehot) + list(eye_onehot))
        condition = np.tile(condition, (5,1))
        
        feed_dict={z: rand_t, label: condition}
        f_imgs = sess.run(sampler, feed_dict=feed_dict)
        
        result_path = os.path.join(dump_dir, 'sample_%d' % idx)
        plot_result(f_imgs, result_path)