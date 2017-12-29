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

########## Load data ########## 
try:
    with open('img_data_unk.pkl', 'rb') as f:
        img_data = pickle.load(f)
    with open('label_data_unk.pkl', 'rb') as f:
        label_data = pickle.load(f)
    print('Preprocessed data loaded')
    
except:
    print('Load tag data...')
    with open('tags_clean.csv', 'r') as file:
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
        
            text_labels.append(pair)
    print('End loading %d tag data' % len(text_labels))
           
    print('Load raw image data...')
    
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
            print('data skipped')
            continue
        
        try:
            hair_idx = hair2index[hair_idx_list[0]]
        except:
            hair_idx = hair2index['unk']
        hair_one_hot = np.zeros(len(hair2index))
        hair_one_hot[hair_idx] = 1.
        
        try:
            eye_idx = eye2index[eye_idx_list[0]]
        except:
            eye_idx = eye2index['unk']
        eye_one_hot = np.zeros(len(eye2index))
        eye_one_hot[eye_idx] = 1.

        label = np.array(list(hair_one_hot) + list(eye_one_hot))
        
        img_list.append(img_r)
        label_list.append(label)
        
    img_data = np.array(img_list)
    label_data = np.array(label_list)

    with open('img_data_unk.pkl', 'wb') as f:
        pickle.dump(img_data, f, protocol=4)
    with open('label_data_unk.pkl', 'wb') as f:
        pickle.dump(label_data, f)

print(img_data.shape, label_data.shape)
print('End loading/creating %d data' % len(img_data))

########## End loading data ##########
def next_noise_batch(size, dim):
    z_sampler = scipy.stats.truncnorm((-1 - 0.) / 1., (1 - 0.) / 1., loc=0., scale=1)
    return z_sampler.rvs([size, dim])

def sample_batch(img_data, label_data, batch_size=64):
    assert(len(img_data) == len(label_data))
    
    order = np.random.choice(np.arange(len(img_data)), batch_size)
    
    img_batch = img_data[order]
    label_batch = label_data[order]
    
    # data augmentation: choose random data to flip/rotate
    aug_idx = np.random.choice(np.arange(batch_size), 16)
    for idx in aug_idx:
        img_batch[idx] = np.fliplr(img_batch[idx])

    aug_idx = np.random.choice(np.arange(batch_size), 16)
    for idx in aug_idx[:8]:
        tmp = scipy.misc.imrotate(img_batch[idx], 5) / 255.
        tmp[np.where(tmp == 0.)] = 1.
        img_batch[idx] = tmp
    for idx in aug_idx[8:]:
        tmp = scipy.misc.imrotate(img_batch[idx], -5) / 255.
        tmp[np.where(tmp == 0.)] = 1.
        img_batch[idx] = tmp
    
    return img_batch, label_batch

def plot_result(result_list, path):
    # combine 8*8=64 images
    # input shape=(64, 64, 64, 3)
    comb_list = []
    for i in range(len(result_list)):
        comb_list.append(np.hstack(result_list[i]))
    comb_img = np.vstack(comb_list)
    
    scipy.misc.imsave(path, comb_img)
    print('Saving Image in %s' % path)

# NN settings
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

BATCH_SIZE = 64
iters = 50000
lr_d = 2e-4
lr_g = 2e-4
dump_dir = 'dump/'
model_path = 'cdcgan/'

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

d = d_net(r_label, r_img, reuse=False)
d_1 = d_net(r_label, f_img)
d_2 = d_net(w_label, r_img)
d_3 = d_net(r_label, w_img)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_1, labels=tf.ones_like(d_1)))
d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d, labels=tf.ones_like(d))) + ( tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_1, labels=tf.zeros_like(d_1))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_2, labels=tf.zeros_like(d_2))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_3, labels=tf.zeros_like(d_3))) ) / 3

global_step = tf.Variable(0, name='g_global_step', trainable=False)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    d_updates = tf.train.AdamOptimizer(lr_d, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_net.vars)
    g_updates = tf.train.AdamOptimizer(lr_g, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_net.vars, global_step=global_step)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

# create random noise to observe output
grids = 16
try:
    rand_t = pickle.load(open('z.pkl', 'rb'))
except:
    rand_t = next_noise_batch(grids, z_dim)
    pickle.dump(rand_t, open('z.pkl', 'wb'))

for _ in range(iters):
    img_batch, label_batch = sample_batch(img_data, label_data)
    w_img_batch, w_label_batch = sample_batch(img_data, label_data)
    rand = next_noise_batch(BATCH_SIZE, z_dim)
    
    feed_dict={label: label_batch, img: img_batch, w_label: w_label_batch, w_img: w_img_batch, z:rand}
    _, d_loss_cur = sess.run([d_updates, d_loss], feed_dict=feed_dict)
    
    rand = next_noise_batch(BATCH_SIZE, z_dim)
    
    feed_dict={label: label_batch, img: img_batch, w_label: w_label_batch, w_img: w_img_batch, z:rand}
    _, g_loss_cur, step = sess.run([g_updates, g_loss, global_step], feed_dict=feed_dict)
    
    current_step = tf.train.global_step(sess, global_step)
    
    if current_step % 20 == 0:
        print("Step {}/{}: {}, {}".format(current_step, iters, d_loss_cur, g_loss_cur))
    if current_step % 500 == 0:
        # Save model checkpoint
        checkpoint_path = os.path.join(model_path, 'network.ckpt')
        path = saver.save(sess, checkpoint_path, global_step=current_step)
        print("Save model checkpoint to {}".format(path))
        
        result_list = []
        for i in range(8):
            condition = np.zeros(y_dim)
            condition[np.array([11-i,24-i])] = 1.
            condition = np.tile(condition, (16,1))
            
            feed_dict={z: rand_t, label: condition}
            f_imgs = sess.run(sampler, feed_dict=feed_dict)
            
            result_list.append(f_imgs)
        
        result_path = os.path.join(dump_dir, 'train_output_%d.jpg' % (current_step))
        plot_result(result_list, result_path)
        print("Dump test image")

sess.close()
