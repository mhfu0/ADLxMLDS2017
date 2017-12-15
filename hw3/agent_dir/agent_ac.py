# Reference of the code:
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/8_Actor_Critic_Advantage/AC_CartPole.py

import numpy as np
import tensorflow as tf

import os, sys
import random
from collections import deque
import pickle

from agent_dir.agent import Agent

# Hyperparameter settings
MAX_EPISODE = 3000
MAX_EP_STEPS = 10000  # maximum time step in one episode
GAMMA = 0.9  # reward discount in TD-error
LEARNING_RATE_A = 0.001 # learning rate for actor
LEARNING_RATE_C = 0.01  # learning rate for critic

class Actor(object):
    def __init__(self, sess, observation_space, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, observation_space]) # state
        self.a = tf.placeholder(tf.int32, None) # action
        self.td_error = tf.placeholder(tf.float32, None) # TD-error
        
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(inputs=self.s,
                                 units=20,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1))

            self.acts_prob = tf.layers.dense(inputs=l1,
                                             units=n_actions, 
                                             activation=tf.nn.softmax,   # get action probabilities
                                             kernel_initializer=tf.random_normal_initializer(0., .1),
                                             bias_initializer=tf.constant_initializer(0.1))

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

class Critic(object):
    def __init__(self, sess, observation_space, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, observation_space], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(inputs=self.s,
                                 units=128,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1))

            self.v = tf.layers.dense(inputs=l1,
                                     units=1,
                                     activation=None,
                                     kernel_initializer=tf.random_normal_initializer(0., .1),
                                     bias_initializer=tf.constant_initializer(0.1))

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
            
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

class Agent_AC(Agent):
    def __init__(self, env, args):

        super(Agent_AC,self).__init__(env)
        
        # Choose simpler game here
        import gym
        self.env = gym.make('CartPole-v0')
        self.env = self.env.unwrapped
        self.env.seed(1239)

        self.observation_space = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        
        # Configuration for nlg-workstation
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        
        # Parameter settings
        self.checkpoints_dir = 'ac_network/'
        self.gamma = GAMMA

        # Reproducibility
        random.seed(1239)
        np.random.seed(1239)
        tf.set_random_seed(1239)
        
        # Build network
        self.sess = tf.Session()
        
        self.actor = Actor(self.sess, observation_space=self.observation_space, n_actions=self.n_actions, lr=LEARNING_RATE_A)
        self.critic = Critic(self.sess, observation_space=self.observation_space, lr=LEARNING_RATE_C)  

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(self.checkpoints_dir,'ac_network.ckpt')
        
        if args.test_pg:
            print('loading trained model')
            self.load_checkpoint()


    def init_game_setting(self):
        pass

    def train(self):
        max_reward = 0
        running_reward = None
        
        for i_episode in range(MAX_EPISODE):
            
            state = self.env.reset()
            timestep = 0
            track_reward = []
            
            while True:
                action = self.actor.choose_action(state)
                state_, reward, done, _ = self.env.step(action)
                
                if done:
                    reward = -20
                    
                track_reward.append(reward)
                td_error = self.critic.learn(state, reward, state_)  # gradient = grad[r + gamma * V(s_) - V(s)]
                self.actor.learn(state, action, td_error)     # true_gradient = grad[logPi(s,a) * td_error]
                state = state_
                timestep += 1

                if done or timestep >= MAX_EP_STEPS:
                    ep_rewards_sum = sum(track_reward)
                    if ep_rewards_sum >= max_reward:
                        max_reward = ep_rewards_sum
                        #self.save_checkpoint()

                    if running_reward == None:
                        running_reward = ep_rewards_sum
                    else:
                        running_reward = running_reward * 0.99 + ep_rewards_sum * 0.01
                    print("episode:", i_episode, ", running_reward:", int(running_reward))
                    break
 
    def make_action(self, observation, test=True):
        return self.env.get_random_action()
 
    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)