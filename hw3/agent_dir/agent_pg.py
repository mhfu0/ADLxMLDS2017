import numpy as np
import tensorflow as tf
import scipy

import os, sys
import random
from collections import deque
import pickle

from agent_dir.agent import Agent

# Hyperparameter settings
LEARNING_RATE = 1e-4
NUM_EPISODES = 100000
MAX_NUM_STEPS = 10000

# Action values to send to gym environment
UP_ACTION = 2
DOWN_ACTION = 3

# Mapping from action values to outputs from the policy network
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}

class Agent_PG(Agent):
    def __init__(self, env, args):

        super(Agent_PG,self).__init__(env)
        
        # Configuration for nlg-workstation
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        
        # Parameter settings
        self.checkpoints_dir = 'pg_network/'
        self.gamma = args.gamma

        self.time_step = 0
        self.learning_rate = LEARNING_RATE
        self.state_dim = [80,80,1]
        self.reward_history = []
        
        # Reproducibility
        random.seed(1239)
        np.random.seed(1239)
        tf.set_random_seed(1239)
        
        # Build network
        self.create_network()
        
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(self.checkpoints_dir, 'pg_network.ckpt')
        
        if args.test_pg:
            print('Loading model parameters...')
            self.checkpoint_file = os.path.join(self.checkpoints_dir,'pg_network.ckpt')
            self.load_checkpoint()

    def init_game_setting(self):
        self.test_last_observation = self.env.reset()
        self.test_last_observation = self.prepro(self.test_last_observation)
        np.random.seed(23)

    def train(self):
        self.states = []
        self.actions = []
        self.rewards = []
        
        reward_moving_avg = None
        episode = 1
        
        while True:
            print('Starting episode %d...' % episode)
            done = False
            episode_reward_sum = 0
            n_rounds = 1
            
            last_observation = self.env.reset()
            last_observation = self.prepro(last_observation)
            action = self.env.action_space.sample()
            observation, _, _, _ = self.env.step(action)
            observation = self.prepro(observation)
            n_steps = 1
            
            while not done:
                observation_delta = observation - last_observation
                last_observation = observation
                up_probability = self.predict_action(observation_delta)[0]
                if np.random.uniform() < up_probability:
                    action = UP_ACTION
                else:
                    action = DOWN_ACTION

                observation, reward, done, _ = self.env.step(action)
                observation = self.prepro(observation)
                episode_reward_sum += reward
                n_steps += 1
                
                self.states.append(observation_delta)
                self.actions.append(action_dict[action])
                self.rewards.append(reward)

                if reward != 0:
                    n_rounds += 1
                    n_steps = 0
                if reward == -1:
                    print('Round %d: %d timesteps; lost...' % (n_rounds, n_steps))
                elif reward == +1:
                    print('Round %d: %d timesteps; won!' % (n_rounds, n_steps))

            print('Episode %d finished after %d rounds' % (episode, n_rounds-1))

            # exponentially smoothed version of reward
            if reward_moving_avg is None:
                reward_moving_avg = episode_reward_sum
            else:
                reward_moving_avg = reward_moving_avg * 0.99 + episode_reward_sum * 0.01
            print('Reward total was %.3f; discounted moving average of reward is %.3f' \
                % (episode_reward_sum, reward_moving_avg))
            self.reward_history.append(episode_reward_sum)
            
            # Discount and normalize rewards
            self.rewards = self.discount_rewards(self.rewards, self.gamma)
            self.rewards -= np.mean(self.rewards)
            self.rewards /= np.std(self.rewards)
            self.train_network(self.states, self.actions, self.rewards)
            self.states, self.actions, self.rewards = [], [], []
            ### Episode ends here ###
            
            if episode % 200 == 0:
                self.save_checkpoint()
                with open(os.path.join(self.checkpoints_dir, 'pg_reward_history.pickle'), 'wb') as f:
                    pickle.dump(self.reward_history, f)
                print('Model saved and reward history dumped...')

            episode += 1
                
    def make_action(self, observation, test=True):
        observation = self.prepro(observation)
        observation_delta = observation - self.test_last_observation
        self.test_last_observation = observation
        
        up_prob = self.predict_action(observation_delta)[0]
        action = UP_ACTION if np.random.uniform() < up_prob else DOWN_ACTION
        return action
    
    def create_network(self):
        # input layer
        self.observations = tf.placeholder(tf.float32,[None] + self.state_dim)
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(tf.float32, [None, 1])
        
        # conv layers weights
        W_conv1 = self.weight_variable([8,8,1,16])
        b_conv1 = self.bias_variable([16])
        
        W_conv2 = self.weight_variable([4,4,16,32])
        b_conv2 = self.bias_variable([32])
        
        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(self.observations,W_conv1,4) + b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)
        h_conv2_shape = h_conv2.get_shape().as_list()
        h_conv2_dim = h_conv2_shape[1]*h_conv2_shape[2]*h_conv2_shape[3]
        
        h_conv2_flat = tf.reshape(h_conv2,[-1,h_conv2_dim])
        
        # fc layers
        h = tf.layers.dense(h_conv2_flat,units=128,activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.up_probability = tf.layers.dense(h,units=1,activation=tf.sigmoid,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.loss = tf.losses.log_loss(labels=self.sampled_actions,
                                       predictions=self.up_probability,
                                       weights=self.advantage)
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        
    def train_network(self, states, actions, rewards):
        print('Training with %d tuples' % len(states))
        
        states = np.array(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        feed_dict = {self.observations: states, self.sampled_actions: actions, self.advantage: rewards}
        self.sess.run(self.train_op, feed_dict)
        
    def discount_rewards(self, rewards, discount_factor):
        discounted_rewards = np.zeros_like(rewards)
        for t in range(len(rewards)):
            discounted_reward_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                discounted_reward_sum += rewards[k] * discount
                discount *= discount_factor
                if rewards[k] != 0:
                    # Don't count rewards from subsequent rounds
                    break
            discounted_rewards[t] = discounted_reward_sum
        return discounted_rewards
    
    def predict_action(self, observations):
        up_probability = self.sess.run(self.up_probability,
            feed_dict={self.observations: observations[np.newaxis,:]})
        return up_probability

    def load_checkpoint(self):
        print('Loading checkpoint...')
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print('Saving checkpoint...')
        self.saver.save(self.sess, self.checkpoint_file)

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)
    
    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
        
    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = 'VALID')
    
    def prepro(self, o, image_size=[80,80]):
        """
        Call this function to preprocess RGB image to grayscale image if necessary
        This preprocessing code is from
            https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py

        Input: 
        RGB image: np.array
            RGB screen of game, shape: (210, 160, 3)
        Default return: np.array 
            Grayscale image, shape: (80, 80, 1)

        """
        y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
        y = y.astype(np.uint8)
        resized = scipy.misc.imresize(y, image_size)
        return np.expand_dims(resized.astype(np.float32),axis=2)
        