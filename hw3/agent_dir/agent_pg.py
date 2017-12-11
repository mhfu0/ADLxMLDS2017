import numpy as np
import pandas as pd
import tensorflow as tf

import os, sys
import random
from collections import deque
import pickle

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam

from agent_dir.agent import Agent

NUM_EPISODE = 30000

class Agent_PG(Agent):
    def __init__(self, env, args):

        super(Agent_PG,self).__init__(env)
        
        # Configuration for nlg-workstation
        from keras.backend.tensorflow_backend import set_session
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        set_session(tf.Session(config=config))

        # Parameter settings
        self.env = env
        self.args = args
        
        self.state_size = 80*80
        self.action_size = self.env.action_space.n
        self.gamma = 0.99
        self.learning_rate = 1e-4
        
        # Reproducibility
        random.seed(1239)
        np.random.seed(1239)
        tf.set_random_seed(1239)
        #self.env.seed(1239)
        
        # Memory for training
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        
        self.model_path = 'pg_network/'
        self.score_history = []
        
        # Build network
        self.model = self._build_model()
        self.model.summary()
        
        if args.test_pg:
            #you can load your model here
            print('Loading trained model')
            self.load_model(os.join.path(self.model_path,'pong.h5'))

    def train(self):
        state = self.env.reset()
        prev_x = None
        
        score = 0
        episode = 0
        
        while True:
            cur_x = self.preprocess_ob(state)
            x = cur_x - prev_x if prev_x is not None else np.zeros(self.state_size)
            prev_x = cur_x

            action, prob = self.predict(x)
            state, reward, done, _ = self.env.step(action)
            score += reward
            self.remember(x, action, prob, reward)

            if done:
                episode += 1
                loss = self.train_network()
                
                print('Episode: %d - Score: %d, loss: %f' % (episode, score, loss))
                self.score_history.append(score)
                
                score = 0
                state = self.env.reset()
                prev_x = None
                
                if episode > 1 and episode % 500 == 0:
                    self.save_model(os.path.join(self.model_path,'pong.h5'))
                    self.save_model(os.path.join(self.model_path,'pong_%d.h5' % episode))
                    with open(self.model_path+'score_history.pickle', 'wb') as f:
                        pickle.dump(self.score_history, f)
                    print('Model saved and score history dumped...')
            
            if episode > NUM_EPISODE:
                break
    
    def train_network(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        r = self.model.train_on_batch(X, Y)
        
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []
        
        return r
    
    def predict(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob
        
    def _build_model(self):
        model = Sequential()
        model.add(Reshape((80, 80, 1), input_shape=(self.state_size,)))
        model.add(Conv2D(16, kernel_size=(8,8), strides=(4,4), padding='valid',
                         activation='relu', init='he_uniform', data_format='channels_last'))
        model.add(Conv2D(32, kernel_size=(4,4), strides=(2,2), padding='valid',
                         activation='relu', init='he_uniform', data_format='channels_last'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', init='he_uniform'))
        #model.add(Dense(32, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model
    
    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)
        
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
        
    def preprocess_ob(self,I):
        I = I[35:195]
        I = I[::2, ::2, 0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(np.float).ravel()

    def init_game_setting(self):
        pass
    
    def make_action(self, observation, test=True):
        action, _ = self.predict(observation)    
        
    def load_model(self, path):
        self.model.load_weights(path)

    def save_model(self, path):
        self.model.save_weights(path)

