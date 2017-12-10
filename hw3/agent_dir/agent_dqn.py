import numpy as np
import pandas as pd
import tensorflow as tf

import random
from collections import deque
import pickle

from agent_dir.agent import Agent

# Hyperparameter settings for DQN
#REPLAY_SIZE = 1000000
#REPLAY_SIZE =  100000
NUM_EPISODES = 50000
MAX_NUM_STEPS = 10000
#UPDATE_TIME = 30000
OBSERVE = 50000 # timesteps to observe before training
EXPLORE = 1000000 # frames over which to anneal epsilon
BATCH_SIZE = 32

INITIAL_EPSILON = .9
FINAL_EPSILON = .1
#GAMMA = .99

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)

        # Configuration for nlg-workstation
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        
        # Parameter initialization
        self.env = env
        self.args = args
        np.random.seed(1239)
        random.seed(1239)
        tf.set_random_seed(1239)
        
        self.replay_size = args.replay_size
        self.update_time = args.update_time
        self.train_skip = args.skip
        self.gamma = args.gamma
        print('settings =', self.replay_size, self.update_time, self.train_skip, self.gamma)
        
        self.double_dqn = args.double_dqn
        
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape
        self.actions = env.action_space.n
        
        # Initialize replay memory
        # instance: (state,one_hot_action,reward,next_state,done)
        self.replay_memory = deque()
        
        self.reward_memory = deque(maxlen=30)
        self.reward_history = []

        # build more complicated DQN
        self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork_complex()
        self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork_complex()
        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]
        
        # build DQN (eval and target)
        '''
        self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()
        self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()
        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]
        '''
        print('Model bulit...')
        
        self.actionInput = tf.placeholder("float",[None,self.actions])
        self.yInput = tf.placeholder("float", [None]) 
        Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.optimizer = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.cost)
        
        self.saver = tf.train.Saver()
        if self.double_dqn:
            self.model_path='double_dqn_networks_%d_%d_%d_%.2f/' % (self.replay_size, self.update_time, self.train_skip, self.gamma)
        else:
            self.model_path='dqn_networks_%d_%d_%d_%.2f/' % (self.replay_size, self.update_time, self.train_skip, self.gamma)
        print('self.model_path =', self.model_path)
        
        if args.test_dqn:
            #you can load your model here
            print('Loading model parameters...')
            self.sess = tf.InteractiveSession(config=config)
            model_file=tf.train.latest_checkpoint(self.model_path)
            self.saver.restore(self.sess, model_file)
            print("Model restored...")
                
        else:
            self.sess = tf.InteractiveSession(config=config)
            self.sess.run(tf.global_variables_initializer())
            self.cost_history = []
         
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        for episode in range(NUM_EPISODES):
            state = self.env.reset()
            step_count = 0
            total_reward = 0
            
            for _ in range(MAX_NUM_STEPS):
                action = self.make_action(state, test=False)
                next_state, reward, done, _ = self.env.step(action)                
                #reward = -1 if done else reward
                
                self.perceive(state, action, reward, next_state, done)
                state = next_state
                
                step_count += 1
                total_reward += reward
                
                if done:
                    print('episode', episode, ': step %d/%d, reward = %d' % (step_count,self.time_step,total_reward))
                    self.reward_memory.append(total_reward)
                    if len(self.reward_memory) == 30:
                        self.reward_history.append(np.mean(np.array(self.reward_memory)))
                    break

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.actions)
        one_hot_action[action] = 1
        
        self.replay_memory.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.popleft()
        if len(self.replay_memory) > BATCH_SIZE and self.time_step % self.train_skip == 0:
            self.train_Q_network()
        
        self.time_step += 1
    
    def train_Q_network(self):
    
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]
        
        # Step 2: calculate y
        if self.double_dqn:
            # Double DQN
            y_batch = []
            QValue_batch = self.QValue.eval(feed_dict={self.stateInput:nextState_batch})
            QValueT_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})

            for i in range(BATCH_SIZE):
                done = minibatch[i][4]
                if done:
                    y_batch.append(reward_batch[i])
                else:
                    argmax_a = np.argmax(QValue_batch, axis=1)[i]
                    y_batch.append(reward_batch[i] + self.gamma * QValueT_batch[i][argmax_a])

            self.optimizer.run(feed_dict={self.yInput:y_batch,self.actionInput:action_batch,self.stateInput:state_batch})
            if self.time_step % 100 == 0:
                self.cost_history.append(self.cost.eval(feed_dict={self.yInput:y_batch,self.actionInput:action_batch,self.stateInput:state_batch}))
            
        else:
            # Natural DQN
            y_batch = []
            QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
            for i in range(BATCH_SIZE):
                done = minibatch[i][4]
                if done:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + self.gamma * np.max(QValue_batch[i]))

            self.optimizer.run(feed_dict={self.yInput:y_batch,self.actionInput:action_batch,self.stateInput:state_batch})
        
        # Saving parameters / records
        if self.time_step % 100 == 0:
            self.cost_history.append(self.cost.eval(feed_dict={self.yInput:y_batch,self.actionInput:action_batch,self.stateInput:state_batch}))
        
        if self.time_step % self.update_time == 0 and self.time_step > OBSERVE:
            self.saver.save(self.sess, self.model_path + 'network-dqn', global_step=self.time_step)
            print('Parameters saved..., time_step =', self.time_step)
            with open(self.model_path+'cost_history.pickle', 'wb') as f:
                pickle.dump(self.cost_history, f)
                print('Cost history saved...')
            with open(self.model_path+'reward_history.pickle', 'wb') as f:
                pickle.dump(self.reward_history, f)
                print('Reward history saved...')
            
        if self.time_step % self.update_time == 0 and self.time_step > OBSERVE:
            self.copyTargetQNetwork()
            print('Parameters updated..., time_step =', self.time_step)
            print('Current epsilon =', self.epsilon)
            print('Current loss =', self.cost_history[-1])
            
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        observation = observation.reshape((1,84,84,4))
        QValue = self.QValue.eval(feed_dict={self.stateInput:observation})[0]
        
        # epsilon-greedy 
        if random.random() <= self.epsilon and not test:
            action = random.randrange(self.actions)
        else:
            action = np.argmax(QValue)
        
        if test and random.random() < 0.01:
            action = random.randrange(self.actions)
        
        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            
        return action
        #return self.env.get_random_action()

    
    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8,8,4,16])
        b_conv1 = self.bias_variable([16])
        
        W_conv2 = self.weight_variable([4,4,16,32])
        b_conv2 = self.bias_variable([32])
        
        W_fc1 = self.weight_variable([2592,256])
        b_fc1 = self.bias_variable([256])
        
        W_fc2 = self.weight_variable([256,self.actions])
        b_fc2 = self.bias_variable([self.actions])
        
        # input layer
        stateInput = tf.placeholder("float",[None,84,84,4])
        
        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)
        
        h_conv2_flat = tf.reshape(h_conv2,[-1,2592])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat,W_fc1) + b_fc1)
        
        # Q Value layer
        QValue = tf.matmul(h_fc1,W_fc2) + b_fc2
        
        return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2
    
    
    def createQNetwork_complex(self):
        # network weights
        W_conv1 = self.weight_variable([8,8,4,32])
        b_conv1 = self.bias_variable([32])
        
        W_conv2 = self.weight_variable([4,4,32,64])
        b_conv2 = self.bias_variable([64])
        
        W_conv3 = self.weight_variable([3,3,64,64])
        b_conv3 = self.bias_variable([64])
        
        W_fc1 = self.weight_variable([3136,512])
        b_fc1 = self.bias_variable([512])
        
        W_fc2 = self.weight_variable([512,self.actions])
        b_fc2 = self.bias_variable([self.actions])
        
        # input layer
        stateInput = tf.placeholder("float",[None,84,84,4])
        
        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
        #h_pool1 = self.max_pool_2x2(h_conv1)
        
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)
        
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
        h_conv3_shape = h_conv3.get_shape().as_list()
        #print("dimension:",h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3])
        h_conv3_flat = tf.reshape(h_conv3,[-1,3136])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
        
        # Q Value layer
        QValue = tf.matmul(h_fc1,W_fc2) + b_fc2
        
        return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2
    
    def copyTargetQNetwork(self):
        self.sess.run(self.copyTargetQNetworkOperation)
        
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)
    
    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
        
    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")