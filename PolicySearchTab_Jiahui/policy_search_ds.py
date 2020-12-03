import matplotlib.pyplot as plt
import time
import itertools
import matplotlib
import pickle
import pandas as pd

import numpy as np
import sys
import os
import collections
from collections import namedtuple

import gym

import keras
from keras.models import Sequential
#from keras.layers import Dense
from keras.optimizers import Adam
#from keras import backend as K

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from keras import initializers

import tensorflow as tf

from scipy.linalg import norm, pinv

#import environment
import sys
sys.path.append(r'../virl')
import virl

#Approximation
class NeuralNetworkPolicyEstimator():
    """ 
    A very basic MLP neural network approximator and estimator for poliy search    
    
    The only tricky thing is the traning/loss function and the specific neural network arch
    """
    
    def __init__(self, alpha, n_actions, d_states, nn_config, verbose=False):        
        self.alpha = alpha    
        self.nn_config = nn_config                   
        self.n_actions = n_actions        
        self.d_states = d_states
        self.verbose=verbose # Print debug information        
        self.n_layers = len(nn_config)  # number of hidden layers        
        self.model = []
        self.__build_network(d_states, n_actions, nn_config)
        self.__build_train_fn()
             

    def __build_network(self, input_dim, output_dim, nn_config):
        """Create a base network usig the Keras functional API"""
        self.X = layers.Input(shape=(input_dim,))
        net = self.X
        
        for h_dim in nn_config:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("relu")(net)
            
        net = layers.Dense(output_dim, kernel_initializer=initializers.Zeros())(net)
        net = layers.Activation("softmax")(net)
        self.model = Model(inputs=self.X, outputs=net)

    def __build_train_fn(self):
        """ Create a custom train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.        
        Called using self.train_fn([state, action_one_hot, target])`
        which would train the model. 
        
        Hint: you can think of K. as np.
        
        """
        # predefine a few variables
        action_onehot_placeholder   = K.placeholder(shape=(None, self.n_actions),name="action_onehot") # define a variable
        target                      = K.placeholder(shape=(None,), name="target") # define a variable       
        
        # this part defines the loss and is very important!
        action_prob        = self.model.output # the outlout of the neural network        
        action_selected_prob        = K.sum(action_prob * action_onehot_placeholder, axis=1) # probability of the selcted action        
        log_action_prob             = K.log(action_selected_prob) # take the log
        loss = -log_action_prob * target # the loss we are trying to minimise
        loss = K.mean(loss)
        
        # defining the speific optimizer to use
        adam = optimizers.Adam(lr=self.alpha)# clipnorm=1000.0) # let's use a kereas optimiser called Adam
        updates = adam.get_updates(params=self.model.trainable_weights,loss=loss) # what gradient updates to we parse to Adam
            
        # create a handle to the optimiser function    
        self.train_fn = K.function(inputs=[self.model.input,action_onehot_placeholder,target],
                                   outputs=[],
                                   updates=updates) # return a function which, when called, takes a gradient step
      
    
    def predict(self, s, a=None):              
        if a==None:            
            return self._predict_nn(s)
        else:                        
            return self._predict_nn(s)[a]
        
    def _predict_nn(self,state_hat):                          
        """
        Implements a basic MLP with tanh activations except for the final layer (linear)               
        """                
        x = self.model.predict(state_hat)                                                    
        return x
  
    def update(self, states, actions, target):  
        """
            states: a interger number repsenting the discrete state
            actions: a interger number repsenting the discrete action
            target: a real number representing the discount furture reward, reward to go
            
        """
        action_onehot = np_utils.to_categorical(actions, num_classes=self.n_actions) # encodes the state as one-hot
        self.train_fn([states, action_onehot, target]) # call the custom optimiser which takes a gradient step
        return 
        
    def new_episode(self):        
        self.t_episode  = 0.
 

def discrete_sample(state):
    return state


"""
Reinforce learning
"""

def reinforce(env, estimator_policy, num_episodes, discount_factor=1.0, use_discrete=False):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized         
        num_episodes: Number of episodes to run for
        discount_factor: reward discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    
    Adapted from: https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb
    """

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        if(use_discrete):
            state = discrete_sample(state)
        episode = []
        
        # One step in the environment
        for t in itertools.count():
            # Take a step                       
            action_probs = estimator_policy.predict(state_)
            action_probs = action_probs.squeeze()
            
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            ##
            next_state, reward, done, _ = env.step(action)
            if(use_discrete):
                next_state = discrete_sample(next_state)
                
            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")            

            if done:
                break
                
            state = next_state
    
        # Go through the episode, step-by-step and make policy updates (note we sometime use j for the individual steps)
        estimator_policy.new_episode()
        new_theta=[]
        for t, transition in enumerate(episode):                 
    
            # The return, G_t, after this timestep; this is the target for the PolicyEstimator
            G_t = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
           
            # Update our policy estimator
            estimator_policy.update(transition.state, transition.action,np.array(G_t))            
         
    return stats