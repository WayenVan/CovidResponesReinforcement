import matplotlib.pyplot as plt
import time
import itertools
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

import numpy as np
import sys
import os
import random
from collections import namedtuple
import collections
import copy

# Import the open AI gym
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
print(sys.version)



#import environment
import sys
sys.path.append(r'../virl')
import virl

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

class Stats_storage():
    def __init__(self, stats):
        self.episode_rewards=stats.episode_rewards
        self.episode_length = stats.episode_lengths


def reinforce(env, estimator_policy, num_episodes, tabular, discount_factor=1.0):
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
        
        episode = []
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            state_oh = one_hot(state, tabular)
            
            action_probs = estimator_policy.predict(state_oh)
            action_probs = action_probs.squeeze()
            
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            ##
            next_state, reward, done, _ = env.step(action)
            
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
            
            state_oh = one_hot(state, tabular)
            
            # The return, G_t, after this timestep; this is the target for the PolicyEstimator
            G_t = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
           
            # Update our policy estimator
            estimator_policy.update(state_oh, transition.action,np.array(G_t))            
         
    return stats

def one_hot(state, tabular):
    state = state/1.0e8
    split = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    state_ds = np.digitize(state, split)
    state_ds = tuple(state_ds)
    state_oh = np.zeros((1, len(tabular)))
   
    state_oh[0,tabular[state_ds]] = 1.0
    return state_oh