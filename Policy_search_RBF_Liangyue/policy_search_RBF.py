# The basics
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

# Let's import basic tools for defining the function and doing the gradient-based learning
import sklearn.pipeline
import sklearn.preprocessing
#from sklearn.preprocessing import PolynomialFeatures # you can try with polynomial basis if you want (It is difficult!)
from sklearn.linear_model import SGDRegressor # this defines the SGD function
from sklearn.kernel_approximation import RBFSampler # this is the RBF function transformation method

from scipy.linalg import norm, pinv


EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])



"""
Policy function
"""

def create_policy(func_approximator, epsilon, nA):
    
    def policy_fn(state):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = func_approximator.predict(state)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A,q_values  # return the potentially stochastic policy (which is due to the exploration)

    return policy_fn # return a handle to the function so we can call it in the future



"""
Execute the policy
"""
def exec_policy(env, func_approximator, verbose=False):

    # The policy is defined by our function approximator (of the utility)... let's get a hdnle to that function
    policy = create_policy(func_approximator, 0.0, env.action_space.n)
            
    # Reset the environment and pick the first action
    state = env.reset()
                    
    # One step in the environment
    for t in itertools.count():
        env.render()

        pi_action_state, q_values = policy(state)
        action = np.random.choice(np.arange(len(pi_action_state)), p=pi_action_state)
        #print("Action (%s): %s" % (action_probs,action)

        # Execute action and observe where we end up incl reward
        next_state, reward, done, _ = env.step(action)
        
        if verbose:
            print("Step %d/199:\n" % (t), end="")
            print("\t state     : %s\n" % (state), end="")            
            print("\t q_approx  : %s\n" % (q_values.T), end="")
            print("\t pi(a|s)   : %s\n" % (pi_action_state), end="")            
            print("\t action    : %s\n" % (action), end="")
            print("\t next_state: %s\n" % (next_state), end="")
            print("\t reward    : %s\n" % (reward), end="")                        
        else:
            print("\rStep {}".format(t), end="")
       
        if done:
            break
            
        state = next_state
        
        
        
"""
Function Approximation
"""

class FunctionApproximator():
 
    def __init__(self, env,scaler,feature_transformer, eta0= 0.01, learning_rate= "constant", read_approximator = None):
      
        self.eta0=eta0
        self.learning_rate=learning_rate
        self.env = env
        self.scaler = scaler
        self.feature_transformer = feature_transformer
        
        self.models = []
        
        if read_approximator is None:
             for _ in range(self.env.action_space.n):

                model = SGDRegressor(learning_rate=learning_rate, tol=1e-5, max_iter=1e5, eta0=eta0)
            
                model.partial_fit([self.featurize_state(self.env.reset())], [0])
                self.models.append(model)
                
        else:
            for d in read_approximator:
                f = open(d, 'rb')
                model = pickle.load(f)
                self.models.append(model)
                f.close()
            
    
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        s_scaled = self.scaler.transform([state])
        s_transformed = self.feature_transformer.transform(s_scaled)
        return s_transformed[0]
    
    def predict(self, s, a=None):

        features = self.featurize_state(s)
        if a==None:
            return np.array([m.predict([features])[0] for m in self.models])
        else:            
            return self.models[a].predict([features])[0]
    
    def save_models(self, path):
        for index in range(len(path)):
            f = open(path[index], 'wb')
            pickle.dump(self.models[index], f)
            f.close()
            
    def update(self, s, a, td_target):
        """
        Updates the approximator's parameters (i.e. the weights) for a given state and action towards
        the target y (which is the TD target).
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [td_target]) # recall that we have a seperate funciton for each a 
    
    def new_episode(self):        
        self.t_episode  = 0.  
        

        
        
"""
Reinforce learning
"""


def reinforce(env, func_approximator, num_episodes, use_training=True,  epsilon=0.015, discount_factor=1.0, epsilon_decay=1.0):

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        
        episode = []
        policy = create_policy(
            func_approximator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step                                   
            action_probs, q_vals = policy(state)
            
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
        
        if(use_training):
            # Go through the episode, step-by-step and make policy updates (note we sometime use j for the individual steps)
            func_approximator.new_episode()
            new_theta=[]
            for t, transition in enumerate(episode):                 
                # The return, G_t, after this timestep; this is the target for the PolicyEstimator
                G_t = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))

                # Update our policy estimator
                func_approximator.update(transition.state, transition.action,np.array(G_t))            
         
    return stats





class Save_stats():
    
    def __init__(self, stats_test):
        self.stats_test_length = stats_test.episode_lengths
        self.stats_test_reward = stats_test.episode_rewards
        
        




