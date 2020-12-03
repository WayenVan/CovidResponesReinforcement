
from collections import deque
from os import unsetenv
from matplotlib.pyplot import phase_spectrum

import numpy as np
import sys
import random
from collections import namedtuple

# Keep track of some stats
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

#this class provided for save stats using pickle
class Stats_storage():
    
    def __init__(self, stats):
        self.episode_lengths = stats.episode_lengths
        self.episode_rewards = stats.episode_rewards

#define a replay buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','is_not_terminal_state'))

class ReplayMemory():
    """
    Implement a replay buffer using the deque collection
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)               

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def pop(self):
        return self.memoery.pop()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)   

    def __len__(self):
        return len(self.memory)
#define a q_learning agent with


def q_learning_nn(env, func_approximator, func_approximator_target, num_episodes,
                use_normalization = False,
                max_steps_per_episode=500,discount_factor=0.95, 
                epsilon_init=0.01,epsilon_decay=0.99995,epsilon_min=0.01,
                use_batch_updates=True, BATCH_SIZE=20,
                fn_model_in=None, fn_model_out=None, BUFFER_SIZE=10000):
    """
    Q-Learning algorithm for Q-learning using Function Approximations.
    Finds the optimal greedy policy while following an explorative greedy policy.
    
    Args:
        env: OpenAI environment.bb
        func_approximator: Action-Value function estimator, behavior policy (i.e. the function which determines the next action)
        func_approximator_target: Action-Value function estimator, updated less frequenty than the behavior policy 
        num_episodes: Number of episodes to run for.
        max_steps_per_episode: Max number of steps per episodes
        discount_factor: Gamma discount factor.
        epsilon_init: Exploration strategy; chance the sample a random action. Float between 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
        epislon_min: Min epsilon value        
        use_batch_updates=True, 
        fn_model_in: Load the model from the file if not None
        fn_model_out: File name of the saved model, saves the best model in the last 100 episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
        
    memory = ReplayMemory(BUFFER_SIZE) # init the replay memory    
    n_actions = env.action_space.n        
    d_states  = env.observation_space.shape[0]    
    best_reward = 0

    # Synch the target and behavior network
    if not fn_model_in is None:
        func_approximator.model.load_weights(fn_model_in)
    func_approximator_target.model.set_weights(func_approximator.model.get_weights())

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))            

    epsilon = epsilon_init

    for i_episode in range(num_episodes):
        
        sys.stdout.flush()
               
        # Reset the environment and pick the first action
        state = env.reset()
        if(use_normalization):
            state = normalization(state)
        state = np.reshape(state, [1, d_states]) # reshape to the a 1xd_state numpy array
        
        # One step in the environment
        for t in range(max_steps_per_episode):#itertools.count():

            # Select an action usign and epsilon greedy policy based on the main behavior network
            if np.random.rand() <= epsilon:
                action = random.randrange(n_actions)
            else:
                act_values = func_approximator.predict(state)[0]
                action = np.argmax(act_values)  # returns action                        
            
            # Take a step            
            next_state, reward, done, _ = env.step(action) 
            if(use_normalization):
                next_state = normalization(next_state)                       
            next_state = np.reshape(next_state, [1, d_states] )
            

            # Add observation to the replay buffer
            if done:
                memory.push(state, action, next_state, reward, 0.0)            
            else:
                memory.push(state, action, next_state, reward, 1.0)            
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
                     
            # Update network (if learning is on, i.e. alpha>0 and we have enough samples in memory)
            if func_approximator.alpha > 0.0 and len(memory) >= BATCH_SIZE:                         
                # Fetch a bacth from the replay buffer and extract as numpy arrays 
                transitions = memory.sample(BATCH_SIZE)            
                batch = Transition(*zip(*transitions))                                
                train_rewards = np.array(batch.reward)
                train_states = np.array(batch.state)
                train_next_state = np.array(batch.next_state)
                train_actions = np.array(batch.action)
                train_is_not_terminal_state = np.array(batch.is_not_terminal_state) # 
                                
                if(use_batch_updates):
                    # Do a single gradient step computed based on the full batch
                    train_td_targets    = func_approximator.predict(train_states.reshape(BATCH_SIZE,4)) # predict current values for the given states
                    q_values_next       = func_approximator_target.predict(np.array(batch.next_state).reshape(BATCH_SIZE,d_states))                    
                    train_td_targetstmp = train_rewards + discount_factor * train_is_not_terminal_state * np.amax(q_values_next,axis=1)                
                    train_td_targets[ (np.arange(BATCH_SIZE), train_actions.reshape(BATCH_SIZE,).astype(int))] = train_td_targetstmp                                                                              
                    func_approximator.update(train_states.reshape(BATCH_SIZE,d_states), train_td_targets) # Update the function approximator using our target       
                else:
                    # Do update in a truely online sense where a gradient step is performaed per observation
                    for s in range(train_rewards.shape[0]):                        
                        target = func_approximator.predict(train_states[s])[0]
                        q_next = func_approximator_target.predict(train_next_state[s])[0]
                        target[train_actions[s]] = train_rewards[s] + discount_factor * train_is_not_terminal_state[s] * np.amax(q_next)                        
                        func_approximator.update(train_states[s], target.reshape(1,n_actions)) # Update the function approximator using our target                                            
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
            
            state = next_state                
            
            # 
            if done:
                # Synch the target and behavior network
                func_approximator_target.model.set_weights(func_approximator.model.get_weights())
              
                print("\repisode: {}/{}, score: {}, epsilon: {:.2}".format(i_episode, num_episodes, t, epsilon), end="")                               

                # Save the best model so far    
                if fn_model_out is not None and (t >= best_reward):
                    func_approximator.model.save_weights(fn_model_out)
                    best_reward = t
                
                break
            
    return stats

def normalization(state):
    return state/np.sum(state)