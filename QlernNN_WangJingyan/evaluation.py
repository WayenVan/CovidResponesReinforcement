import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle 


def policy_generator(env, approximator=None, policy_type="random"):
    """
    Creates an greedy policy with the exploration defined by the epsilon and nA parameters
    
    Input:
        epsilon: The probability to select a random action . float between 0 and 1.
        env: The environment
        policy_type: The type of policy, either random or greedy
    
    Output:
        A function that takes the observation as an argument and returns an action
    """
    def random_policy(s):
        action=np.random.choice(env.action_space.n)
        return action
    
    def greedy_policy(s):
        #make sure approximator has ben imported
        assert(not approximator is None)

        act_values = approximator.predict(np.reshape(s, [1, env.observation_space.shape[0]]))[0]
        action = np.argmax(act_values)  # returns action
        return action
    
    if(policy_type=="random"):
        return random_policy
    if(policy_type=="greedy"):
        return greedy_policy
    else:
        return None

def exec_agent(policy, env, max_steps=100):
    
    d_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    states = []
    rewards = []
    actions = []
    total_steps = 0    
    
    #reset the environment
    state = env.reset()
    states.append(state)
    
    for i in itertools.count():
        action = policy(state)
        
        #exec the policy
        state, reward, done, _= env.step(action)
        
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        
        if done or i >= max_steps:
            total_steps = i
            break
            
    return states, rewards, actions, total_steps


def plot_states(states, property=[0,1,2,3]):
    """plot states in one episode"""
    
    state_name = ["susceptible", "infectious", "quanrantined", "recovered"]
    states_swp = np.array(list(zip(*states)))

    for i in property:
        plt.plot(states_swp[i], label=state_name[i])
        
    plt.legend()
    

def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()

def load_variable(filename):
    f = open(filename, 'rb')
    v = pickle.load(f)
    f.close()
    return v
