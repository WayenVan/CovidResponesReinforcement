import numpy as np
import itertools
import random


def policy_generator(env, epsilon, approximator, policy_type="random"):
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
         # Select an action usign and epsilon greedy policy based on the main behavior network
        if np.random.rand() <= epsilon:
            action = random.randrange(env.action_space.n)
        else:
            act_values = approximator.predict(np.reshape(s, [1, env.observation_space.shape[0]]))[0]
            action = np.argmax(act_values)  # returns action
        return action
    
    if(policy_type=="random"):
        return random_policy
    if(policy_type=="greedy"):
        return greedy_policy
    else:
        return None

def exec_agent(policy, env):
    
    d_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    states = []
    rewards = []
    actions = []    
    
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
        
        if done:
            break
            
    return states, rewards, actions