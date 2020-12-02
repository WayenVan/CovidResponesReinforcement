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

def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()

def load_variable(filename):
    f = open(filename, 'rb')
    v = pickle.load(f)
    f.close()
    return v

#plot function
def plot_states(states: np.array, property=[0,1,2,3]):
    """plot states in one episode"""
    
    state_name = ["susceptible", "infectious", "quanrantined", "recovered"]
    states_swp = np.array(list(zip(*states)))

    for i in property:
        plt.plot(states_swp[i], label=state_name[i])
        
    plt.legend()


#plot rewards from stats
def plot_rewards(stats, num_average, problem_id):

    rewards = stats.episode_rewards.flatten()

    rewards_mean = np.resize(rewards, (len(rewards)//num_average, num_average))
    rewards_mean = np.mean(rewards_mean, axis=1)

    plt.title("average rewards every {} episodes".format(num_average))
    plt.plot(rewards_mean, label="problem_id={}".format(problem_id))
    plt.legend()