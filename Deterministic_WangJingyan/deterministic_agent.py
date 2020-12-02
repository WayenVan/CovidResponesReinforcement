from QlernNN_WangJingyan.q_learning_nn import *
import numpy as np
import sys

def deterministic_agent(env, num_episodes, max_steps_per_episode = 500):

    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)) 
    
    for i_episode in range(num_episodes):
        sys.stdout.flush()
        state = env.reset()

        for t in range(max_steps_per_episode):
            next_state, reward, done, _ = env.step(action=0) #no interval at all!

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t+1

            if done :
                print("\repisode:{}/{} score:{}".format(i_episode, num_episodes, t+1), end="")
                break
            
            state = next_state 

    return stats