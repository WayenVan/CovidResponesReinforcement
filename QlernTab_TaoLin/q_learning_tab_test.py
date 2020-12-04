import numpy as np
import pandas as pd
from itertools import product

class TabQLearningAgent:
    """ 
    An exploratory Q-learning agent. [Figure 21.8]

    """
    def __init__(self, env, num_episodes, max_step_week, args):
        self.gamma = 0.90
        self.all_act = range(4)
        self.env = env
        self.num_episodes = num_episodes
        self.max_step_week = max_step_week
        self.args = args
        self.alpha = 0.5
        self.s = None
        self.a = None
        self.r = None

    def rl(self):
        # initialization variable
        env = self.env
        alpha, gamma = self.alpha, self.gamma
        num_episodes, max_step_week = self.num_episodes, self.max_step_week
        _args = self.args
        # all_act = self.all_act

        s, a, r = self.s, self.a, self.r
        
        q_dict = self.q_dictionary()
        # Q = self.q_table(q_dict, all_act)
        Q = pd.read_csv('./data/q_table/Q_table0{}_{}{}'.format(_args[0], _args[1], _args[2]), 
                        header=None)
        Q.drop(Q.columns[0], axis=1, inplace=True)

        # get action quantity
        n_action = env.action_space.n

        r_array = []
        a_array = np.zeros(max_step_week)

        # iteration of every episodes
        for i_episodes in range(num_episodes):
            s = env.reset()
            r_episodes = 0.0

            # iterations of every weeks
            for i_week in range(max_step_week):

                # format state into integer array
                _s = self.num_format(s)
                
                # find the q index
                s = self.find_index(q_dict, _s)[0]

                # Update for next iteration
                a = np.argmax(Q.loc[s,:].values + np.random.randn(n_action)*(1./(i_episodes+1)))

                a_array[i_week] = a

                # get next_state and rewards
                s1, r, done, _ = env.step(a)

                # calculate the rewards
                r_episodes += r

                # format state
                # _s1 = self.num_format(s1)

                # s1_index = self.find_index(q_dict, _s1)

                # q-learning
                # Q.loc[s, a] += alpha * (r + gamma * np.max(Q.loc[s1_index, :].values) - Q.loc[s, a])
            
                # Print out which step we're on, useful for debugging.
                print("\rProblem {}(stochastic={}, noisy={}): Step {} @ Episode {}/{} ({})".format(
                        _args[0], bool(_args[1]), bool(_args[2]),  i_week, i_episodes + 1, num_episodes, r), end="")

                # pop when done
                if done:
                    break

                # move to next state
                s = s1

            r_array.append(r_episodes)

            # Q.to_csv('./data/q_table/Q_table0{}_{}{}'.format(_args[0], _args[1], _args[2]))

        return r_array

    # create a q_table dictionary of the link between states and indexes
    def q_dictionary(self):
        count = 0
        dic = {}
        i = j = k = l = range(7)
        for comb in product(i, j, k, l):
            dic[count] = comb
            count += 1
        return dic

    # create a q_table based on q_dictionary
    def q_table(self, dic, act):
        table = pd.DataFrame(np.zeros((len(dic), len(act))),     # q_table initial values
                            columns=act,    # actions's name
                            )
        return table

    # format array into integer
    def num_format(self, array):
        for i in range(len(array)):
            array[i] = round(array[i] / 1e+08)
            if array[i] > 6:
                array[i] = 6
        new_array = array.astype(int)
        return new_array

    # index - states
    def find_index(self, dic, state):
        return [index for index, _state in dic.items() if (_state == state).all()]