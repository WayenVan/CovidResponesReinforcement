import virl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class SARSA:
    '''
    Agent class for poliyc search method Sarsa, which is a tabular based reinforcement learning metho
    '''
    def __init__(self, actions, lr=0.01, reward_decay_factor=0.95, eps=0.8):
        self.actions = actions  # 动作空间 action space
        self.lr = lr # 学习率 learning rate
        self.gamma = reward_decay_factor # 奖励折扣系数 reward factor
        self.epsilon = eps # 初始eps值，用于e-greedy探索方法
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) # 存储Q表 storing Q table

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def select_action(self, observation):
        # 检查是否该state是否存在，不存在就插入
        self.check_state_exist(observation)
 
        if np.random.uniform() >= self.epsilon:
            # 选取Q值最大的动作
            state_action_values = self.q_table.loc[observation, :]
            action = np.random.choice(state_action_values[state_action_values == np.max(state_action_values)].index) # handle multiple argmax with random
        else:
            # 随机选取动作
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)

        if s_ != 'done':
            a_ = self.select_action(str(s_)) # argmax action
            q_target = r + self.gamma * self.q_table.loc[s_, a_] # max state-action value
        else:
            q_target = r  # next state is terminal

        self.q_table.loc[s, a] = self.q_table.loc[s, a] + self.lr * (q_target - self.q_table.loc[s, a])

        self.epsilon = max(0.2, self.epsilon*0.995) # 更新 epsilon的值，希望训练得到后期随机选动作的概率变小

        return s_, a_

    # 动态添加Q表记录
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 如果当前state不在Q表中，插入该记录 insert the record if the current state is not in Q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


# 原本的观测状态空间过大，因此做一个状态空间压缩，先算出每种人数的所占比例，然后放大100倍。使得每个state都在[0,100]范围内
# The original massive observation space make the problem unsolvable with tabular method.
def discretized(state, N):
    new_state = state/N*100 # 按比例压缩
    new_state = np.round(new_state).astype(int)
    return new_state

# 训练函数
def train(env, agent, epochs=10):
    for epoch in range(epochs):
        s = env.reset() # 重置环境 开始训练 reset the env to start new epoch of training
        done = False
        while not done:
          # action = np.random.choice(env.action_space.n)
            d_s = discretized(s, env.N) # 获得离散后的观测状态 get the processed state
            action = agent.select_action(str(d_s)) # 选取当前动作
            # print(action)
            s_next, reward, done, i = env.step(action=action) # 执行动作 execute an action

            d_s_next = discretized(s_next, env.N) # 获得离散后的观测状态
            action_next = agent.select_action(str(d_s_next)) # 根据当前策略预测下一个状态的动作

            if done:
                d_s_next = 'done'
            agent.learn(str(d_s), action, reward, str(d_s_next), action_next) # 学习经验

            s = s_next
            action = action_next


# 评估函数
def evaluation(env, agent):
    states = [] # 存储状态
    rewards = [] # 存储奖励值

    s = env.reset() # 重置环境
    while True:
        d_s = discretized(s, env.N) # 获得离散后的观测状态
        action = agent.select_action(str(d_s)) # 选取当前动作

        s, reward, done, i = env.step(action=action) # 执行动作

        states.append(s)
        rewards.append(reward)

        if done:
            break

    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
    states = np.array(states)
    for i in range(4):
        axes[0].plot(states[:,i], label=labels[i]);
    axes[0].set_xlabel('weeks since start of epidemic')
    axes[0].set_ylabel('State s(t)')
    axes[0].legend()
    axes[1].plot(rewards);
    axes[1].set_title('Reward')
    axes[1].set_xlabel('weeks since start of epidemic')
    axes[1].set_ylabel('reward r(t)')

    plt.savefig(dpi=300, fname='reward.png')

    print('total reward for evaluation', np.sum(rewards))
    return np.sum(rewards)


def evaluation_all():
    fig, ax = plt.subplots(figsize=(12, 6))
    for index in range(10):
        env = virl.Epidemic(problem_id=index, stochastic=False, noisy=False)
        states = []
        rewards = []
        agent = SARSA(actions=list(range(env.action_space.n)))
        print('pid {} start'.format(index))
        for epoch in range(400):
          done = False
          s = env.reset()
          # states.append(s)
          total_rewards = 0
          while not done:
              # s, r, done, info = env.step(action=0) # deterministic agent
              d_s = discretized(s, env.N) # 获得离散后的观测状态 get the processed state
              action = agent.select_action(str(d_s)) # 选取当前动作
              s_next, reward, done, i = env.step(action=action)
              d_s_next = discretized(s_next, env.N) # 获得离散后的观测状态
              action_next = agent.select_action(str(d_s_next)) # 根据当前策略预测下一个状态的动作
              if done:
                d_s_next = 'done'
              agent.learn(str(d_s), action, reward, str(d_s_next), action_next) # 学习经验

              s = s_next
              action = action_next
              total_rewards += reward
              # states.append(s)
              # rewards.append(r)
          rewards.append(total_rewards)
        # print(rewards)
        ax.plot(np.array(rewards)[:], label=f'problem_id={index}')
    ax.set_xlabel('training episodes')
    ax.set_ylabel('rewards')
    # ax.set_title('Simulation of problem_ids without intervention')
    ax.legend(loc = 4)
    plt.savefig(dpi=300, fname='problems_reward.png')


if __name__ == '__main__':
    # P_ID = 0
    # env = virl.Epidemic(problem_id=P_ID, stochastic=False, noisy=False) # 创建环境
    # agent = SARSA(actions=list(range(env.action_space.n)))
    # train(env, agent, 10)
    # evaluation(env, agent)
    evaluation_all()