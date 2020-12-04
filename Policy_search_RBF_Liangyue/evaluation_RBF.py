import matplotlib.pyplot as plt
import pandas as pd


class Plot_stats():
    
    def __init__(self, stats_test):
        self.stats_test = stats_test
    
    
    def plot_stats_test(self, smoothing_window = 1):
        # Plot the episode reward over time
    
        fig1 = plt.figure(figsize=(10,5))
        rewards_test = pd.Series(self.stats_test.stats_test_reward).rolling(smoothing_window, min_periods=smoothing_window).mean()

        plt.plot(rewards_test)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
        plt.grid(True)
    

