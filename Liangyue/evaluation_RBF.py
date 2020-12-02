import matplotlib.pyplot as plt
import pandas as pd


class Plot_stats():
    
    def __init__(self, stats_test_FF):
        self.stats_test_FF = stats_test_FF
    
    
    def plot_stats_test(self, smoothing_window = 1):
        # Plot the episode reward over time
    
        fig1 = plt.figure(figsize=(10,5))
        rewards_smoothed = pd.Series(self.stats_test_FF.stats_test_reward).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (Smoothed)")
        plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
        plt.grid(True)
    

