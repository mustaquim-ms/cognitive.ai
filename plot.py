import matplotlib.pyplot as plt
import numpy as np

# --- Data for Plot 1 (Conceptual CartPole-v1) ---
episodes_plot1 = np.arange(200)
dqn_rewards_plot1 = np.clip(100 + 80 * (1 - np.exp(-episodes_plot1 / 30)), 0, 200) # Simulating DQN learning
snn_rewards_plot1 = np.clip(50 + 115 * (1 - np.exp(-episodes_plot1 / 50)), 0, 165) # Simulating SNN learning

# Create Plot 1
plt.figure(figsize=(10, 6))
plt.plot(episodes_plot1, dqn_rewards_plot1, label='DQN', color='blue')
plt.plot(episodes_plot1, snn_rewards_plot1, label='Cognitive SNN', color='red')
plt.xlabel('Training Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs. Training Episodes (CartPole-v1)')
plt.grid(True)
plt.legend()
plt.ylim(0, 210)
plt.show()

# --- Data for Plot 2 (Conceptual Grid World Adaptability) ---
episodes_plot2 = np.arange(200)
change_point = 100

# Simulating DQN adaptability
dqn_rewards_phase1 = np.clip(20 + 60 * (1 - np.exp(-np.arange(change_point) / 20)), 0, 80)
dqn_rewards_phase2 = np.clip(dqn_rewards_phase1[-1] * np.exp(-(np.arange(len(episodes_plot2) - change_point)) / 40), 0, 80) # Drop and slow recovery
dqn_rewards_plot2 = np.concatenate((dqn_rewards_phase1, dqn_rewards_phase2))

# Simulating SNN adaptability
snn_rewards_phase1 = np.clip(15 + 55 * (1 - np.exp(-np.arange(change_point) / 15)), 0, 70)
snn_rewards_phase2 = np.clip(snn_rewards_phase1[-1] * np.exp(-(np.arange(len(episodes_plot2) - change_point)) / 20) + 50 * (1 - np.exp(-(np.arange(len(episodes_plot2) - change_point)) / 20)), 0, 75) # Drop and faster recovery
snn_rewards_plot2 = np.concatenate((snn_rewards_phase1, snn_rewards_phase2))

# Create Plot 2
plt.figure(figsize=(10, 6))
plt.plot(episodes_plot2, dqn_rewards_plot2, label='DQN', color='blue')
plt.plot(episodes_plot2, snn_rewards_plot2, label='Cognitive SNN', color='red')
plt.axvline(change_point, color='gray', linestyle='--', label='Environment Change')
plt.xlabel('Training Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs. Training Episodes (Grid World - Adaptability)')
plt.grid(True)
plt.legend()
plt.ylim(0, 90)
plt.show()
