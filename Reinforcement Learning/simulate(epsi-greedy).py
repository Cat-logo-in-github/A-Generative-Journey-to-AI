import numpy as np
import matplotlib.pyplot as plt
from environment import SlotMachine, SlotMachineGame


# -----------------------------
# ε-greedy agent
# -----------------------------
class EpsilonGreedyAgent:
    def __init__(self, n_actions, epsilon=0.1, alpha=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_values = np.zeros(n_actions)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_values)

    def update(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])


# -----------------------------
# Training function
# -----------------------------
def train_agent(agent_id, episodes=500):
    # fresh environment per agent
    machineA = SlotMachine(8, 2)
    machineB = SlotMachine(4, 3)
    machineC = SlotMachine(6, 4)

    env = SlotMachineGame([machineA, machineB, machineC])

    agent = EpsilonGreedyAgent(n_actions=3, epsilon=0.1, alpha=0.1)

    avg_rewards = []
    total_reward = 0

    for episode in range(episodes):
        action = agent.choose_action()
        reward = env.play(action + 1)

        agent.update(action, reward)

        total_reward += reward
        avg_rewards.append(total_reward / (episode + 1))

    return avg_rewards


# -----------------------------
# Run multiple agents
# -----------------------------
if __name__ == "__main__":

    num_agents = 10
    episodes = 500

    all_curves = []

    for i in range(num_agents):
        curve = train_agent(i, episodes)
        all_curves.append(curve)

    # -----------------------------
    # Plot all agents
    # -----------------------------
    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    for i in range(num_agents):
        plt.plot(all_curves[i], color=colors[i], alpha=0.8, label=f"Agent {i+1}")

    plt.title("Learning Curves of 10 ε-Greedy Agents")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)

    plt.show()