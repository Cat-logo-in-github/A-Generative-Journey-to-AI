import numpy as np
import matplotlib.pyplot as plt

# Assuming a SlotMachine class exists to be used with this game
class SlotMachine(object):
    def __init__(self, mean=0, stdev=1):
        self.mean = mean
        self.stdev = stdev
    
    def pull_lever(self):
        reward = np.random.normal(self.mean, self.stdev)
        return np.round(reward, 1)

class SlotMachineGame(object):
    def __init__(self, slot_machines):
        self.slot_machines = slot_machines
        np.random.shuffle(self.slot_machines)
        self.reset_game()

    def play(self, choice):
        # Assuming choice is 1-indexed based on the prompt
        reward = self.slot_machines[choice - 1].pull_lever()
        self.rewards.append(reward)
        self.total_reward += reward
        self.n_played += 1
        return reward

    def user_play(self):
        self.reset_game()
        print("Slot Machine Game started.\nEnter 0 or any non-machine number to end the game.")
        
        while True:
            print(f"\nRound {self.n_played + 1}")
            try:
                user_input = input(f"Select a machine from 1 to {len(self.slot_machines)}: ")
                choice = int(user_input)
                
                if 1 <= choice <= len(self.slot_machines):
                    reward = self.play(choice)
                    print(f"Machine {choice} gave a reward of {reward}.")
                    
                    avg_rew = self.total_reward / self.n_played
                    print(f"Your average reward so far is {avg_rew:.2f}.")
                else:
                    break
            except ValueError:
                break
                
        print("Slot Machine Game has ended.")

        if self.n_played > 0:
            # Fixed f-string and print syntax
            print(f"Total reward is {self.total_reward} after {self.n_played} round(s).")
            
            avg_rew = self.total_reward / self.n_played
            print(f"Average reward is {avg_rew:.2f}.")

            # Custom plot of rewards
            colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple', 'pink']
            plt.figure(figsize=(10, 6))

            for i, reward in enumerate(self.rewards):
                x_pos = i + 1
                # Fixed barh arguments and color indexing
                plt.barh(x_pos, reward, color=colors[i % len(colors)], edgecolor='black')
                plt.text(reward + 0.1, x_pos, str(reward), va='center', fontsize=10)

            plt.xlabel('Reward')
            plt.ylabel('Round of Play')
            plt.title('Distribution of Rewards in Slot Machine Game')
            
            # Setting y-ticks to match the number of rounds played
            plt.yticks(np.arange(1, len(self.rewards) + 1))
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

    def reset_game(self):
        self.rewards = []
        self.total_reward = 0
        self.n_played = 0

# Initializing the machines with diff mean and stdev
machineA = SlotMachine(8, 2)
machineB = SlotMachine(4, 3)
machineC = SlotMachine(6, 4)

#Initializing the game with the machines
game = SlotMachineGame([machineA, machineB, machineC])

# Play the game
if __name__ == "__main__":
    game.user_play()
