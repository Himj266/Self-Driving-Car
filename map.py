# Whole map
# Whole Game
# Game avoiding obstacles in game

# import deep q network from brain or ai
from ai import Dqn

brain = Dqn(5,3,0.9)    # arguements: inputs, nb_action, gamma(discount factor)
action2rotation = [0, 20, -20]  # All actions 
last_reward = 0     # Reward at this time
score = []      # Rewards onto the sliding windows