# Whole map
# Whole Game
# Game avoiding obstacles in game

# Importing required libraries
import numpy as np

# import deep q network from brain or ai
from ai import Dqn

# Brain of Self driving car initializing, its a neural network
brain = Dqn(5,3,0.9)    # arguements: inputs, nb_action, gamma(discount factor)
action2rotation = [0, 20, -20]  # All actions 
last_reward = 0     # Reward at this time
score = []      # Rewards onto the sliding windows

# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros