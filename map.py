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
    global sand     # will contain info about sand position
    global goal_x   # x-coordinate
    global goal_y   # y-doordinate
    global first_update # it remembers wheather its first initialization or not
    sand = np.zeros((longueur, largeur))    # initialized sand to zero depicting that there is no sand at the start
    goal_x = 20     # x-coordinate of destination
    goal_y = largeur - 20   # y-coordinate of destination
    first_update = False    # depicting that initialization is done
    