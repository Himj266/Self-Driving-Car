# Brain of the Car
# Based on Neual Networks
# AI for self driving car

# Importing libraries
import numpy as np  # This is for working with arrays
import os   # This is to retrieve saved brain
import random
import torch
import torch.nn as nn   # Neural Network class of torch
import torch.nn.functional as F # Functional Class of torch
from torch.autograd import Variable # for making pytorch variables (tensors)

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)    # Dense Connection between input layer and hidden layer
        self.fc2 = nn.Linear(30, nb_action)     # Dense Connection between hidden layer and output layer
    
    def forward(self, state):
        x = F.relu(self.fc1(state))             # Hidden State : x
        q_values = self.fc2(x)                  # Q values
        return q_values

# Implementing Experience Replay
## Analyzing last 100 memories in batches to learn from the events that occured for less time in our training

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity    # This will check if our memory contains 100 transitions or not
        self.memory = []            # This will contain all 100 transitions

    def push(self, event):          # Push function appends last event to memory, event is a tupple of 4 elements containg S(t), S(t+1), a(t), Reward(t)
        self.memory.append(event)   # Appending transition/event to memory
        if len(self.memory) > self.capacity:    # Checking the memory for its capacity 
            del self.memory[0]                  # If the memory is full then we delete first memory that our car gathered

    def sample(self, batch_size):               # Sampling experiences
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x,0)), samples)