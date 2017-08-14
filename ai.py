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
import torch.optim as optim
import torch.autograd as autograd
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
        self.capacity = capacity    # This will check if our memory contains 100000 transitions or not
        self.memory = []            # This will contain all 100000 transitions

    def push(self, event):          # Push function appends last event to memory, event is a tupple of 4 elements containg S(t), S(t+1), a(t), Reward(t)
        self.memory.append(event)   # Appending transition/event to memory
        if len(self.memory) > self.capacity:    # Checking the memory for its capacity 
            del self.memory[0]                  # If the memory is full then we delete first memory that our car gathered

    def sample(self, batch_size):               # Sampling experiences
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x,0)), samples)

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = 0
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Using Adam opimizer
        self.last_state = torch.Tensor(input_size).unsqueeze(0)         # Making a tensor of state
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self, self.model(Variable(state))*100)  # Converting state tensor to Variable for fast computation but we have to declare Volatile=True to specify that we dont want to compute its gradients and dont want to have it in backpropagation operation.
        action = probs.multinomial()
        return action.data[0,0]


