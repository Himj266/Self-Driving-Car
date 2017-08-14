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
    
    # Select action of direction of movement of car
    def select_action(self, state):
        # Converting state tensor to Variable for fast computation but we have to declare Volatile=True to specify that we dont want to compute its gradients and dont want to have it in backpropagation operation.
        probs = F.softmax(self, self.model(Variable(state))*100)  # Temperature coff = 100 so that we can have better look at each option.(At Temperature coff -> infinity the probablity for each action tend to equalize and if temperature coff -> 0 probability for each action tend to shift towards one on the action)
        action = probs.multinomial()
        return action.data[0,0]

    # learning from experience 
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)   # getting q values
        next_outputs = self.model(batch_next_state).detach().max(1)[0]                      # getting q value of next state
        target = batch_reward + self.gamma*next_outputs                                     # calculating theoritical q value i.e q(s(t)) = r(s,a) + gamma*max(q(a,s(t+1))) for every action possible
        td_loss = F.smooth_l1_loss(outputs, target)                                         # Loss = actual - predicted
        self.opimizer.zero_grad()                                                           # Reinitializing the optimizer at each iteration of loop
        td_loss.backward(retain_variables=True)                                             # backpropagation
        self.opimizer.step()                                                                # Updating weights

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)   # new state 
        self.memory.push((last_state, new_state, torch.LongTensor([int(self.last_action)], torch.Tensor(self.last_reward))))    # Updating memory after reaching new state
        action = self.select_action(new_state)          # action selected based on new state
        if len(self.memory.memory) > 100:               # if we have more than 100 memories do sampling
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
        self.last_state = new_state                     # update last_state, last_action, last_reward with new values
        self.last_action = action
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:              # if our reward window has more than 1000 values than delete the first value in it
            del self.reward_window[0]
        return action                                   # return action as required

    def score(self):                                    # Mean of all rewards in the reward window
        return sum(self.reward_window)/(len(self.reward_window)+1.)  # adding 1 to denominator to avoid infinity problem
        
    def save(self):                                     # saving our brain in last_brain.pth file
        torch.save({
            "state_dict":self.model.state_dict(),
            "optimizer" :self.optimizer.state_dict()
        }, 'last_brain.pth')

    