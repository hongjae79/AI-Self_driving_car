# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module): #nn here is neural network
    
    def __init__(self, input_size, nb_action):
        #input_size is 5.  the input vector. these are enought to describe the state
        super(Network, self).__init__()
        #Network is class. self is object  this is to use all tools in nn.module
        self.input_size = input_size
        self.nb_action = nb_action
        #nb_action is 3.  3 possible move
        self.fc1 = nn.Linear(input_size, 30) #full connections
        #nn.Linear gives fully connected.  
        #There are 30 neurons. 
        #30 comes up after trail and error.  
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state): #state is input of neural network.  remember state is size 5 vector. 
        #activate hidden neurons
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity  #100000 trainsition here.  you can increase this value
        self.memory = []
    
    def push(self, event): #keep it 100000
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        #if list = [[1,2,3], [4,5,6]], 
        #then zip(*list) = [(1,4), (2,5), (3,6)] 
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning, deep Q network

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma): #gamma is delay parameter
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        #softmax better than argmax.  dlef.model here is Network, which is q-value
        #state is torch tensor. we convert it to torch variable
        #T is temperature parameter.  closer to 0 nn is unsure.  larger T gives nn is sure. 
        action = probs.multinomial()
        #multinomial function gives random draw from action
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #outputs is Q(a, s_t_B )        
        #self.model is from init.  so it would be Network
        #we are only interested in action.  so gather 1
        #unsaueeze 0 is fake dimension state, 1 is fake dimension action
        #kill the fake batch with a squeeze.  to get back simple form
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        
        # next_outputs is Q(a, s_t_{ B +1 } )
        # max(1)[0],  1 is take max respect to action.  0 is state s_t_{B + 1} next state
        target = self.gamma*next_outputs + batch_reward
        #target is r_t_B + gamma * max ( Q(a, s_t_{B + 1} )) 
        td_loss = F.smooth_l1_loss(outputs, target)
        #TD is r_t + gamma * max (Q(a, s_{t+1} )) - Q(a_t, s_t) 
        #Loss is 1/2 * TD_t(a_t , s_t)^2
        self.optimizer.zero_grad()
        #apply optimizer Stochastic Gradient Descent
        #zero_grad will reinitialize the optimizer at each iteration of the loop
        td_loss.backward(retain_variables = True)
        #back propagation.  backward will backpropagate errors into the nn.  
        self.optimizer.step()
        #this will update weights
    
    def update(self, reward, new_signal):
    #this def makes connection between ai and map
    #reward here is last_reward in map.py, and new_signal here is last_signal in map.py
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #note that new_signal is 5 dimension vector: singal 1,2,3, and +- orientation. 
        #This is a simply list, so we change to tensor.  make sure it is a float
        #create fake dimenstion by unsqueeze and index 0
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), 
                          torch.Tensor([self.last_reward])))
        #update memory.  
        #push function will update new transition.  
        # last_state is s_t
        # new_state is s_{t+1}
        # last_action a_t is action either 0,1, or 2.  Long is a type that contains integer. 
        # LongTensor will change last_action to tensor object. 
        # last reward is r_t
        # last element of transition is last_reward. and we will tensor it.  
        action = self.select_action(new_state)
        #take action
        if len(self.memory.memory) > 100:
            #first memory is object  or the replay, second memory is attribute
            # if memory is more than 100 then we can learn
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            #now learn from 100 transition.  note learn function is from DQN class
        #ai is learning from 100 random sample transition
        self.last_action = action
        #update last_action
        self.last_state = new_state
        #update last_state
        self.last_reward = reward
        #update last_reward
        self.reward_window.append(reward)
        #reward_window is from DQN init.  it is for keeping track of how this train is going 
        #by taking the average of last 100 rewards. we need to add last_reward to reward list
        
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        #we need to keep the window size fixed.  we delete the first element of the window if 
        # the size is more than 1000. 
        #1000 means of last 100 rewards
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict()}, 
                    'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")