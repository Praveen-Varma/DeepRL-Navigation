# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:48:29 2018

@author: r.varma.kucherlapati
"""
import torch
import random
import torch.nn as nn
device = torch.device('cpu')
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.action_size = action_size
        self.layers = nn.Sequential( 
                                    nn.Linear(state_size, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128), 
                                    nn.ReLU(),
                                    nn.Linear(128, action_size)
        )
    def forward(self, input):
        print('x')
        return self.layers(input)
    def act(self, state, eps):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        if random.random() < eps:
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.forward(state)
            action = q_values.max(1)[1].item()
        return action
                