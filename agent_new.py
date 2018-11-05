# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:00:15 2018

@author: r.varma.kucherlapati
"""
import math
import torch
import numpy as np
import torch.optim as optim
from memory import Memory
from model_new import DQN

device = torch.device('cpu')

TAU = 0.4
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2000
LR = 5e-4
BATCH_SIZE = 64
MEMORY_SIZE = 50000
UPDATE_EVERY = 4
ALPHA = 0.6

class Agent():
    
    def __init__(self, state_size, action_size):
        self.memory = Memory(MEMORY_SIZE, ALPHA)
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.t_step     = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.t_step += 1
        upd_step = (self.t_step) % UPDATE_EVERY
        if upd_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences, indices, weights = self.memory.sample(BATCH_SIZE, self.beta_func)
                self.learn(experiences, weights, indices, GAMMA)
    @property                
    def eps_func(self):
        return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.t_step / EPS_DECAY)
    @property
    def beta_func(self):
        beta_start = 0.4
        beta_frames = 5000 
        return min(1.0, beta_start + self.t_step * (1.0 - beta_start) / beta_frames)
    def act(self, state):
        return self.policy_net.act(state, self.eps_func)
    def learn(self, experiences, weights, indices, gamma):
        states, actions, rewards, next_states, dones = experiences
        states = torch.from_numpy(states).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        actions = torch.from_numpy(np.array(actions)).long().to(device).unsqueeze(1)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device).unsqueeze(1)
        dones = torch.from_numpy(np.vstack([d for d in dones]).astype(np.uint8)).float().to(device)
        weights = torch.tensor(weights).float().to(device)
        action = self.policy_net(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.target_net(next_states).detach().gather(1, action)
       # Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.policy_net(states).gather(1, actions)
        #loss = F.mse_loss(Q_expected, Q_targets)
        loss = (Q_expected.squeeze(1) - Q_targets.squeeze(1)).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.policy_net, self.target_net, TAU)                    
        self.memory.update_priorities(indices, prios)
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)