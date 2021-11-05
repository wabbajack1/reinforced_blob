#!/usr/bin/env python3

import gym
from gym import spaces
import numpy as np
import time
import random
import matplotlib.pyplot as plt


class CustomEnv(gym.Env):
    """
    The behavior of the agent and the world which he interacts

    """

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 0, shape=(8,8), dtype=np.int)
        self.map = self.observation_space.sample()
        self.x = random.randint(0, 7) # x pos
        self.y = random.randint(0, 7) # y pos
        self.steps_counter_expire = 0

        self.map[self.x, self.y] = 1 # 1 == agent
        self.map[0, 0] = 8 # 8 == target
        
        
    def step(self, action):
        
        reward = -0.1
        done = False
        
        if self.decode_action(action) == "osten" and self.y < 7 and self.y >= 0: # rechts
            self.map[self.x, self.y + 1] = 1
            self.map[self.x, self.y] = 0
            
            if self.map[self.x, self.y + 1] == self.map[0, 0]:
                reward = 1
                done = True
            
            self.x, self.y = self.x, self.y + 1
                
        elif self.decode_action(action) == "westen" and self.y <= 7 and self.y > 0: # links
            self.map[self.x, self.y - 1] = 1
            self.map[self.x, self.y] = 0
            
            if self.map[self.x, self.y - 1] == self.map[0, 0]:
                reward = 1
                done = True
            
            self.x, self.y = self.x, self.y - 1
            
        elif self.decode_action(action) == "norden" and self.x <= 7 and self.x > 0: # oben
            self.map[self.x - 1, self.y] = 1
            self.map[self.x, self.y] = 0
            
            if self.map[self.x - 1, self.y] == self.map[0, 0]:
                reward = 1
                done = True
                
            self.x, self.y = self.x - 1, self.y
                
        elif self.decode_action(action) == "sueden" and self.x < 7 and self.x >= 0: # unten
            self.map[self.x + 1, self.y] = 1
            self.map[self.x, self.y] = 0
            
            if self.map[self.x + 1, self.y] == self.map[0, 0]:
                reward = 1
                done = True
            
            self.x, self.y = self.x + 1, self.y
            
        self.steps_counter_expire += 1

        observation = self.encode_state((self.x, self.y))
        
        return (observation, reward, done)
     
    def reset(self):
        # Write the reset method that results in the starting state
        self.x, self.y = random.randint(0, 7), random.randint(0, 7)
        self.steps_counter_expire = 0
        self.map = self.observation_space.sample()
        self.map[self.x, self.y] = 1 # 1 == agent
        self.map[0, 0] = 8 # 8 == target
        
        #return self.encode_state((self.x, self.y))
        
    def render(self):
        """
        The render method visualizes the state of the environment. There are many different ways to do so i.e. creating a visual representation by 
        using vector graphics or printing to the terminal.
        """

        #Write a render method for your enviroment to visualize the current state in the terminal
        print(self.map)
        pass
    
    def decode_action(self, action):
        #decode a linear action to 2D
        action_name = None
        
        if action == 0:
            action_name = "osten"
        
        if action == 1:
            action_name =  "westen"
        
        if action == 2:
            action_name = "norden"
        
        if action == 3:
            action_name = "sueden"
        
        return action_name

    def decode_state(self, state):
        #decode a linear state to 2D
        stateX, stateY = np.where(np.arange(0, env.map.shape[0]*env.map.shape[0]).reshape(8,8) == state) # abbildung des unique state tables
        return (int(stateX), int(stateY))
    
    def encode_state(self,state):
        #encode a 2D state in 1D
        x, y = state # 2D coords (states in 2D)
        encoded_state_table = np.arange(0, env.map.shape[0]*env.map.shape[0]).reshape(8,8) # abbildung des unique state tables
        return encoded_state_table[x, y]





########### Init algos and Q-Table ################

gamma = 0.99
eta = 0.01
epsilon = 0.1

def init_table():
    # your code here
    return np.zeros((64, env.action_space.n))

def eps_greedy(state):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
        #print("radn", action)
    else:
        action = np.argmax(table[state])
        #print("max", action, state) #debuginfo

    return action

def td_error(s, a, r, s_prime):
    # your code here
    delta = r + gamma * np.max(table[s_prime]) - table[s,a]
    return delta


def update_table(s, a, delta):
    # your code here
    table[s, a] += eta*delta




###################### TRAIN ######################

env = CustomEnv()
table = init_table()
episodes = 10000

def train():
    # your code here
    acc_reward_list = []
    
    for episode in range(episodes):
        acc_reward = 0
        observation = env.reset()
        done = False
        
        while not done:
            #print("Episode {}".format(episode+1))
            
            action = eps_greedy(observation)
            observation_new, reward, done = env.step(action)
            #action_new = eps_greedy(observation_new)
            
            error = td_error(observation, action, reward, observation_new) # only for the off-policy learners (agents)
            update_table(observation, action, error)
            
            observation = observation_new
            #action = action_new
            
            acc_reward += reward
            
            #env.render()
            #time.sleep(1)
            #print("state {} reward {} steps {}".format(state, reward, env.epoch_counter))
            
        
        acc_reward_list.append(acc_reward)
        
        if episode % 100 == 0:
            print("Episode: {}".format(episode))
    
    
    return acc_reward_list


def evaluate(policy):
    # your code here
    acc_reward_list = []
    count_score = 0
    avg_steps = 0
    epochs = 5
    
    for episode in range(epochs):
        acc_reward = 0
        observation = env.reset()
        done = False
        reward_lowest_points = 0
        
        while not done:
            #print("Episode {}".format(episode+1))
            
            action = policy(observation) # env.action_space.sample() --> please uncomment and test the untrained version
            observation, reward, done = env.step(action)
            acc_reward += reward
            
            if done:
                count_score += 1
                
            env.render()
            time.sleep(0.05)
        
        avg_steps += env.steps_counter_expire
        acc_reward_list.append(acc_reward)
        
    print(acc_reward_list)

    return acc_reward_list, "Accumulated reward: {} Avg_steps: {}".format(sum(acc_reward_list), avg_steps/epochs)


if __name__ == "__main__":
  
  train_rewards = train()
  
  
  evaluate(eps_greedy)
  

  plt.plot(train_rewards) # plot results of training
  t = []
  for i in table:
      t.append(np.max(i))

  u = plt.imshow(np.array(t).reshape(8,8))
  plt.colorbar(u) # hell == näher an target ; dunkel == unsicher Entscheidung (target entfernt)

  plt.show()
  

  pass
