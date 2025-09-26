"""
Abhiram Dodda
"""

import numpy as np
import random
import matplotlib.pyplot as plt


#board setup
BOARD_ROWS = 5
BOARD_COLS = 5
#states
START = (0, 0)
WIN_STATE = (4, 4)
HOLE_STATE = [(1,0),(3,1),(4,2),(1,3)]
        
class State:
    def __init__(self, state=START):
        self.state = state
        self.isEnd = False        

    def getReward(self):
        #rewards for each state -5 for loss, +1 for win, -1 for others
        for i in HOLE_STATE:
            if self.state == i:
                return -5
        if self.state == WIN_STATE:
            return 10       
        
        else:
            return -1

    def isEndFunc(self):
        if (self.state == WIN_STATE):
            self.isEnd = True
            
        for i in HOLE_STATE:
            if self.state == i:
                self.isEnd = True

    def nxtPosition(self, action):     
        if action == 0:                
            nxtState = (self.state[0] - 1, self.state[1])             
        elif action == 1:
            nxtState = (self.state[0] + 1, self.state[1])
        elif action == 2:
            nxtState = (self.state[0], self.state[1] - 1)
        else:
            nxtState = (self.state[0], self.state[1] + 1) 

        if (nxtState[0] >= 0) and (nxtState[0] <= 4):
            if (nxtState[1] >= 0) and (nxtState[1] <= 4):                  
                    return nxtState    
        return self.state 
    
#implement reinforcement learning through grid  
class Agent:
    def __init__(self):
        #states and actions 
        self.states = []
        self.actions = [0,1,2,3]  
        self.State = State()
        #learning and greedy values
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.isEnd = self.State.isEnd
        self.plot_reward = []
        #Q values as a dictionary for current and new
        self.Q = {}
        self.new_Q = {}
        self.rewards = 0

        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for k in range(len(self.actions)):
                    self.Q[(i, j, k)] =0
                    self.new_Q[(i, j, k)] = 0
        
        print(self.Q)
        
    
    def Action(self):
        rnd = random.random()
        mx_nxt_reward =-10
        action = None

        if(rnd >self.epsilon) :
            #iterate through actions, find Q  value and choose best 
            for k in self.actions:
                i,j = self.State.state
                nxt_reward = self.Q[(i,j, k)]
                
                if nxt_reward >= mx_nxt_reward:
                    action = k
                    mx_nxt_reward = nxt_reward
        else:
            action = np.random.choice(self.actions)
        
        #select the next state based on action chosen
        position = self.State.nxtPosition(action)
        return position,action
    
    
    #Q-learning Algorithm
    def Q_Learning(self,episodes):
        x = 0
        #iterate through best path for each episode
        while(x < episodes):
            if self.isEnd:
                reward = self.State.getReward()
                self.rewards += reward
                self.plot_reward.append(self.rewards)
                
                #get state, assign reward to each Q_value in state
                i,j = self.State.state
                for a in self.actions:
                    self.new_Q[(i,j,a)] = round(reward,3)
                    
                #reset state and rewards
                self.State = State()
                self.isEnd = self.State.isEnd
                self.rewards = 0
                x+=1
            else:
                mx_nxt_value = -10
                #get current state, next state, action and current reward
                next_state, action = self.Action()
                i,j = self.State.state
                reward = self.State.getReward()
                self.rewards +=reward
                for a in self.actions:
                    nxtStateAction = (next_state[0], next_state[1], a)
                    q_value = (1-self.alpha)*self.Q[(i,j,action)] + self.alpha*(reward + self.gamma*self.Q[nxtStateAction])
                    if q_value >= mx_nxt_value:
                        mx_nxt_value = q_value
                
                #next state is now current state, check if end state
                self.State = State(state=next_state)
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd
                
                #update Q values with max Q value for next state
                self.new_Q[(i,j,action)] = round(mx_nxt_value,3)
            
            self.Q = self.new_Q.copy()
        print(self.Q)
        
    def plot(self,episodes):
        plt.plot(self.plot_reward)
        plt.show()
        
    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('-----------------------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                mx_nxt_value = -10
                for a in self.actions:
                    nxt_value = self.Q[(i,j,a)]
                    if nxt_value >= mx_nxt_value:
                        mx_nxt_value = nxt_value
                out += str(mx_nxt_value).ljust(6) + ' | '
            print(out)
        print('-----------------------------------------------')
        
    
        
if __name__ == "__main__":
    #agent for 10,000 episdoes 
    ag = Agent()
    episodes = 20000
    ag.Q_Learning(episodes)
    ag.plot(episodes)
    ag.showValues()