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


# defines the board and decides reward, end and next position
class State:
    def __init__(self, state=START):
        self.state = state
        self.isEnd = False        

    def getReward(self):
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


        if (nxtState[0] >= 0) and (nxtState[0] < BOARD_ROWS):
            if (nxtState[1] >= 0) and (nxtState[1] < BOARD_COLS):             
                    return nxtState     
        return self.state 


        
# Class agent to implement reinforcement learning through grid  
class Agent:
    def __init__(self): 
        self.states = []
        self.actions = [0,1,2,3]   
        self.State = State()
        
        # =================================================================
        # IMPROVEMENT: Dynamic Learning and Exploration Rates
        # =================================================================
        self.gamma = 0.95   
        
        # Epsilon (Exploration)
        self.epsilon_start = 1.0  # Start with 100% exploration
        self.epsilon_min = 0.01   # Minimum exploration rate
        self.epsilon_decay = 0.0001 # Decay rate (tuned for 20000 episodes)
        self.epsilon = self.epsilon_start
        
        # Alpha (Learning Rate)
        self.alpha_start = 0.5    
        self.alpha_min = 0.01
        self.alpha_decay = 0.0001 
        self.alpha = self.alpha_start

        self.isEnd = self.State.isEnd
        self.plot_reward = []
        self.Q = {}
        self.new_Q = {}
        self.rewards = 0
        
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for k in range(len(self.actions)):
                    self.Q[(i, j, k)] = 0
                    self.new_Q[(i, j, k)] = 0
        
        print(self.Q)
        
    
    # choose action with Epsilon greedy policy, and move to next state
    def Action(self):
        # Random value vs epsilon
        rnd = random.random()
        mx_nxt_reward =-1e9 
        action = None
     
        if(rnd > self.epsilon) :
            for k in self.actions:
                i,j = self.State.state
                nxt_reward = self.Q[(i,j, k)]
                
                if nxt_reward >= mx_nxt_reward:
                    action = k
                    mx_nxt_reward = nxt_reward
        else:
            action = np.random.choice(self.actions)

        position = self.State.nxtPosition(action)
        return position, action
    
    
    # Q-learning Algorithm 
    def Q_Learning(self, episodes):
        x = 0
        while(x < episodes):
            if self.isEnd:
                reward = self.State.getReward()
                self.rewards += reward
                self.plot_reward.append(self.rewards)
                
                i,j = self.State.state
                for a in self.actions:
                    self.new_Q[(i,j,a)] = round(reward,3)
                    
                self.State = State()
                self.isEnd = self.State.isEnd
                self.rewards = 0
                
                # =============================================================
                # IMPROVEMENT: Apply Decay at Episode End
                # =============================================================
                # Exponential decay formula: E_min + (E_start - E_min) * exp(-decay * episode)
                self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * np.exp(-self.epsilon_decay * x)
                self.alpha = self.alpha_min + (self.alpha_start - self.alpha_min) * np.exp(-self.alpha_decay * x)
                x += 1
                
            else:
                current_state_coords = self.State.state
                action_taken = None 

                next_state_coords, action_taken = self.Action() 
                i, j = current_state_coords
                
                # reward 
                self.State.state = next_state_coords 
                reward = self.State.getReward()
                self.State.state = current_state_coords 
                self.rewards += reward
                
                max_q_for_next_state = -1e9 
                for a_prime in self.actions:
                    nxt_q_value = self.Q[(next_state_coords[0], next_state_coords[1], a_prime)]
                    if nxt_q_value > max_q_for_next_state:
                        max_q_for_next_state = nxt_q_value

                # Temporal Difference Target (TD Target)
                # TD Target = R + gamma * max_a' Q(s', a')
                td_target = reward + self.gamma * max_q_for_next_state

                # Q-Learning formula (TD update)
                # Q(s, a) <- Q(s, a) + alpha * [TD Target - Q(s, a)]
                current_q = self.Q[(i, j, action_taken)]
                td_error = td_target - current_q
                
                # decaying alpha for the update
                new_q_value = current_q + self.alpha * td_error 
                
                self.new_Q[(i, j, action_taken)] = round(new_q_value, 3)
                
                self.State.state = next_state_coords
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd
                
            self.Q = self.new_Q.copy()
            
        print(self.Q)
        
    def plot(self, episodes):
        plt.title(f'Q-Learning Cumulative Reward Over {episodes} Episodes (Improved)')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
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
    # create agent for 20,000 episdoes implementing a Q-learning algorithm 
    ag = Agent()
    episodes = 20000
    ag.Q_Learning(episodes)
    ag.plot(episodes)
    ag.showValues()