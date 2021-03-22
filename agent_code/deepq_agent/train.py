import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List
import errno
import os
from datetime import datetime

#Imports for neural network
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.tensorboard import SummaryWriter

import scipy

#File name for neural network
MODEL_FILE_NAME = "DQNN.pt"


import events as e
import matplotlib.pyplot as plt

#Create folder with timestamp of current run
def create_folder():
    mydir = os.path.join(
        os.getcwd(), 
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return mydir


ROWS = 17
COLS = 17
FEATURES_PER_FIELD = 7


#Converges for 7x7 network
"""
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv = nn.Conv2d(7, 32, kernel_size=2)
        self.pool = nn.MaxPool2d(2)
        self.hidden= nn.Linear(32*3*3, 128)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(128, 6)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv(x)) # [batch_size, 28, 26, 26]
        #print("First step: ",x.shape)
        x = self.pool(x) # [batch_size, 28, 13, 13]
        #print("Second step: ",x.shape)
        x = x.view(x.size(0), -1) # [batch_size, 28*13*13=4732]
        #print("Third step: ",x.shape)
        x = self.act(self.hidden(x)) # [batch_size, 128]
        #print("Fourth step: ",x.shape)
        x = self.drop(x)
        #print("Fifth step: ",x.shape)
        x = self.out(x) # [batch_size, 10]
        #print("Sixth step: ",x.shape)
        return x
"""

class NeuralNet(nn.Module):
  
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(7, 32, kernel_size = 5, padding = 2)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')

        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')

        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.1)

        self.linear3 = nn.Linear(64*4*4, 64)
        torch.nn.init.kaiming_uniform_(self.linear3.weight, mode='fan_in', nonlinearity='relu')
        self.drop3 = nn.Dropout(0.1)

        self.linear4 = nn.Linear(64,6)
        torch.nn.init.kaiming_uniform_(self.linear4.weight, mode='fan_in', nonlinearity='relu')
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.pool1(x))
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.relu(self.pool2(x))
        x = self.drop2(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x
  
class QLearner:
    #Learning rate for neural network
    learning_rate: float = 0.0005
    #Punishes expectation values in future
    gamma: float = 0.95
    #Maximum size of transitions deque
    memory_size: int = 4096
    #Batch size used for training neural network
    batch_size: int = 128 #500
    #Balance between exploration and exploitation
    exploration_decay: float = 0.999 #0.98
    exploration_max: float = 1.0 #1.0
    exploration_min: float = 0.5
    #For debug plots
    rewards: List[float] = [0]
    rewards_per_episode: List[float] = [0]
    is_training: bool = True
    Loss: List[float] = [0]

    actions: List[str] = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    action_id: dict = {
        'UP': 0,
        'RIGHT': 1,
        'DOWN': 2,
        'LEFT': 3,
        'WAIT': 4,
        'BOMB': 5
    }


    def __init__(self, logger):
        #Memory of the model, stores states and actions
        self.transitions = deque(maxlen=self.memory_size)
        self.logger      = logger

        #Determines exploration vs exploitation
        self.exploration_rate = self.exploration_max
        self.use_cuda: bool = True #Use GPU to train and play model

        # Double QNN
        self.TNN = NeuralNet() #Target Neural Network (TNN)
        self.PNN = NeuralNet() #Prediction Neural Network (PNN)
        if self.use_cuda:
            self.TNN = self.TNN.cuda()
            self.PNN = self.PNN.cuda()
        
        # Update frequency of target network
        self.TR: int = 128 #How often the parameters of the TNN should be replaced with the PNN
        self.tr: int = 0 #counter of TR

        # Loss weights for DQfD
        self.l1: float = 1
        self.l2: float = 1
        self.l3: float = 1e-5

        # Set learning rate and regularisation
        self.optimizer = optim.Adam(self.PNN.parameters(), lr=self.learning_rate, weight_decay=self.l3) #Add L2 regularization for Adam optimized
        
        #Set TNN-paramters as PNN at start
        self.TNN.load_state_dict(self.PNN.state_dict())
        self.TNN.eval()
        self.criterion = nn.MSELoss()

        # writer for tensorboard
        self.writer = SummaryWriter()
        self.epoch = 0


    def remember(self, state : np.array, action : str, next_state : np.array, reward: float , game_over : bool):
        #Store rewards for performance assessment
        self.rewards.append(reward) 
        self.transitions.append([state, action, next_state, reward, game_over])

    def propose_action(self, game_state : dict):
        #Obtain hybrid vector representation from game_state dictionary
        state = state_to_features_hybrid_vec(game_state)

        # Exploration vs exploitation
        if self.is_training and random.random() < self.exploration_rate:
            self.logger.debug(f'Choosing action purely at random with an exploration rate: {np.round(self.exploration_rate, 2)}')
        # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(self.actions, p=[.2, .2, .2, .2, .1, .1])
        
        #Have NN predict next move
        self.logger.debug(f'Predict q value array in step {game_state["step"]}')
        q_values = self.PNN_predict([state]).reshape(1, -1)

        proposed_action = self.actions[np.argmax(q_values[0])]
        
        self.logger.debug(f'state_sum= {np.sum(state)} Q-values: {np.round(q_values[0], 4)} -> {proposed_action}')
        self.logger.info(f'Choosing {proposed_action} where {list(zip(self.actions, q_values[0]))}')

        return proposed_action

    #Use PNN for prediction of next move
    def PNN_predict(self, state):
        #Set PNN to evaluation mode
        self.PNN.eval() 

        state = torch.Tensor(state).float()
        if self.use_cuda: state = state.cuda()

        #Prediction
        q_values = self.PNN(state)

        #Read data from GPU
        if self.use_cuda:
            q_values = q_values.cuda().detach().cpu().clone().numpy()
        else:
            q_values = q_values.detach().numpy()

        return q_values

    def prioritized_experience_replay(self): #not prioritized but normal experience replay still
        #Check whether there are enough instances in experience buffer
        if len(self.transitions) < self.batch_size:
            self.logger.debug("Experience buffer still insufficient for experience replay")
            return
        else:
            self.logger.debug(f'{len(self.transitions)} in experience buffer')
        
        #Sample random batch of experiences
        batch = random.sample(self.transitions, self.batch_size)

        #Run batch on NN
        self.NN_update(batch)

        #Update decay of exploration rate down to self.exploration_min
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    #Update NN with small batch sampled from transitions
    def NN_update(self, batch):

        batch = np.array(batch, dtype=object)
       
        #Assemble arrays of initial and final states as well as rewards and actions
        initial_states = np.stack(batch[:, 0])
        actions        = np.array([self.action_id[_] for _ in batch[:, 1]])
        rewards        = np.array(batch[:, 3], dtype=np.float32)


        #Form tensor of initial Q values using the predictive NN for prediction
        initial_states = torch.tensor(initial_states).type(torch.FloatTensor)
        if self.use_cuda: initial_states = initial_states.cuda()

        #Set both networks to evaluation mode
        self.PNN.eval()
        self.TNN.eval()

        #Predict initial Q values
        q_values = self.PNN( initial_states ) 
        if self.use_cuda: q_values = q_values.cuda()

        #Q-values of actions chosen in each initial state
        predict_q = q_values[np.arange(self.batch_size), actions]
        if self.use_cuda: predict_q = predict_q.cuda()

        #Form tensor from rewards to compute targets for Q matrix regression
        target_q = torch.tensor(rewards).type(torch.FloatTensor)
        if self.use_cuda: target_q = target_q.cuda()

        # New state is none if state is terminal which corresponds to the variable game_over = True
        # Only compute maxq_actions for non-terminal states to update Q function
        # Therefore we exclude transitions which ended in a game over
        non_terminal    = (batch[:, 4] == False)
        non_terminal_batch = batch[non_terminal, :] 
        if len(non_terminal_batch) > 0:
            #Form tensor from new states
            new_states     = np.stack(non_terminal_batch[:, 2])
            new_states     = torch.tensor(new_states).type(torch.FloatTensor)
            if self.use_cuda: new_states = new_states.cuda()

            #Use target NN to compute next actions
            q_next = self.TNN(new_states)

            #Choose action with highest Q value
            maxq_actions = torch.amax(q_next, axis=1 ) 
            #print(maxq_actions.shape, maxq_actions.dtype, rewards.shape, rewards.dtype)
            if self.use_cuda: maxq_actions = maxq_actions.cuda()

            #Only update Q value with non-terminal states with predicted Q values
            mask = torch.tensor(non_terminal)
            target_q[mask] += self.gamma * maxq_actions

        #Set back to training mode
        self.TNN.train()
        self.PNN.train()
        
        L = self.criterion(predict_q, target_q) #Total Loss

        loss = 0

        if self.use_cuda:
            loss = L.cuda().detach().cpu().clone().numpy()
        else:
            loss = L.detach().numpy()

        #Store loss for debugging
        self.Loss.append(loss)
        self.writer.add_scalar("Loss/train", loss, self.epoch)

        # Clear out the gradients of all variables in this optimizer before backpropagation
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step() # backpropagation of PNN
    
        # Update TNN-parameters with PNN every TR steps
        if self.tr < self.TR: 
            self.tr += 1
        else:
            self.logger.debug(f'Current loss of NN: {loss}')
            self.tr = 0
            self.TNN.load_state_dict(self.PNN.state_dict())
            self.TNN.eval()

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.step_counter = 0
    self.steps_before_replay = 8
    self.num_rounds = 400
    #Create new folder with time step for test run
    self.directory = create_folder()

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_f = state_to_features_hybrid_vec(old_game_state)
    new_f = state_to_features_hybrid_vec(new_game_state)

    if old_game_state == None:
        self.logger.debug(f'First game state initialised in step {new_game_state["step"]}')
        return

    if new_game_state == None:
        self.logger.debug(f'No new game state initialised after step {old_game_state["step"]}')
        return

    reward = reward_from_events(self, events)

    self.qlearner.remember(old_f, self_action, new_f, reward, game_over = False)

    #Only replay experiences every fixed number of steps
    self.step_counter   += 1
    self.qlearner.epoch += 1

    if self.step_counter == self.steps_before_replay:
        self.qlearner.prioritized_experience_replay()
        self.step_counter = 0

def movingaverage(interval, window_size = 30):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    last_f = state_to_features_hybrid_vec(last_game_state) 
    reward = reward_from_events(self, events)

    self.qlearner.remember(last_f, last_action, None, reward, game_over = True)
    self.qlearner.epoch += 1
    self.qlearner.prioritized_experience_replay()

    #Add rewards for current episode
    self.qlearner.rewards_per_episode.append(np.sum(self.qlearner.rewards))
    #Clear reward list for next episode
    self.qlearner.rewards = []



    #If training finishes create folder for run
    if last_game_state["round"] % self.num_rounds == 0:

        #Store tensorboard log
        self.qlearner.writer.flush()

        suffix = "_round_"+str(last_game_state["round"])

        #Store cumulative rewards
        np.savetxt(self.directory + "/rewards" + suffix + ".txt", self.qlearner.rewards_per_episode, fmt='%.3f')
        #Store loss
        np.savetxt(self.directory + "/loss" + suffix + ",txt", self.qlearner.Loss, fmt='%.3f')

        #Create plot for cumulative rewards
        x = np.arange(len(self.qlearner.rewards_per_episode))
        y = movingaverage(self.qlearner.rewards_per_episode)
        plt.plot(x, y)
        plt.title("Cumulative rewards per episode")
        plt.savefig(self.directory + "/rewards" + suffix + ".png")
        plt.clf()
        #Create plot for loss
        x2 = np.arange(len(self.qlearner.Loss))
        y2 = movingaverage(self.qlearner.Loss)
        plt.plot(x2, y2)
        plt.title("Loss")
        plt.savefig(self.directory + "/loss" + suffix + ".png")
        plt.clf()
        
        # save the NN-model
        torch.save(self.qlearner.PNN, MODEL_FILE_NAME)
        self.logger.info("Model saved to " + MODEL_FILE_NAME)

        #Store the training memory
        #with open("transitions.pt", "wb") as file:
        #    pickle.dump(self.qlearner.transitions, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.MOVED_DOWN: -0.05,
        e.MOVED_LEFT: -0.05,
        e.MOVED_UP: -0.05,
        e.MOVED_RIGHT: -0.05,
        e.BOMB_DROPPED: -0.05,
        e.CRATE_DESTROYED: 10,
        e.WAITED: -0.1,
        e.INVALID_ACTION: -0.1,
        e.KILLED_SELF: -50, 
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def state_to_features_hybrid_vec(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    # We want to represent the state as vector.
    # For each cell on the field we define a vector with 8 entries, each either 0 or 1
    # [0, 0, 0, 0, 0, 0, 0] --> free
    # [1, 0, 0, 0, 0, 0, 0] --> stone
    # [0, 1, 0, 0, 0, 0, 0] --> crate
    # [0, 0, 1, 0, 0, 0, 0] --> coin
    # [0, 0, 0, 5, 0, 0, 0] --> bomb, number shows bomb countdown
    # [0, 0, 0, 0, 1, 0, 0] --> fire
    # [0, 0, 0, 0, 0, 1, 0] --> enemy
    # [0, 0, 0, 0, 0, 0, 1] --> player
    # in principle with this encoding multiple cases could happen at the same time
    # e.g. [0, 0, 0, 1, 1] --> bomb and fire
    # but in our implementation of the game this is not relevant
    # because they are a combination of one-hot and binary map
    # they are called hybrid vectors

    # initialize empty field
    hybrid_vectors = np.zeros((FEATURES_PER_FIELD, COLS, ROWS), dtype=int)
    
    # check where there are stones on the field)
    # set the first entry in the vector to 1
    hybrid_vectors[0, game_state['field'] == -1] = 1

    # check where there are crates
    # set the second entry in the vector to 1
    hybrid_vectors[1, game_state['field'] ==  1] = 1

    # check where free coins are
    # set the third entry in the vector to 1
    # user np.moveaxis to transform list of tuples in numpy array
    # https://stackoverflow.com/questions/42537956/slice-numpy-array-using-list-of-coordinates
    if len(game_state['coins']) > 0:
        coin_coords = np.moveaxis(np.array(game_state['coins']), -1, 0)
        hybrid_vectors[2, coin_coords[0], coin_coords[1]] = 1
    
    # check where bombs are
    # set the fourth entry in the vector to 1
    # discard the time since this can be learned by the model because we
    # use a LSTM network
    if len(game_state['bombs']) > 0:
        bomb_coords = np.array([[bomb[0][0], bomb[0][1], bomb[1]] for bomb in game_state['bombs']]).T
        hybrid_vectors[3, bomb_coords[0], bomb_coords[1]] = bomb_coords[2]

    # check where fire is
    # set the fifth entry in the vector to 1
    hybrid_vectors[4, :, :] = game_state['explosion_map']

    # add enemy coords and their bomb boolean as additional entries at the end
    # non-existing enemies have -1 at each position as default
    for i in range(len(game_state['others'])):
        enemy = game_state['others'][i]
         #Value is 1 if enemy cannot place bomb and 2 if otherwise
        hybrid_vectors[5, enemy[3][0],enemy[3][1]] = 1 + int(enemy[2])
    
    # add player coordinates
    hybrid_vectors[6, game_state['self'][3][0], game_state['self'][3][1]] = 1 + int(game_state['self'][2])

    return hybrid_vectors