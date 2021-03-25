import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List
from random import shuffle
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
MODEL_FILE_NAME = "DQNN_DD2.pt"


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


#Additional events
SAFE_DESPITE_BOMB = "SAFE_DESPITE_BOMB"
WITHIN_BOMB_REACH = "WITHIN_BOMB_REACH"

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
        #print(x.shape)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear3(x))
        #print(x.shape)
        #x = self.drop3(x)
        x = self.linear4(x)
        #print(x.shape)
        return x
  

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

def rule_act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] <= 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    # modified (16-->8)
    dead_ends = [(x, y) for x in range(1, 8) for y in range(1, 8) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 8) for y in range(1, 8) if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, None)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a


class QLearner:
    #Learning rate for neural network
    learning_rate: float = 0.0005 #0.1
    #Punishes expectation values in future
    gamma: float = 0.9
    #Maximum size of transitions deque
    memory_size: int = 2048*2
    #Batch size used for training neural network
    batch_size: int = 256 #500

    #Number of expert transitions taken in batches after pretraining
    expert_share: int = 13

    #Add pretraining phase before 
    is_pretraining: bool = True
    #Number of transitions collected before pretraining
    expert_memory_size: int = 512
    #Number of pretraining iterations with batch_size
    expert_training_rounds: int = 16
    #Stores when expert has demonstrated action
    last_action_was_expert = False

    #Balance between exploration and exploitation
    exploration_decay: float = 0.99 #0.98
    exploration_max: float = 0.1 #1.0
    exploration_min: float = 0.05
    
    expert_decay: float = 0.999 #0.98
    expert_max: float = 0.0 #1.0
    expert_min: float = 0.0

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
        self.transitions   = deque(maxlen=self.memory_size)
        self.expert_buffer = deque(maxlen=self.expert_memory_size)
        self.logger      = logger

        #Determines exploration vs exploitation
        self.exploration_rate = self.exploration_max        
        self.expert_rate = self.expert_max
        self.use_cuda: bool = False #Use GPU to train and play model

        # Double QNN
        self.TNN = NeuralNet() #Target Neural Network (TNN)
        self.PNN = NeuralNet() #Prediction Neural Network (PNN)
        if self.use_cuda:
            self.TNN = self.TNN.cuda()
            self.PNN = self.PNN.cuda()
        
        # Update frequency of target network
        self.TR: int = 32 #How often the parameters of the TNN should be replaced with the PNN
        self.tr: int = 0 #counter of TR

        # Loss weights for DQfD
        self.l1: float = 1
        self.l2: float = 1
        self.l3: float = 1e-5
        self.expert_margin: float = 0.8

        # Set learning rate and regularisation
        self.optimizer = optim.Adam(self.PNN.parameters(), lr=self.learning_rate, weight_decay=self.l3) #Add L2 regularization for Adam optimized
        
        #Set TNN-paramters as PNN at start
        self.TNN.load_state_dict(self.PNN.state_dict())
        self.TNN.eval()
        self.criterion = nn.MSELoss()

        # writer for tensorboard
        self.writer = SummaryWriter()
        self.epoch = 0

        #Add rule based agent code
        np.random.seed()
        # Fixed length FIFO queues to avoid repeating the same actions
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        # While this timer is positive, agent will not hunt/attack opponents
        self.ignore_others_timer = 0
        self.current_round = 0



    def remember(self, state : np.array, action : str, next_state : np.array, reward: float , game_over : bool):
        #Store rewards for performance assessment
        self.rewards.append(reward)

        if self.is_pretraining:
            self.expert_buffer.append([state, action, next_state, reward, game_over, self.last_action_was_expert])
        else:
            self.transitions.append([state, action, next_state, reward, game_over, self.last_action_was_expert])

    def propose_action(self, game_state : dict):
        # Pretraining with expert data
        if self.is_pretraining:
            action = rule_act(self, game_state)
            self.last_action_was_expert = True
            if action is None:
              action = "WAIT"
            return action

        # Exploration vs exploitation
        if self.is_training and random.random() < self.exploration_rate:
            self.last_action_was_expert = False
            self.logger.debug(f'Choosing action purely at random with an exploration rate: {np.round(self.exploration_rate, 2)}')
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(self.actions, p=[.2, .2, .2, .2, .1, .1])
        #Optionally also choose certain percentage of new expert actions during Q training
        elif self.is_training and random.random() < self.expert_rate:
            action = rule_act(self, game_state)
            self.last_action_was_expert = True
            if action is None:
              action = "WAIT"
            return action

        self.last_action_was_expert = False
        
        #Obtain hybrid vector representation from game_state dictionary
        state = state_to_features_hybrid_vec(game_state)

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

    def expert_experience_replay(self):
        if len(self.expert_buffer) < self.batch_size:
            return
        
        #Sample random batch of experiences from expert buffer
        batch = random.sample(self.expert_buffer, self.batch_size)
        #Run batch on NN
        self.NN_update(batch)

    def prioritized_experience_replay(self): #not prioritized but normal experience replay still
        #Check whether there are enough instances in experience buffer
        if len(self.transitions) < self.batch_size:
            self.logger.debug("Experience buffer still insufficient for experience replay")
            return
        else:
            self.logger.debug(f'{len(self.transitions)} in experience buffer')
        
        #Sample random batch of experiences from Q learning
        batch   = random.sample(self.transitions, self.batch_size - self.expert_share)
        #Always keep a certain share of experiences from expert buffer even after pretraining
        expert_batch = random.sample(self.expert_buffer, self.expert_share)

        batch.extend(expert_batch)



        #Run batch on NN
        self.NN_update(batch)

        #Update decay of exploration rate down to self.exploration_min
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

        #Update decay of exploration rate down to self.exploration_min
        self.expert_rate *= self.expert_decay
        self.expert_rate = max(self.expert_min, self.expert_rate)

    #Compute 'large margin classification loss' from expert data
    def NN_expert_loss(self, batch_size, initial_states, actions, q_values, predict_q):
        count = torch.arange(6)
        actions = np.array(actions)
        actions = torch.tensor(actions.reshape(batch_size, 1)).int()
        if self.use_cuda:
            count = count.cuda()
            actions = actions.cuda()
        mask = (actions != count)*self.expert_margin
        if self.use_cuda:
            mask = mask.cuda()
        L = torch.amax(q_values + mask, axis=1) - predict_q
        return L

    #Update NN with small batch sampled from transitions
    def NN_update(self, batch):

        batch = np.array(batch, dtype=object)
        batch_size = len(batch)
        #Assemble arrays of initial and final states as well as rewards and actions
        initial_states = np.stack(batch[:, 0])
        actions        = np.array([self.action_id[_] for _ in batch[:, 1]])
        rewards        = np.array(batch[:, 3], dtype=np.float32)
        is_expert      = np.array(batch[:, 5], dtype=bool) # True if the transition is taken from expert data

        is_expert = torch.tensor(is_expert).type(torch.BoolTensor)
        if self.use_cuda:
            is_expert = is_expert.cuda()

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
        #Set back to training mode
        non_terminal_batch = batch[non_terminal, :] 
        if len(non_terminal_batch) > 0:
            #Form tensor from new states
            new_states     = np.stack(non_terminal_batch[:, 2])
            new_states     = torch.tensor(new_states).type(torch.FloatTensor)
            if self.use_cuda: new_states = new_states.cuda()

            #Use target NN to compute next actions
            q_next = self.PNN(new_states)           #all PNN Q-values of next state
            a_next = torch.argmax(q_next, axis=1)   #best action of next state
            maxq = self.TNN(new_states)             #all TNN Q-values of next state
            maxq = Variable(maxq, requires_grad = False)
            maxq_actions = maxq[np.arange(len(non_terminal_batch)), a_next] #new Q-value
            #Choose action with highest Q value
            #maxq_actions = torch.amax(q_next, axis=1 ) 
            #print(maxq_actions.shape, maxq_actions.dtype, rewards.shape, rewards.dtype)
            if self.use_cuda: maxq_actions = maxq_actions.cuda()

            #Only update Q value with non-terminal states with predicted Q values
            mask = torch.tensor(non_terminal)
            target_q[mask] += self.gamma * maxq_actions


        #Set back to training mode
        self.PNN.train()
        
        L = torch.sum( (predict_q - target_q) ** 2 )/batch_size 

        expert_size = torch.sum(is_expert)
        L_expert = torch.sum( self.NN_expert_loss(batch_size, initial_states, actions, q_values, predict_q)*is_expert ) / expert_size
        L += L_expert*self.l2
        loss = 0

        if self.use_cuda:
            loss = L.cuda().detach().cpu().clone().numpy()
        else:
            loss = L.detach().numpy()

        #Store loss for debugging
        self.Loss.append(loss)

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
    self.num_rounds = 50
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

    
    #Determine bomb rewards
    #Iterate through bombs
    #Use knowledge about game board
    # xxxxxxx
    # x     x
    # x x x x
    # x     x
    # x x x x
    # x     x
    # xxxxxxx
    #
    # Stone walls that can block bombs are in rows and columns with even indices 0, 2, 4 ...
    
    within_bomb_reach = False
    safe_despite_bomb = False

    #Player position
    x, y = new_game_state['self'][3]

    for bomb in new_game_state['bombs']:
        #Compute x and y distance from bomb
        x_dist = np.abs(bomb[0][0] - x)
        y_dist = np.abs(bomb[0][1] - y)

        if  x_dist == 0 and y_dist == 0:
            safe_despite_bomb = False
            within_bomb_reach = True
            
        #Check whether bomb is threat horizontally
        #Therefore we must be wihin bomb reach (x_dist <= 3)
        #And we must be in uneven row
        if  x_dist < 4 and y_dist == 0 and (y % 2) != 0:
            safe_despite_bomb = False
            within_bomb_reach = True

        #Check whether bomb is threat vertically
        if  y_dist < 4 and x_dist == 0 and (x % 2) != 0:
            safe_despite_bomb = False
            within_bomb_reach = True
        
        # If bomb is close and we are safe from all! bombs
        if  x_dist + y_dist < 4 and not within_bomb_reach:
            safe_despite_bomb = True

    if within_bomb_reach:
        events.append(WITHIN_BOMB_REACH)

    if safe_despite_bomb:
        events.append(SAFE_DESPITE_BOMB)
        

    reward = reward_from_events(self, events)

    self.qlearner.remember(old_f, self_action, new_f, reward, game_over = False)

    #Only collect transitions during pretraining
    #No prioritized experience replay during pretraining
    if self.qlearner.is_pretraining:
        #If we have enough expert transitions
        #Train network for self.expert_training_rounds with self.batch_size sized batches from expert buffer
        #Then start regular Q learning
        if len(self.qlearner.expert_buffer) == self.qlearner.expert_memory_size:
            for i in range(self.qlearner.expert_training_rounds):
                self.qlearner.expert_experience_replay()
            
            self.qlearner.is_pretraining = False
    else:
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

    if not self.qlearner.is_pretraining:
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
        np.savetxt(self.directory + "/loss" + suffix + ".txt", self.qlearner.Loss, fmt='%.3f')

        #Create plot for cumulative rewards
        if len(self.qlearner.rewards_per_episode) > 0:
          x = np.arange(len(self.qlearner.rewards_per_episode))
          y = movingaverage(self.qlearner.rewards_per_episode)
          plt.plot(x, y)
          plt.title("Cumulative rewards per episode")
          plt.savefig(self.directory + "/rewards" + suffix + ".png")
          plt.clf()
        #Create plot for loss
        if len(self.qlearner.Loss) > 0:
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
        e.KILLED_OPPONENT: 1.0,
        e.SURVIVED_ROUND: 0.1,
        e.COIN_COLLECTED: 0.2,
        e.CRATE_DESTROYED: 0.1,
        SAFE_DESPITE_BOMB: 0.02,
        WITHIN_BOMB_REACH: -0.000666,
        e.MOVED_DOWN: -0.01,
        e.MOVED_LEFT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_RIGHT: -0.01,
        e.BOMB_DROPPED: -0.01,
        e.WAITED: -0.01,
        e.INVALID_ACTION: -0.02,
        e.GOT_KILLED: -0.5,
        e.KILLED_SELF: -1.0, 
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