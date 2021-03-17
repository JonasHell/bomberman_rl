import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List
import errno
import os
from datetime import datetime

from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import scipy

MODEL_FILE_NAME = "DQNN.pt"

import events as e
import matplotlib.pyplot as plt

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'game_over'))

#lf: 1 if field left of player free, 0 otherwise
#rf: 1 if field right of player free, --
#uf: 1 if field above player free, --
#df: 1 if field below player free, --
#xc: Horizontal squared distance to nearest coin
#yc: Vertical squared distance to nearest coin
#cxd: 1 if coin is left of player, -1 otherwise
#cyd: 1 if coin is below player, -1 otherwise
features = ["lf", "rf", "uf", "df", "xc", "yc", "cxd", "cyd"]

# Hyper parameters -- DO modify
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
# Events
APPROACH_COIN_EVENT = "APPROACH_COIN"
MOVING_AWAY_COIN_EVENT = "MOVING_AWAY_COIN"
ALREADY_VISITED_EVENT = "ALREADY_VISITED"


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

class OurNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(OurNeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 32)
        self.linear5 = nn.Linear(32, 6)

    def forward(self, x):
        # könnte auch andere activation function nehmen
        out = self.linear1(x)
        out = F.selu(out)
        out = self.linear2(out)
        out = F.selu(out)
        out = self.linear3(out)
        out = F.selu(out)
        out = self.linear4(out)
        out = F.selu(out)
        out = self.linear5(out)
        return out

class QLearner:
    alpha: float = 0.5  #Lower alpha means slower but more stable convergence
    learning_rate: float = 5e-3 #0.1
    gamma: float = 0.9 #0.95 #Punishes expectation values in fucture
    memory_size: int = 5000
    batch_size: int = 500 #500
    exploration_decay: float = 0.99 #0.98
    exploration_max: float = 0.8 #1.0
    exploration_min: float = 0.1
    rewards: List[float] = [0]
    is_fit: bool = False
    is_training: bool = True
    actions: List[str] = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    action_id: dict = {
        'UP': 0,
        'RIGHT': 1,
        'DOWN': 2,
        'LEFT': 3,
        'WAIT': 4,
        'BOMB': 5
    }

    Loss: List[float] = [0]

    def __init__(self, logger):
        self.transitions = deque(maxlen=self.memory_size)
        self.model       = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=4))
        self.logger      = logger
        self.exploration_rate = self.exploration_max

        self.hyb_feat: bool = True #Use features from hybrid vector
        self.use_cuda: bool = True #Use GPU to train

        # Double QNN
        self.QNN = True #Use a Q-Learning Neural Network (QNN) instead of Gradient Boosting
        self.features_size = 1137 #8
        self.TNN = OurNeuralNetwork(self.features_size) #Target Neural Network (TNN)
        self.PNN = OurNeuralNetwork(self.features_size) #Prediction Neural Network (PNN)
        if self.use_cuda:
            self.TNN = self.TNN.cuda()
            self.PNN = self.PNN.cuda()
        self.TR: int = 10 #How often the parameters of the TNN should be replaced with the PNN
        self.tr: int = 0 #counter of TR
        # loss weights for DQfD
        self.l1: float = 1
        self.l2: float = 1
        self.l3: float = 1e-5
        self.optimizer = optim.Adam(self.PNN.parameters(), lr=self.learning_rate, weight_decay=self.l3) #Add L2 regularization for Adam optimized
        #Set TNN-paramters as PNN at start
        self.TNN.load_state_dict(self.PNN.state_dict())
        self.TNN.eval()

    def remember(self, state : np.array, action : str, next_state : np.array, reward: float , game_over : bool):
        #Store cumulative reward for performance assessment
        self.rewards.append(self.rewards[-1] + reward) 
        self.transitions.append([state, action, next_state, reward, game_over])

    def propose_action(self, game_state : dict):
        if not self.hyb_feat:
            state = self.state_to_features(game_state)
            #Plot current state of game
            self.logger.debug(f'Current state: {list(zip(features, state))}')
        else:
            state = state_to_features_hybrid_vec(game_state)
            #Plot current state of game
            #self.logger.debug(f'Current state: {state}')

        # Exploration vs exploitation
        # Epsilon - Greedy - Policy with Epsilon = 0.05
        if self.is_training and random.random() < self.exploration_rate:
            self.logger.debug(f'Choosing action purely at random with an exploration rate: {np.round(self.exploration_rate, 2)}')
        # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(self.actions, p=[.2, .2, .2, .2, .1, .1])
        
        if self.is_fit:
            self.logger.debug(f'Predict q value array in step {game_state["step"]}')
            if not self.QNN :
                q_values = self.model.predict(state.reshape(1, -1))
            else:
                q_values = self.PNN_predict(state).reshape(1, -1)
        else:
            self.logger.debug(f'Initialised q value array in step {game_state["step"]}')
            q_values = np.zeros(len(self.actions)).reshape(1, -1)

        proposed_action = self.actions[np.argmax(q_values[0])]
        #print('state_sum=', np.sum(state), 'Q-values:', np.round(q_values[0], 4), '->', proposed_action)

        if not self.hyb_feat: self.logger.info(f'Current state {state}')
        self.logger.info(f'Choosing {proposed_action} where {list(zip(self.actions, q_values[0]))}')

        return proposed_action

    def PNN_predict(self, state):
        self.PNN.eval() #set PNN in evaluation mode
        state = torch.Tensor(state).float()
        if self.use_cuda: state = state.cuda()
        q_values = self.PNN(state)
        if self.use_cuda:
            q_values = q_values.cuda().detach().cpu().clone().numpy()
        else:
            q_values = q_values.detach().numpy()
        self.PNN.train() #set PNN back in training modey
        return q_values

    def linear_propose_action(self, game_state):

        if not self.hyb_feat:
            state = self.state_to_features(game_state)
        else:
            state = state_to_features_hybrid_vec(game_state)

        #Plot current state of game
        self.logger.debug(f'Current state: {list(zip(features, state))}')

        # Exploration vs exploitation
        # Epsilon - Greedy - Policy with Epsilon = 0.05
        if self.is_training:
            if random.random() < self.exploration_rate:
                self.logger.debug("Choosing action purely at random.")
                # 80%: walk in any direction. 10% wait. 10% bomb.
                return np.random.choice(self.actions, p=[.2, .2, .2, .2, .2, .0])
        else:
            if random.random() < self.exploration_min:
                self.logger.debug("Choosing action purely at random.")
                # 80%: walk in any direction. 10% wait. 10% bomb.
                return np.random.choice(self.actions, p=[.2, .2, .2, .2, .2, .0])

        
        
        if self.is_fit:
            q_values = state.reshape(1, -1) @ self.beta
        else:
            self.logger.debug(f'Initialised q value array in step {game_state["step"]}')
            q_values = np.zeros(len(self.actions)).reshape(1, -1)

        proposed_action = self.actions[np.argmax(q_values[0])]
        
        return proposed_action


    def linear_experience_replay(self):
        if len(self.transitions) < self.batch_size:
            self.logger.debug("Experience buffer still insufficient for experience replay")
            return
        else:
            self.logger.debug(f'{len(self.transitions)} in experience buffer')


        #Sample random batch of experiences
        batch = random.sample(self.transitions, self.batch_size)

        X = [[]] * len(self.actions)
        targets_a = [[]] * len(self.actions)
        

        #Iterate through batch and generate training data
        for state, action, next_state, reward, game_over in batch:
            q_update = reward
            action_id = self.action_id[action]
                
            if self.is_fit:
                if not game_over:
                    q_update += self.gamma * np.amax(next_state.reshape(1, -1) @ beta)
                #self.logger.debug(f'Model fit before updating Q value.')

                #predict takes n_samples times n_features as argument
                #therefore we need to reshape
                q_values  = state.reshape(1, -1) @ beta
            
            else:
                q_values = np.ones(len(self.actions)).reshape(1, -1) * 35
                self.beta        = np.zeros((len(state), len(self.actions)))
            
                
            #self.logger.debug(f'Choosing action {action} we obtained reward {reward} and Q value = {q_values[0][action_id]} changed to Q = {q_update} after update in given state.')

            q_values[0][action_id] = (1 - self.alpha) * q_values[0][action_id] + self.alpha * q_update

            X[action_id].append(state)
            targets[action_id].append(q_values[0])

        #Fit model using training data
        #X: n_samples, n_features
        #targets: n_samples, n_actions

        for i in range(len(actions)):
            update = 0
            for j in range(len(X[i])):
                update += X[i][j] * (targets[i][j] - X[i][j] @ beta[:, i])
            beta[:, i] += self.learning_rate /  len(X[i]) * update
            
        self.is_fit = True
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)


    def experience_replay(self):
        #Check whether there are enough instances in experience buffer
        if len(self.transitions) < self.batch_size:
            self.logger.debug("Experience buffer still insufficient for experience replay")
            return
        else:
            self.logger.debug(f'{len(self.transitions)} in experience buffer')

        
        #Sample random batch of experiences
        batch = random.sample(self.transitions, self.batch_size)
        X = []
        targets = []

        #Iterate through batch and generate training data
        for state, action, next_state, reward, game_over in batch:
            q_update = reward
                
            if self.is_fit:
                if not game_over:
                    q_update += self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0])
                #self.logger.debug(f'Model fit before updating Q value.')

                #predict takes n_samples times n_features as argument
                #therefore we need to reshape
                q_values  = self.model.predict(state.reshape(1, -1))
            
            else:
                q_values = np.ones(len(self.actions)).reshape(1, -1) * 35
            
            action_id = self.action_id[action]
                
            #self.logger.debug(f'Choosing action {action} we obtained reward {reward} and Q value = {q_values[0][action_id]} changed to Q = {q_update} after update in given state.')

            q_values[0][action_id] = (1 - self.alpha) * q_values[0][action_id] + self.alpha * q_update

            X.append(state)
            targets.append(q_values[0])

        #Fit model using training data
        #X: n_samples, n_features
        #targets: n_samples, n_actions
        self.model.fit(X, targets) 
        self.is_fit = True
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

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

        self.is_fit = True
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def NN_update(self, batch):
        self.optimizer.zero_grad()
        batch = np.array(batch, dtype=object)
        game_overs = batch[:, 4]
        batch = batch[game_overs==0, :] #exclude transitions which ended in a game_over
        batch_size = len(batch)
        initial_states = np.concatenate(batch[:, 0]).reshape(batch_size, self.features_size)
        actions = [self.action_id[_] for _ in batch[:, 1]] #actions from 0 to 5
        new_states = np.concatenate(batch[:, 2]).reshape(batch_size, self.features_size)
        rewards = np.array(batch[:, 3], dtype=np.float32)
        
        new_states = torch.tensor(new_states).float()
        if self.use_cuda: new_states = new_states.cuda()
        maxq_actions = torch.amax( self.TNN( new_states ), axis=1 ) #actions with highest Q-value calculated with TNN
        #print(maxq_actions.shape, maxq_actions.dtype, rewards.shape, rewards.dtype)
        if self.use_cuda: maxq_actions = maxq_actions.cuda()
        target_q = torch.Tensor(rewards)
        if self.use_cuda: target_q = target_q.cuda()
        target_q += self.gamma * maxq_actions

        initial_states = torch.tensor(initial_states).float()
        if self.use_cuda: initial_states = initial_states.cuda()
        q_values = self.PNN( initial_states )#.detach().numpy() #All Q-values caclulated with PNN at initial states
        if self.use_cuda: q_values = q_values.cuda()
        predict_q = q_values[torch.arange(batch_size), actions] #Q-value of previously chosen action
        #predict_q = .clone().detach().requires_grad_(True)
        if self.use_cuda: predict_q = predict_q.cuda()

        #print('batchsize=', batch_size)
        #print(predict_q.shape, predict_q)
        #print(target_q.shape, target_q)
        criterion = nn.MSELoss()
        L = criterion(predict_q, target_q) #Total Loss
        if self.use_cuda:
            loss = L.cuda().detach().cpu().clone().numpy()
        else:
            loss = L.detach().numpy()
        self.Loss.append(loss)
        L.backward()
        self.optimizer.step() # backpropagation of PNN
        
        #self.NN_new_loss(loss, batch)

        # Update TNN-parameters with PNN every TR steps
        if self.tr < self.TR: self.tr += 1
        else:
            print('Loss:', loss)
            self.tr = 0
            self.TNN.load_state_dict(self.PNN.state_dict())
            self.TNN.eval()

    #Only used as debugging in NN_update() to calculate the loss after one step of backpropagation
    def NN_new_loss(self, old_loss, batch):
        batch = np.array(batch, dtype=object)
        game_overs = batch[:, 4]
        batch = batch[game_overs==0, :] #exclude transitions which ended in a game_over
        batch_size = len(batch)
        initial_states = np.concatenate(batch[:, 0]).reshape(batch_size, self.features_size)
        actions = [self.action_id[_] for _ in batch[:, 1]] #actions from 0 to 5
        new_states = np.concatenate(batch[:, 2]).reshape(batch_size, self.features_size)
        rewards = np.array(batch[:, 3], dtype=np.float32)
        
        new_states = torch.tensor(new_states).float()
        if self.use_cuda: new_states = new_states.cuda()
        maxq_actions = torch.amax( self.TNN( new_states ), axis=1 ) #actions with highest Q-value calculated with TNN
        #print(maxq_actions.shape, maxq_actions.dtype, rewards.shape, rewards.dtype)
        if self.use_cuda: maxq_actions = maxq_actions.cuda()
        target_q = torch.Tensor(rewards)
        if self.use_cuda: target_q = target_q.cuda()
        target_q += self.gamma * maxq_actions

        initial_states = torch.tensor(initial_states).float()
        if self.use_cuda: initial_states = initial_states.cuda()
        q_values = self.PNN( initial_states )#.detach().numpy() #All Q-values caclulated with PNN at initial states
        if self.use_cuda: q_values = q_values.cuda()
        predict_q = q_values[torch.arange(batch_size), actions] #Q-value of previously chosen action
        #predict_q = .clone().detach().requires_grad_(True)
        if self.use_cuda: predict_q = predict_q.cuda()

        #print('batchsize=', batch_size)
        #print(predict_q.shape, predict_q)
        #print(target_q.shape, target_q)
        criterion = nn.MSELoss()
        L = criterion(predict_q, target_q) #Total Loss
        if self.use_cuda:
            loss = L.cuda().detach().cpu().clone().numpy()
        else:
            loss = L.detach().numpy()
        print('Loss:', old_loss, 'New - Old Loss:', loss-old_loss )
        if loss == old_loss: print('Loss unchanged!')

    def state_to_features(self, game_state: dict) -> np.array:
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

        #First four entries indicate whether player can walk left, right, up or down
        #Next two entries indicate x- and y- distance to nearest coin
        #Last two entries indicate direction to nearest coin
        features = np.zeros(8)

        self_position = game_state['self'][3]
        field = game_state['field']
        coins = game_state['coins']

        if field[self_position[0] - 1][self_position[1]] == 0:
            features[0] = 1

        if field[self_position[0] + 1][self_position[1]] == 0:
            features[1] = 1

        if field[self_position[0]][self_position[1] - 1] == 0:
            features[2] = 1

        if field[self_position[0]][self_position[1] + 1] == 0:
            features[3] = 1

        
        if len(coins) > 0:

            distances = []

            for coin in coins:
                distances.append((coin[0] - self_position[0])**2 + (coin[1] - self_position[1])**2)
            
            idx = np.argmin(distances)

            dx = coins[idx][0] - self_position[0]
            dy = coins[idx][1] - self_position[1]
            if dx > 0.2:
                features[4] = 1
            if dx < -0.2:
                features[5] = 1
            if dy > 0.2:
                features[6] = 1
            if dy < -0.2:
                features[7] = 1

            if features[4] == 0 and features[5] == 0 and features[6] == 0 and features[7] == 0:
                print("something is wrong here")
                print(self_position)
                print(coins)

        #for coin in game_state['coins']:
        #    features[coin[0]][coin[1]] = 2

        #features[self_position[0]][self_position[1]] = 3
        
        #nb_classes = 5 # -1,0,1 for stone walls, free tiles, crates, 2 for coins, 3 for the agent

        #one_hot_features = np.eye(nb_classes)[features.flatten()]

        # For example, you could construct several channels of equal shape, ...
        #coins = game_state["coins"]
        #
      
        #channels = []
        #channels.append(coin_distances)
        # concatenate them as a feature tensor (they must have the same shape), ...
        #stacked_channels = np.stack(channels)
        # and return them as a vector
        return features#one_hot_features.flatten()#stacked_channels.reshape(-1)



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.step_counter = 0
    self.steps_before_replay = 5
    self.num_rounds = 100
    self.visited = np.zeros((17,17))
    self.visited_before = np.zeros((17,17))

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

    # Idea: Add your own events to hand out rewards
    if not self.qlearner.hyb_feat:
        old_f = self.qlearner.state_to_features(old_game_state)
        new_f = self.qlearner.state_to_features(new_game_state)
    else:
        old_f = state_to_features_hybrid_vec(old_game_state)
        new_f = state_to_features_hybrid_vec(new_game_state)

    # self_position = game_state['self'][3]
    # coins = game_state['coins'] 
    # if len(coins) > 0:
    #     distances = []
    #     for coin in coins:
    #         distances.append((coin[0] - self_position[0])**2 + (coin[1] - self_position[1])**2)
    #     idx = np.argmin(distances)

    #     dx = coins[idx][0] - self_position[0]
    #     dy = coins[idx][1] - self_position[1]
        
    if old_game_state != None:
        #Calculate nearest coin distance
        if len(old_game_state['coins']) > 0:
            old_coin_coords = np.moveaxis(np.array(old_game_state['coins']), -1, 0)
        if len(new_game_state['coins']) > 0:
            new_coin_coords = np.moveaxis(np.array(new_game_state['coins']), -1, 0)
        old_pos = old_game_state["self"][3]
        new_pos = new_game_state["self"][3]
        old_nearest_coin = np.min(np.abs(old_coin_coords[0] - old_pos[0]) + np.abs(old_coin_coords[1] - old_pos[1])) #Manhatten-distance
        new_nearest_coin = np.min(np.abs(new_coin_coords[0] - new_pos[0]) + np.abs(new_coin_coords[1] - new_pos[1])) #Manhatten-distance
        dist_diff = new_nearest_coin - old_nearest_coin
        #print('Position:', new_pos ,'Distance to nearest coin:', new_nearest_coin)
        self.logger.debug(f'Distance to nearest coin: {new_nearest_coin}')
        if dist_diff < 0:
            events.append(APPROACH_COIN_EVENT)
        elif dist_diff > 0:
            events.append(MOVING_AWAY_COIN_EVENT)

    x = new_game_state["self"][3][0]
    y = new_game_state["self"][3][1]

    if self.visited_before[x][y] == 1:
        events.append(ALREADY_VISITED_EVENT)

    self.visited_before = self.visited

    self.visited = np.zeros((17,17))
    self.visited[x][y] = 1



    if old_game_state == None:
        self.logger.debug(f'First game state initialised in step {new_game_state["step"]}')
        return

    if new_game_state == None:
        self.logger.debug(f'No new game state initialised after step {old_game_state["step"]}')
        return

    reward = reward_from_events(self, events)

    self.qlearner.remember(old_f, self_action, new_f, reward, game_over = False)

    #Only replay experiences every fixed number of steps
    self.step_counter += 1

    if self.step_counter == self.steps_before_replay:
        if not self.qlearner.QNN: self.qlearner.experience_replay()
        else: self.qlearner.prioritized_experience_replay()
        
        self.step_counter = 0

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

    if not self.qlearner.hyb_feat:
        last_f = self.qlearner.state_to_features(last_game_state)
    else:
        last_f = state_to_features_hybrid_vec(last_game_state) 
    reward = reward_from_events(self, events)

    self.qlearner.remember(last_f, last_action, None, reward, game_over = True)
    if not self.qlearner.QNN: self.qlearner.experience_replay()
    else: self.qlearner.prioritized_experience_replay()

    self.visited = np.zeros((17,17))
    self.visited_before = np.zeros((17,17))

    # Store the model in general directory
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.qlearner, file)

    # save the NN-model
    torch.save(self.qlearner.PNN, MODEL_FILE_NAME)
    self.logger.info("Model saved to " + MODEL_FILE_NAME)

    #If training finishes create folder for run
    #Last round assuming that we train for ten rounds
    if last_game_state["round"] == self.num_rounds:
        #Create new folder with time step for test run
        directory = create_folder()

        #Store cumulative rewards
        np.savetxt(directory + "/rewards.txt", self.qlearner.rewards, fmt='%.3f')
        #Store loss
        np.savetxt(directory + "/loss.txt", self.qlearner.Loss, fmt='%.3f')

        #Create plot for cumulative rewards
        x = np.arange(len(self.qlearner.rewards))
        plt.plot(x, self.qlearner.rewards)
        plt.title("Cumulative rewards")
        plt.savefig(directory + "/rewards.png")
        #Create plot for loss
        x2 = np.arange(len(self.qlearner.Loss))
        plt.plot(x2, self.qlearner.Loss)
        plt.title("Loss")
        plt.savefig(directory + "/loss.png")
        
        # Store the model in folder with timestamp for reference
        with open(directory + "/my-saved-model.pt", "wb") as file:
            pickle.dump(self.qlearner, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 50,
        e.MOVED_DOWN: -0.05,
        e.MOVED_LEFT: -0.05,
        e.MOVED_UP: -0.05,
        e.MOVED_RIGHT: -0.05,
        e.WAITED: -0.1,
        e.INVALID_ACTION: -0.1, #-0.05
        e.BOMB_DROPPED: -30,
        e.KILLED_SELF: -20, #-20
        #ALREADY_VISITED_EVENT: -0.05,
        APPROACH_COIN_EVENT: 1,
        MOVING_AWAY_COIN_EVENT: -0.5
        #Kill player 100
        #Break wall 30
        #Perform action -1
        #e.KILLED_OPPONENT: 5,
        #PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
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
    # For each cell on the field we define a vector with 5 entries, each either 0 or 1
    # [0, 0, 0, 0, 0] --> free
    # [1, 0, 0, 0, 0] --> stone
    # [0, 1, 0, 0, 0] --> crate
    # [0, 0, 1, 0, 0] --> coin
    # [0, 0, 0, 1, 0] --> bomb
    # [0, 0, 0, 0, 1] --> fire
    # in principle with this encoding multiple cases could happen at the same time
    # e.g. [0, 0, 0, 1, 1] --> bomb and fire
    # but in our implementation of the game this is not relevant
    # because they are a combination of one-hot and binary map
    # they are called hybrid vectors

    # initialize empty field
    # note: in the game we have a field of 17x17, but the borders are always
    # stone so we reduce the dimension to 15x15
    hybrid_vectors = np.zeros((15, 15, 5), dtype=int)
    
    # check where there are stones on the field
    # just use the field without the borders (1:-1)
    # set the first entry in the vector to 1
    hybrid_vectors[ np.where(game_state['field'][1:-1, 1:-1] == -1), 0 ] = 1

    # check where there are crates
    # set the second entry in the vector to 1
    hybrid_vectors[ np.where(game_state['field'][1:-1, 1:-1] == 1), 1 ] = 1

    # check where free coins are
    # set the third entry in the vector to 1
    # user np.moveaxis to transform list of tuples in numpy array
    # https://stackoverflow.com/questions/42537956/slice-numpy-array-using-list-of-coordinates
    # -1 in coordintaes because we left out the border
    if len(game_state['coins']) > 0:
        coin_coords = np.moveaxis(np.array(game_state['coins']), -1, 0)
        hybrid_vectors[ coin_coords[0]-1, coin_coords[1]-1, 2 ] = 1

    # check where bombs are
    # set the fourth entry in the vector to 1
    # discard the time since this can be learned by the model because we
    # use a LSTM network
    if len(game_state['bombs']) > 0:
        bomb_coords = np.array([[bomb[0][0], bomb[0][1], bomb[1]] for bomb in game_state['bombs']]).T
        hybrid_vectors[ bomb_coords[0]-1, bomb_coords[1]-1, 3 ] = bomb_coords[2]

    # vectorized version of above implementation
    '''
    bombs = game_state['bombs']
    n_bombs = len(bombs)
    bombs = np.asarray(bombs).T
    bombs_xy = np.concatenate(bombs[0])
    bombs_xy = bombs_xy.reshape(n_bombs, 2).T
    bombs_t = np.concatenate(bombs[1])
    hybrid_vectors[bombs_xy[0]-1, bombs_xy[1]-1, 3 ] = bombs_t
    '''

    # check where fire is
    # set the fifth entry in the vector to 1
    hybrid_vectors[ :, :, 4 ] = game_state['explosion_map'][1:-1, 1:-1]

    # flatten 3D array to 1D vector
    hyb_vec = hybrid_vectors.flatten()

    # add enemy coords and their bomb boolean as additional entries at the end
    # non-existing enemies have -1 at each position as default
    for i in range(3):
        if len(game_state['others']) > i:
            enemy = game_state['others'][i]
            hyb_vec = np.append(hyb_vec, [ enemy[3][0], enemy[3][1], int(enemy[2]) ])
        else:
            hyb_vec = np.append(hyb_vec, [ -1 , -1 , -1 ])

    # add own position and availability of bomb as 3 additional entries at the end
    hyb_vec = np.append(hyb_vec, [ game_state['self'][3][0], game_state['self'][3][1], int(game_state['self'][2]) ])

    return hyb_vec # len(hyb_vec) = (15 x 15 x 5) + (4 x 3) = 1137