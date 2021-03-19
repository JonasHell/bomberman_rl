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


ROWS = 7
COLS = 7
FEATURES_PER_FIELD = 7

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


class ConvNeuralNetwork(nn.Module):
    def __init__(self, input_size): #15 x 15 x 7 (Walls, Crates, Coins, Bombs, Fire, Players, Enemies)
        super(OurNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(FEATURES_PER_FIELD, 32, kernel_size=2, 1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        #self.conv_dropout = nn.Dropout2d()
        input_size = np.int((input_size-5) + 1) #after conv1 =11
        input_size = np.int(np.ceil((input_size-2)/2) + 1) #after max_pool2d =6
        input_size = np.int((input_size-3) + 1) #after conv2 =4
        input_size = 20*input_size*input_size
        self.linear1 = nn.Linear(input_size, 40)
        self.linear2 = nn.Linear(40, 6)

    def forward(self, x):
        # könnte auch andere activation function nehmen
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.selu(out)
        out = self.conv2(out)
        out = F.selu(out)
        out = out.view(-1, 320)
        out = self.linear1(out)
        out = F.selu(out)
        out = self.linear2(out)
        return out

class QLearner:
    #Lower alpha means slower but more stable convergence
    alpha: float = 0.8 
    #Learning rate for neural network
    learning_rate: float = 5e-3 #0.1
    #Punishes expectation values in fucture
    gamma: float = 0.9 #0.95 
    #Maximum site of transitions deque
    memory_size: int = 2048
    #Batch size used for training neural network
    batch_size: int = 128 #500
    exploration_decay: float = 0.96 #0.98
    exploration_max: float = 1.0 #1.0
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
        #Memory of the model, stores states and actions
        self.transitions = deque(maxlen=self.memory_size)
        self.logger      = logger
        #Determines exploration vs exploitation
        self.exploration_rate = self.exploration_max
        self.use_cuda: bool = False #Use GPU to train and play model

        # Double QNN
        self.features_size = ROWS*COLS*FEATURES_PER_FIELD #8
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
        state = state_to_features_hybrid_vec(game_state)

        # Exploration vs exploitation
        # Epsilon - Greedy - Policy with Epsilon = 0.05
        if self.is_training and random.random() < self.exploration_rate:
            self.logger.debug(f'Choosing action purely at random with an exploration rate: {np.round(self.exploration_rate, 2)}')
        # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(self.actions, p=[.2, .2, .2, .2, .1, .1])
        
        #If neural network is fit, have NN predict next move
        if self.is_fit:
            self.logger.debug(f'Predict q value array in step {game_state["step"]}')
            q_values = self.PNN_predict(state).reshape(1, -1)
        #Else, take random decision
        else:
            self.logger.debug(f'Initialised q value array in step {game_state["step"]}')
            q_values = np.zeros(len(self.actions)).reshape(1, -1)

        proposed_action = self.actions[np.argmax(q_values[0])]
        
        self.logger.debug(f'state_sum= {np.sum(state)} Q-values: {np.round(q_values[0], 4)} -> {proposed_action}')
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
        self.PNN.train() #set PNN back in training mode
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
        
        
            #Sample random batch of experiences
        batch = random.sample(self.transitions, self.batch_size)

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

        self.is_fit = True
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)



        #Run batch on NN
        self.NN_update(batch)

        self.is_fit = True
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    #Update NN with small batch sampled from transitions
    def NN_update(self, batch):

        # Clear out the gradients of all variables in this optimizer before backpropagation
        self.optimizer.zero_grad()
        batch = np.array(batch, dtype=object)
       
        #Assemble arrays of initial and final states as well as rewards and actions
        initial_states = np.concatenate(batch[:, 0]).reshape(batch_size, self.features_size)
        actions        = [self.action_id[_] for _ in batch[:, 1]] #actions from 0 to 5
        rewards        = np.array(batch[:, 3], dtype=np.float32)


        #Form tensor of initial Q values using the predictive NN for prediction
        initial_states = torch.tensor(initial_states).float()
        if self.use_cuda: initial_states = initial_states.cuda()
        q_values = self.PNN( initial_states ) 
        if self.use_cuda: q_values = q_values.cuda()

        #Q-values of actions chosen in each initial state
        predict_q = q_values[:, actions] 
        if self.use_cuda: predict_q = predict_q.cuda()

        #Form tensor from rewards to compute targets for Q matrix regression
        target_q = torch.Tensor(rewards)
        if self.use_cuda: target_q = target_q.cuda()

        #New state is none, if state is terminal (game_over = True)
        #Only compute maxq_actions for non-terminal states to update Q function
        non_terminal    = (batch[:, 4] == False)
        non_terminal_batch = batch[non_terminal, :] #exclude transitions which ended in a game_over
        if len(non_terminal_batch) > 0:
            #Form tensor from new states
            new_states     = np.concatenate(non_terminal_batch[:, 2]).reshape(len(non_terminal_batch), self.features_size)
            new_states     = torch.tensor(new_states).float()
            if self.use_cuda: new_states = new_states.cuda()

            #Use target NN to compute actions with highest Q-value
            maxq_actions = torch.amax( self.TNN( new_states ), axis=1 ) 
            #print(maxq_actions.shape, maxq_actions.dtype, rewards.shape, rewards.dtype)
            if self.use_cuda: maxq_actions = maxq_actions.cuda()

            #Only update Q value with non-terminal states with predicted Q values
            mask = torch.tensor(non_terminal)
            target_q[mask, :] += self.gamma * maxq_actions
            print(maxq_actions.shape, target_q[mask, :].shape)

        

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
            self.logger.debug(f'Current loss of NN: {loss}')
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


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.step_counter = 0
    self.steps_before_replay = 4
    self.num_rounds = 100

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
    self.step_counter += 1

    if self.step_counter == self.steps_before_replay:
        self.qlearner.prioritized_experience_replay()
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

    last_f = state_to_features_hybrid_vec(last_game_state) 
    reward = reward_from_events(self, events)

    self.qlearner.remember(last_f, last_action, None, reward, game_over = True)
    self.qlearner.prioritized_experience_replay()


    # save the NN-model
    torch.save(self.qlearner.PNN, MODEL_FILE_NAME)
    self.logger.info("Model saved to " + MODEL_FILE_NAME)

    #If training finishes create folder for run
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
        e.INVALID_ACTION: -0.1,
        e.BOMB_DROPPED: -30,
        e.KILLED_SELF: -20, 
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
    hybrid_vectors = np.zeros((COLS, ROWS, FEATURES_PER_FIELD), dtype=int)
    
    # check where there are stones on the field)
    # set the first entry in the vector to 1
    hybrid_vectors[ game_state['field'] == -1, 0 ] = 1

    # check where there are crates
    # set the second entry in the vector to 1
    hybrid_vectors[ game_state['field'] ==  1, 1 ] = 1

    # check where free coins are
    # set the third entry in the vector to 1
    # user np.moveaxis to transform list of tuples in numpy array
    # https://stackoverflow.com/questions/42537956/slice-numpy-array-using-list-of-coordinates
    if len(game_state['coins']) > 0:
        coin_coords = np.moveaxis(np.array(game_state['coins']), -1, 0)
        hybrid_vectors[ coin_coords[0], coin_coords[1], 2 ] = 1

    # check where bombs are
    # set the fourth entry in the vector to 1
    # discard the time since this can be learned by the model because we
    # use a LSTM network
    if len(game_state['bombs']) > 0:
        bomb_coords = np.array([[bomb[0][0], bomb[0][1], bomb[1]] for bomb in game_state['bombs']]).T
        hybrid_vectors[ bomb_coords[0], bomb_coords[1], 3 ] = bomb_coords[2]

    # check where fire is
    # set the fifth entry in the vector to 1
    hybrid_vectors[ :, :, 4 ] = game_state['explosion_map']

    # add enemy coords and their bomb boolean as additional entries at the end
    # non-existing enemies have -1 at each position as default
    for i in range(len(game_state['others'])):
        enemy = game_state['others'][i]
         #Value is 1 if enemy cannot place bomb and 2 if otherwise
        hybrid_vectors[enemy[3][0],enemy[3][1], 5] = 1 + int(enemy[2])

    # add player coordinates
    hybrid_vectors[ game_state['self'][3][0], game_state['self'][3][1], 6] = 1 + int(game_state['self'][2])

    return hybrid_vectors # len(hyb_vec) = (17 x 17 x 7) = 2023