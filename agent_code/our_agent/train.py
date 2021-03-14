import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torch.utils.tensorboard import SummaryWriter

from callbacks import state_to_features

from modified_rule_based_agent import Modified_Rule_Based_Agent


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE_NAME = "our-saved-model.pt"
LEARNING_RATE = 0.01
WRITER = SummaryWriter("runs")


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.states = [] # array to save the game states that occured
    self.targets = [] # array to save what the rule based agent would do
    self.expert = Modified_Rule_Based_Agent()
    self.logger.debug("Everything is set up for this training game.")


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # append state and expert prediction
    self.states.append(state_to_features(new_game_state))
    target = self.expert.act(new_game_state)
    # self.targets.append((ACTIONS == traget)*1)
    # CrossEntropyLoss just needs the index of target class
    if target is None:
        # self.targets.append(np.random.choice([0, 1, 2, 3, 4, 5], p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]))
        self.targets.append(4) # for WAIT
    else:
        self.targets.append(ACTIONS.index(target))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # append last game state
    #self.states.append(state_to_features(last_game_state))
    #target = self.expert.act(last_game_state)
    # self.targets.append((ACTIONS == traget)*1)
    # CrossEntropyLoss just needs the index of target class
    #self.targets.append(ACTIONS.index(target))
    #self.logger.debug("Last game state appended.")

    # set model to trianing mode
    self.model.train()
    self.logger.info("Model set to training mode.")

    # translate states and targets to tensor and send to device, calculate output of network
    states = torch.tensor(self.states, dtype=torch.float).to(self.device)
    targets = torch.tensor(self.targets).type(torch.LongTensor).to(self.device)
    #targets = torch.tensor(self.targets)
    #targets = targets.type(torch.LongTensor).to(self.device)
    self.logger.debug("States and targets translated to tensors.")

    out = self.model(states).to(self.device)
    self.logger.debug("Output calculated.")

    # actual training with loss calculation, back propagation and optimization step
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, targets)

    self.model.zero_grad
    loss.backward()

    optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    optimizer.step()

    self.logger.debug("Training for this game done.")

    # loss auf tensorboard schieben bzw. erstmal printen um zu schauen obs läuft
    if (last_game_state['round']-1) % 50 == 0:
        print(f"[{last_game_state['round']:4}]: loss = {loss}")
        print(f"{'':6} survived steps = {last_game_state['step']}")
        print(f"{'':6} loss per step = {loss/last_game_state['step']}")
    #print(f"{'':6} score (own) = {self.score}")

    # save the model
    torch.save(self.model, MODEL_FILE_NAME)
    self.logger.info("Model saved to " + MODEL_FILE_NAME)

    # set everything back for next game
    # not sure if necessary, becuase I'm not sure when the setupt method is called
    # once at the beginning or at the beginning of every game
    self.states = []
    self.targets = []
    self.model.eval()
    self.logger.info("Everything set back for new game.")