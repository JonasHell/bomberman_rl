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
MODEL_FILE_NAME = "layer4_batch1_lr1"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # set learning parameters
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
    self.batch_size = 1
    
    # init counter
    self.global_step = 0
    self.correct_counter = 0

    # writer for tensorboard
    self.writer = SummaryWriter("../../runs/"+MODEL_FILE_NAME)

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
    # get opinion of expert
    target = self.expert.act(new_game_state)
    if target is not None:
        # append to states and target
        self.targets.append(ACTIONS.index(target)) # CrossEntropyLoss just needs the index of target class
        self.states.append(state_to_features(new_game_state))
        self.global_step += 1

        # set model to trianing mode
        self.model.train()
        self.logger.info("Model set to training mode.")

        # translate states and targets to tensor and send to device, calculate output of network
        states = torch.tensor(self.states, dtype=torch.float).to(self.device)
        targets = torch.tensor(self.targets).type(torch.LongTensor).to(self.device)

        self.logger.debug("States and targets translated to tensors.")

        out = self.model(states).to(self.device)

        self.logger.debug("Output calculated.")

        # actual training with loss calculation, back propagation and optimization step
        loss = self.criterion(out, targets)

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        # loss auf tensorboard schieben bzw. erstmal printen um zu schauen obs läuft
        our_pred = ACTIONS[torch.argmax(out)]
        if target == our_pred:
            self.correct_counter += 1
        print(f"[{self.global_step:6}]: loss={loss:.4f} acc={self.correct_counter*100./self.global_step:.2f}% our={our_pred:5} exp={target:5}")
        self.writer.add_scalar("training loss", loss, self.global_step)
        self.writer.add_scalar("training correct predictions", self.correct_counter, self.global_step)
        # set everything back for next game
        # not sure if necessary, becuase I'm not sure when the setupt method is called
        # once at the beginning or at the beginning of every game
        self.states = []
        self.targets = []
        self.model.eval()
        self.logger.info("Everything set back for new game.")

    else:
        print("Target is None!!!", self.global_step)
        print(old_game_state['round'], ", ", old_game_state['step'])
        print(new_game_state['round'], ", ", new_game_state['step'])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    print(f"end of round, {last_game_state['round']}, {last_game_state['step']}")
    print("************************************************************************")
    print()
    
    self.states = []
    self.targets = []
    self.model.eval()
    self.global_step += 1
    self.logger.info("Everything set back for new game.")

    
    # save the model
    torch.save(self.model, MODEL_FILE_NAME+".pt")
    self.logger.info("Model saved to " + MODEL_FILE_NAME+".pt")