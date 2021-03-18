import os
import pickle
import random

import numpy as np
import torch

#Import learning algorithm including hyperparameters
from .train import QLearner
from .train import OurNeuralNetwork

MODEL_FILE_NAME = "DQNN.pt"


"""
import types
import tempfile
import keras.models

 
def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

"""

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    #make_keras_picklable()
    
    if self.train or not os.path.isfile(MODEL_FILE_NAME):
        self.logger.info("Setting up model from scratch.")
        self.qlearner   = QLearner(self.logger)
    else:
        self.logger.info("Loading model from saved state.")
        self.qlearner   = QLearner(self.logger)        
        NN = torch.load(MODEL_FILE_NAME)
        self.qlearner.PNN.load_state_dict(NN.state_dict())
        self.qlearner.PNN.eval()
        self.logger.info("Loaded parameters of NN.")
        
        self.qlearner.is_training = False
        self.qlearner.is_fit = True

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    return self.qlearner.propose_action(game_state)