import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class LinearModel:
    Q: float = 0.0

    #agents: List[Agent]

    def __init__(self, args):

    


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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        #Here we could do self.model = MyModel()
        self.model = weights
        self.q = weights
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    # Epsilon - Greedy - Policy with Epsilon = 0.05
    random_prob = .05
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    #or 
    #return np.random.choice(ACTIONS, p=self.model)
    return self.model.propose_action(game_state)


def state_to_features(game_state: dict) -> np.array:
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


    features = np.copy(game_state["field"])

    for coin in game_state["coins"]:
        features[coin[0]][coin[1]] = 2

    self_position = game_state["self"][3]
    features[self_position[0]][self_position[1]] = 3

    nb_classes = 5 # -1,0,1 for stone walls, free tiles, crates, 2 for coins, 3 for the agent

    one_hot_features = np.eye(nb_classes)[features.flatten()]

    # For example, you could construct several channels of equal shape, ...
    #coins = game_state["coins"]
    #
    #coin_distances = []
    
    #for i in range(len(coins)):
    #    coin_distances.append((coins[i][0] - self_position[0])**2 + (coins[i][1] - self_position[1])**2)

    #channels = []
    #channels.append(coin_distances)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector
    return one_hot_features#stacked_channels.reshape(-1)
