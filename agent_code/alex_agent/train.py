import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

import events as e

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'game_over'))

# Hyper parameters -- DO modify
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
# Events
COIN_EVENT = "COIN_COLLECTED"

class QLearner:
    alpha: float = 0.1  #Learning rate for Q function
    gamma: float = 0.95 #Punishes expectation values in fucture
    memory_size: int = 1000
    batch_size: int = 5
    exploration_rate: float = 0.95
    epsilon: float = 0.3
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

    def __init__(self, logger):
        self.transitions = deque(maxlen=self.memory_size)
        self.model       = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))
        self.logger      = logger

    def remember(self, state : dict, action : str, next_state : dict, events : List[str], game_over : bool):
        self.transitions.append([self.state_to_features(state), action, self.state_to_features(next_state), reward_from_events(self, events), game_over])

    def propose_action(self, game_state : dict):
        state = self.state_to_features(game_state)
        # Exploration vs exploitation
        # Epsilon - Greedy - Policy with Epsilon = 0.05
        if self.is_training and random.random() < self.epsilon:
            self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(self.actions, p=[.2, .2, .2, .2, .1, .1])

        if self.is_fit:
            self.logger.debug(f'Predict q value array in step {game_state["step"]}')
            q_values = self.model.predict(state.reshape(1, -1))
        else:
            self.logger.debug(f'Initialised q value array in step {game_state["step"]}')
            q_values = np.zeros(len(self.
            actions)).reshape(1, -1)

        self.logger.debug(f'Propose action with a q value of {np.max(q_values[0])}')

        return self.actions[np.argmax(q_values[0])]

    def experience_replay(self):
        #Check whether there are enough instances in experience buffer
        if len(self.transitions) < self.batch_size:
            self.logger.debug("Experience buffer still insufficient for experience replay")
            return
        else:
            self.logger.debug(f'{len(self.transitions)} in experience buffer')

        
        #Sample random batch of experiences
        #number_of_samples = self.transitions.shape[0]
        #idx = np.random.choice(number_of_samples, self.batch_size, replace=False)
        X = []
        targets = []



        #Iterate through batch and generate training data
        for state, action, next_state, reward, game_over in self.transitions:
            q_update = reward
            q_values = np.zeros(len(self.actions)).reshape(1, -1)

            if self.is_fit:
                self.logger.debug(f'Model fit before updating Q value.')

                #predict takes n_samples times n_features as argument
                #therefore we need to reshape
                if not game_over:
                    q_update += self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0])
                q_values  = self.model.predict(state.reshape(1, -1))
            
            action_id = self.action_id[action]
                
            self.logger.debug(f'Reward {reward}, Q value = {q_values[0][action_id]} before and Q = {q_update} after update.')

            q_values[0][action_id] = q_update

            X.append(state)
            targets.append(q_values[0])

        #Fit model using training data
        #X: n_samples, n_features
        #targets: n_samples, n_actions
        self.model.fit(X, targets) 
        self.is_fit = True

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

        features = game_state['field']

        for coin in game_state['coins']:
            features[coin[0]][coin[1]] = 2

        self_position = game_state['self'][3]
        features[self_position[0]][self_position[1]] = 3

        #nb_classes = 5 # -1,0,1 for stone walls, free tiles, crates, 2 for coins, 3 for the agent

        #one_hot_features = np.eye(nb_classes)[features.flatten()]

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
        return features.flatten()#one_hot_features#stacked_channels.reshape(-1)



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

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
    #if ...:
        #events.append(PLACEHOLDER_EVENT)
    if old_game_state == None:
        self.logger.debug(f'First game state initialised in step {new_game_state["step"]}')
        return

    if new_game_state == None:
        self.logger.debug(f'No new game state initialised after step {old_game_state["step"]}')
        return

    self.qlearner.remember(old_game_state, self_action, new_game_state, events, game_over = False)
    self.qlearner.experience_replay()

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

    self.qlearner.remember(last_game_state, last_action, None, events, game_over = True)
    self.qlearner.experience_replay()

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.qlearner, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.MOVED_DOWN: -0.25,
        e.MOVED_LEFT: -0.25,
        e.MOVED_UP: -0.25,
        e.MOVED_RIGHT: -0.25,
        e.WAITED: -0.5,
        e.INVALID_ACTION: -2,
        e.KILLED_SELF: -300
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