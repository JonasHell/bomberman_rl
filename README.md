# Our Bomberman Agent
This is the setup for a project/competition amongst students to train a machine learning agent for the game Bomberman.
Since we particpated in this challenge this repository also contains the code for our agent.

# Branches
The branch **NN-optim-jonas** was used to develop the direct policy learning of the imitation learning since this was independent of the Q-learning part.

The branch **NN-optim-convolutionalNet** contains in addition to the three-layer fully-connected network the convolutional network and implements the
behavioral cloning part of the imitation learning.

The branch **DLFD** contains an implementation of deep Q learning from demonstrations as suggested here https://arxiv.org/abs/1704.03732. 
In addition, it allows for generating new expert demonstration data during training similar to the epsilon greedy exploration strategy. 
