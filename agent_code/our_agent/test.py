import numpy as np
import torch
import torch.nn.functional as F
import os
#from modified_rule_based_agent import Modified_Rule_Based_Agent
#from callbacks import OurNeuralNetwork

# number of parameters: 3.449.702 with 5 layer net with selu
# number of parameters:   652.646 with 4 layer net with selu
# number of parameters:   652.646 with 4 layer net with relu (1137->512)
# number of parameters:   308.166 with 3 layer net with selu (1137->256)

'''
result = np.zeros((5, 5, 2))

#np.seed = 0
field = np.random.randint(2, size=25)
field = field.reshape((5, 5))
print(field)
print(field[1:-1, 1:-1])

x_coord, y_coord = np.where(field[1:-1, 1:-1] == 1)
print(x_coord)
print(y_coord)

#result[x_ccord, y_coord] = 1
result[x_coord, y_coord, 0] = 1
print(result)

print(result.flatten())

coins = [('a', 2, True, (1, 1)), ('b', 1, False, (4, 3)), ('c', 0, True, (2, 1))]
print(coins)

ar = np.array([c[3] for c in coins]).T
print(ar)

field = np.random.randint(2, size=25)
field = field.reshape((5, 5))
print(field)

field = field.flatten()
print(field)

field = np.append(field, [4, 2, 6])
print(field)

print(int(False))
print(int(True))


#coins = [('a', 2, True, (1, 1)), ('b', 1, False, (4, 3)), ('c', 0, True, (2, 1))]
coins = np.random.randint(9, size=49)
coins = coins.reshape(7, 7)
print(coins)

matrix = np.zeros((5, 5, 5))
#print(matrix)

matrix[:, :, 1] = coins[1:-1, 1:-1]
print(matrix)

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
print(ACTIONS)
print(ACTIONS.index('DOWN'))
#print(np.where(np.array(ACTIONS) == 'LEFT'))


net = OurNeuralNetwork(1137)
print(sum(p.numel() for p in net.parameters() if p.requires_grad))

print(5%1)
print(0%1)
print(7%1)

ACTIONS = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])
print(ACTIONS[[0, 1, 0, 5]])

x = torch.tensor([0., 1., 2., 3., 4., 5.])
print(F.softmax(x, dim=0))
print(F.log_softmax(x, dim=0))

x = [np.array([1, 2, 3]),
     np.array([4, 5, 6]),
     np.array([7, 8, 9]),
     np.array([10, 11, 12])]

y = [5., 3., 8., 7.]

print(x)
#print(x.shape)
print(y)
#print(y.shape)
#data = np.concatenate((x, y), axis=1)
#print(data)
print()
#np.savetxt("foo.csv", data, delimiter=",")
np.savez_compressed("foo", features=x, labels=y)
load = np.load("foo.npz")
print(load["features"])
print(load["labels"])

print(np.concatenate([y for _ in range(3)]))

print(x)
print(np.array(x))
print(np.array(x, dtype=np.float32))

#load = np.genfromtxt('foo', delimiter=',')
load_comp = np.load("foo.npz")
load = load_comp['data']
print(load)
xn = load[:, :-1]
yn = load[:, -1]
print(xn)
print(yn)
'''

print(os.getcwd())
load = np.load("neural_network_pretraining/test_data/coins_.npz")
x = load['features']
y = load['labels']
print(y.shape)
print(y)

print(x)
print(x.shape)

print(len(np.concatenate((x, x), axis=0)))