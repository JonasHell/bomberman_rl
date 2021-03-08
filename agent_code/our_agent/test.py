import numpy as np

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
'''

field = np.random.randint(2, size=25)
field = field.reshape((5, 5))
print(field)

field = field.flatten()
print(field)

field = np.append(field, [4, 2, 6])
print(field)

print(int(False))
print(int(True))