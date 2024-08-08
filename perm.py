from __future__ import print_function
from itertools import product       # forms cartesian products
n = 10                              # number of variables

print('All possible truth tables for n =', n)
inputs = list(product([0, 1], repeat=n))

for i, output in enumerate(product([0, 1], repeat=len(inputs))):
    print(i)
    print('Truth table')
    print('-----------')
    for row, result in zip(inputs, output):
        print(row, '-->', result)
