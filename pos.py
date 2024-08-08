from sympy.logic import POSform
from sympy import symbols
import random
n = 7
x = symbols([f"x_{i}" for i in range(n)])
# minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1],
#             [1, 0, 1, 1], [1, 1, 1, 1]]
# dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]

def generate_binary_list(n):
    binary_list = []
    for _ in range(n):
        # Generate a random binary digit (either 0 or 1)
        binary_digit = random.randint(0, 1)
        binary_list.append(binary_digit)
    return binary_list


minterms = [generate_binary_list(n) for _ in range(2)]
dontcares = [generate_binary_list(n) for _ in range(3)]
k = POSform(x, minterms)
print(k)