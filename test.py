import math


def total_combinations(depth, num_functions, arity):
    comb = math.perm(num_functions,arity)
    return comb**depth


def main():
    # Example parameters
    depth = 2
    num_functions = 50
    arity = 3
    combinations = total_combinations(depth, num_functions, arity)
    print(f"Total combinations for depth {depth} with {num_functions} functions and arity {arity}: {combinations}")


if __name__ == "__main__":
    main()
