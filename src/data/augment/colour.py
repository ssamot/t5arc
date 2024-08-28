import itertools
import math
import random

import numpy as np
from tqdm import tqdm


def apply_colour_augmentation_whole_dataset(dataset, max_perms):
    augmented_images = []
    for image in tqdm(dataset):
        extra_images = generate_consistent_combinations_2d(image, max_perms)
        # print(np.array(extra_images).shape)
        augmented_images.extend(extra_images)

    return np.array(augmented_images, dtype=np.int32)


def random_permutation(possible_values, num_unique_nums):
    return random.sample(possible_values, num_unique_nums)


def generate_consistent_combinations_2d(arr_2d, max_samples=1000):
    # Flatten the 2D array and get unique non-zero numbers
    flat_arr = [num for row in arr_2d for num in row if num != 0]
    unique_nums = sorted(set(flat_arr))

    # Generate all possible mappings
    possible_values = range(1, 11)
    n_mappings = math.perm(len(possible_values), len(unique_nums))
    # print(n_mappings)

    if (n_mappings > max_samples):
        sampled_mappings = set()
        # Sample unique permutations
        while len(sampled_mappings) < max_samples:
            sampled_mappings.add(tuple(random_permutation(possible_values, len(unique_nums))))

        all_mappings = list(sampled_mappings)
    else:
        all_mappings = list(itertools.permutations(possible_values, len(unique_nums)))

    result = []
    for mapping in all_mappings:
        # Create a dictionary to map original numbers to new numbers
        num_map = dict(zip(unique_nums, mapping))

        # Apply the mapping to the original 2D array, keeping zeros unchanged
        new_arr = [[num_map.get(num, 0) if num != 0 else 0 for num in row] for row in arr_2d]
        result.append(new_arr)

    return result


def generate_consistent_combinations_2d_dual(arr_2d_1, arr_2d_2, max_samples=1000):
    # Flatten both 2D arrays and get unique non-zero colors
    flat_arr_1 = [color for row in arr_2d_1 for color in row if color != 0]
    flat_arr_2 = [color for row in arr_2d_2 for color in row if color != 0]
    unique_colors = sorted(set(flat_arr_1 + flat_arr_2))

    # Generate all possible mappings
    possible_values = range(1, 11)  # Assuming we're still using numbers 1-10 as replacements
    all_mappings = list(itertools.permutations(possible_values, len(unique_colors)))

    # If there are more than max_samples mappings, randomly sample max_samples of them
    if len(all_mappings) > max_samples:
        all_mappings = random.sample(all_mappings, max_samples)

    result_1 = []
    result_2 = []
    for mapping in all_mappings:
        # Create a dictionary to map original colors to new numbers
        color_map = dict(zip(unique_colors, mapping))

        # Apply the mapping to both original 2D arrays, keeping zeros unchanged
        new_arr_1 = [[color_map.get(color, 0) if color != 0 else 0 for color in row] for row in arr_2d_1]
        new_arr_2 = [[color_map.get(color, 0) if color != 0 else 0 for color in row] for row in arr_2d_2]

        result_1.append(new_arr_1)
        result_2.append(new_arr_2)

    return result_1, result_2


def print_sample_combinations(original, combinations, num_samples=5):
    print(f"Original 2D array:")
    for row in original:
        print(row)

    print(f"\nTotal number of combinations: {len(combinations)}")
    print(f"\nFirst {num_samples} combinations:")
    for combo in combinations[:num_samples]:
        print(combo)


def print_sample_combinations_dual(original_1, original_2, combinations_1, combinations_2, num_samples=3):
    print("Original 2D arrays:")
    print("Array 1:")
    for row in original_1:
        print(row)
    print("\nArray 2:")
    for row in original_2:
        print(row)

    print(f"\nTotal number of combinations: {len(combinations_1)}")
    print(f"\nFirst {num_samples} combinations:")
    for i in range(min(num_samples, len(combinations_1))):
        print(f"\nCombination {i + 1}:")
        print("Array 1:")
        for row in combinations_1[i]:
            print(row)
        print("Array 2:")
        for row in combinations_2[i]:
            print(row)


if __name__ == '__main__':
    # Example usage for single input
    original_array_2d = [
        [3, 7, 8, 0],
        [1, 2, 3, 0],
        [5, 6, 7, 3]
    ]

    combinations = generate_consistent_combinations_2d(original_array_2d)
    print("Single Input Example:")
    print_sample_combinations(original_array_2d, combinations)

    # Example usage for dual input
    original_array_2d_1 = [
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [3, 0, 1, 2]
    ]

    original_array_2d_2 = [
        [4, 5, 0, 1],
        [2, 0, 4, 5],
        [0, 3, 2, 1]
    ]

    print("\n\nDual Input Example:")
    combinations_1, combinations_2 = generate_consistent_combinations_2d_dual(original_array_2d_1, original_array_2d_2)
    print_sample_combinations_dual(original_array_2d_1, original_array_2d_2, combinations_1, combinations_2)
