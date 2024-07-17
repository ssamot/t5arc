
import numpy as np
from typing import List


def union2d(self, ar1: List | np.ndarray, ar2: List | np.ndarray) -> np.ndarray:
    delete_indices = []
    for i, elem in enumerate(ar1):
        for x in range(len(ar2)):
            if np.all(ar2[x] == elem):
                delete_indices.append(i)
    ar1 = np.delete(ar1, delete_indices, axis=0)

    return np.concatenate((ar1, ar2), axis=0)