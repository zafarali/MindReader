
import numpy as np


def swapcols(arr, cols):
    """Swaps the columns from the cols tuple. expect cols = (col0, col1)"""
    tmp = arr[:,cols[0]].copy()
    arr[:,cols[0]] = arr[:,cols[1]]
    arr[:,cols[1]] = tmp


assoc = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0]
]


arr = np.array(assoc)

print(arr)

