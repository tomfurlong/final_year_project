from random import shuffle

import numpy as np
from torch import cov


def min_set_cover(adj_matrix):
    length_X, length_S = adj_matrix.shape
    # print(length_X, length_S)
    # print(adj_matrix)
    set_cover_idx = list(range(length_S))
    #shuffle(set_cover_idx) # sorted so reproduceable results across runs

    removed = set({})
    for d in set_cover_idx:
        cover = adj_matrix[:,list(set(set_cover_idx) - removed.union({d}))]
        if np.any(cover.sum(axis=1) == 0): # At least one point is not covered
            continue
        removed.add(d)
    
    #returns the lines that are optimal that cover all the squares.
    return list(set(set_cover_idx) - removed)
