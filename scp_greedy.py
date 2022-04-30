import numpy as np

def greedy(adj_matrix):
    length_X, length_S = adj_matrix.shape
    C = set({})
    length_C = 0
    selected = []
    while length_C != length_S:
        min_alpha = None
        min_s = None
        for s in range(length_X):
            if not s in selected:
                uncovered_set_points = len((set(np.where(adj_matrix[s,:] == 1)[0]) - C))
                if uncovered_set_points == 0:
                    continue

                set_cost = 1
                alpha = set_cost / uncovered_set_points # cost-effectiveness (average cost per newly covered point)
                if min_alpha is None or alpha < min_alpha:
                    min_alpha = alpha
                    min_s = s
        if min_s is None:
            return selected
        selected.append(min_s)
        C = C.union(set(np.where(adj_matrix[min_s,:] == 1)[0]))
        length_C = len(C)
    return selected

