from time import time

import numpy as np

from utils import build_valid_remove
from minimal_set_cover import min_set_cover
import dataset_cpp
from scp_greedy import greedy

def search(filehandle, idx, adj_matrix, set_cover_idx: set, TIMEOUT_SECS=180):
    start = time()

    set_uncover_idx = set(list(range(adj_matrix.shape[1]))) - set(set_cover_idx)
    best_cardinality = len(set_cover_idx)

    filehandle.write("%i,%s | %s\n" % (idx, best_cardinality, set_cover_idx))
    filehandle.flush()

    while (time() - start) < TIMEOUT_SECS:
        set_cover_idx_valid = build_valid_remove(adj_matrix, set_cover_idx)

        d = np.random.choice(list(set_cover_idx_valid) + list(set_uncover_idx))

        if d in set_cover_idx_valid:
            set_cover_idx.remove(d)
            set_uncover_idx.add(d)
        else:
            set_cover_idx.add(d)
            set_uncover_idx.remove(d)
        
        cardinality = len(set_cover_idx)
        if cardinality < best_cardinality:
            best_cardinality = cardinality
            filehandle.write("%i,%f,%i | %s\n" % (idx, (time() - start), best_cardinality, str(set_cover_idx)))
            filehandle.flush()
    
    filehandle.write("\n")

if __name__ == "__main__":
    instances_500 =  dataset_cpp.load_dataset_pickle("fix_500_0.1")
    # instances_1000 =  dataset_cpp.load_dataset_pickle("fix_1000_0.1")
    # instances_1500 =  dataset_cpp.load_dataset_pickle("fix_1500_0.1")
    # instances_2500 =  dataset_cpp.load_dataset_pickle("fix_2500_0.1")
    val_idx = 400
    instances_test = instances_500[val_idx:]

    filehandle = open("./output/random_search_" + str(round(time())), "w")

    for i in range(len(instances_test[:10])):
        adj_matrix, optimal_solution = instances_test[i]
        search(filehandle, i, adj_matrix, set(min_set_cover(adj_matrix)))
