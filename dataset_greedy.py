import numpy as np
import os
from time import time
from ilp import ilp_problem
import pickle

DIR = "./instances/"

def generate_scp(N):
    """
    Given N, which is a power of 2
    Build Set Cover Problem with O(1) disks in optimal solution where greedy algorithm achieves O(log n) disks 
    Returns adjacency_matrix
    """

    P = list(range(N-1)) # N-1 points (aka O1)
    Q = list(range(N-1,2*(N-1))) # N-1 points (aka O2)

    adj_matrix = np.zeros((2*(N-1), int(np.log2(N)) + 2), dtype=int)
    adj_matrix[P,0] = 1
    adj_matrix[Q,1] = 1

    adj_idx = 2
    i = 0
    S = []
    while N/2 >= 1:
        s = P[i:int(i+N/2)] + Q[i:int(i+N/2)]
        S.append(s)
        adj_matrix[s,adj_idx] = 1

        i = int(i+N/2)
        N = N/2

        adj_idx += 1

    return adj_matrix

def build_dataset(name, N=[2**14, 2**15, 2**16]): #N=[256, 512, 1024, 2048]):
    instances = []
    for i in N:
        adj_matrix = generate_scp(i)
        instances.append(adj_matrix)

    pickle.dump(instances, open(DIR+ name +".p", "wb"))

def load_dataset(name):
    instances = pickle.load(open(DIR + name + ".p", "rb"))

    instances = [(i, [0, 1], disk_adj_matrix_greedy_dataset(i)) for i in instances]

    return instances

def compute_solution(name):
    instances = load_dataset(name)

    total_time = 0
    for i in instances:
        start_time = time()
        s = ilp_problem(i[0])
        end_time = time()

        diff = end_time - start_time
        total_time += diff

        assert list(np.where(np.array(s) == 1)[0]) == i[1]
    print(total_time)

def disk_adj_matrix_greedy_dataset(instance):
    """ Adjacency matrix for disks 
        A disk is said to be adjacent to another if they have points in common
        Here P (0) and Q (1) are adjacent to every disk in S
    """
    disk_adj_matrix = np.zeros((instance.shape[1], instance.shape[1]))

    # a disk is adjacent to itself
    disk_adj_matrix[0, 0] = 1
    disk_adj_matrix[1, 1] = 1
    for i in range(2,instance.shape[1]):
        disk_adj_matrix[i, i] = 1

        disk_adj_matrix[0, i] = 1
        disk_adj_matrix[1, i] = 1
        # matrix is symmetric
        disk_adj_matrix[i, 0] = 1
        disk_adj_matrix[i, 1] = 1

    return disk_adj_matrix

if __name__ == "__main__":
    build_dataset("greedy")
    # ilp_problem(generate_scp(2**11))
    #compute_solution("greedy_64")