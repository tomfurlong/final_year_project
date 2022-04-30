from distutils import core
import numpy as np
import os
os.environ["XPAUTH_PATH"] = "/home/dajwani/software/xpress-mp-server/bin/xpauth.xpr"

from time import time
from ilp import ilp_problem
import pickle
import math

DIR = "./instances/"

def generate_square_stabbing_problem(N):
    """
    Given N, which is a power of 2
    Build Stabbing Squares Problem with O(1) disks in optimal solution where greedy algorithm achieves O(log n) disks 
    Returns adjacency_matrix
    """
    """
    random dataset select randomly between 2n random number a Y cordinate
    and an X corddinate that defines the bottom left of the square 
    """
    '''We dont want number of squares or size of squares to be too small as this
    wont lead to interesting solutions, optimal solution won't be great
    possibly boring instance so we reduce the size of the space'''
    x_cords=[]
    y_cords=[]
    
    x_cords = math.sqrt(N)*np.random.random(N)
    y_cords = math.sqrt(N)*np.random.random(N)
    
    corner_of_squares = list(zip(x_cords, y_cords))
    
    P = list(range(N-1)) # N-1 points (aka O1)
    Q = list(range(N-1,2*(N-1))) # N-1 points (aka O2)

    # '''matrix of size 2*N x N
    # 2*N being the amount of lines , N horzontal and 
    # N vertical lines and the N squares'''

    # checking to see if the bottom line of the squares pass through another line
    # print(adj_matrix)
    # print(cords)

    '''matrix of size 2*N x N
    2*N being the amount of lines , N horzontal and
    N vertical lines and the N squares'''
    adj_matrix = np.zeros((2*(N), N), dtype=int)

    '''Horizontal lines corresponding to the bottom of the square'''
    for i in range(N):
        for j in range(N):
            if (y_cords[j]<=y_cords[i]) and (y_cords[i]<=y_cords[j] + 1):
                adj_matrix[i][j]=1

    '''Vertical lines corresponding to the right side of the square'''
    for i in range(N):
        for j in range(N):
            if (x_cords[j]<=x_cords[i]+1) and (x_cords[i]+1 <= x_cords[j]+1):
                adj_matrix[N+i][j]=1


    print(adj_matrix)
    return adj_matrix.T

def build_dataset(name, N=[2**14, 2**15, 2**16]): #N=[256, 512, 1024, 2048]):
    instances = []
    # N=[ 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4,
    #     2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4,    
    #     2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4,
    #     2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4,    
    #     2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4,
    #     2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4, 2**4]   
    # N=[2**4]*108
    # N=[2**10]*108
    N = [2**7]*12
    N1 = [2**8]*12
    N2 = [2**9]*12
    N3 = [2**7]*12
    N4 = [2**8]*12
    N5 = [2**9]*12
    N6 = [2**7]*12
    N7 = [2**8]*12
    N8 = [2**9]*12
    N.extend(N1)
    N.extend(N2)
    N.extend(N3)
    N.extend(N4)
    N.extend(N5)
    N.extend(N6)
    N.extend(N7)
    N.extend(N8)
    
    for i in N:
        adj_matrix = generate_square_stabbing_problem(i)
        instances.append(adj_matrix)

    pickle.dump(instances, open(DIR+ name +".p", "wb"))

def load_dataset(name):
    instances = pickle.load(open(DIR + name + ".p", "rb"))

    instances = [(i, [0, 1], lines_adj_matrix_greedy_dataset(i)) for i in instances]

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

def lines_adj_matrix_greedy_dataset(instance):
    """ Adjacency matrix for lines  
        A lines is said to be adjacent to another if they have points are stabbed by both
        horizontal and vertical lines (any two lines) two ses of lines stab the same set of 
        squares then one can be eliminated
    """
    lines_adj_matrix = np.zeros((instance.shape[0], instance.shape[0]))

    # a disk is adjacent to itself
    lines_adj_matrix[0, 0] = 1
    lines_adj_matrix[1, 1] = 1
    for i in range(2,instance.shape[0]):
        lines_adj_matrix[i, i] = 1
        
        lines_adj_matrix[0, i] = 1
        lines_adj_matrix[1, i] = 1
        # matrix is symmetric
        lines_adj_matrix[i, 0] = 1
        lines_adj_matrix[i, 1] = 1

    return lines_adj_matrix

if __name__ == "__main__":
    build_dataset("squares")
    # ilp_problem(generate_scp(2**11))
    #compute_solution("greedy_64")