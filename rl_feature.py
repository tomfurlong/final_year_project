"""" Agent with one action (current cover set as part of state) """
from array import array
from calendar import c
from operator import ne
import pickle
import os
from pydoc import ModuleScanner
from minimal_set_cover import min_set_cover
import random
from time import time
from copy import deepcopy
from traceback import walk_stack

import numpy as np
import pandas as pd
import numpy.ma as ma
import torch
import statistics
from torch import nn

import matplotlib.pyplot as plt

from lp import lp_solver
from ilp import ilp_problem
from lp_ilp import lp_relaxation_ssp
from utils import build_valid_remove_v2, update_valid_remove_v2, test, build_valid_remove_v3, pick_action, take_action, num_features, ACTION_REMOVE, ACTION_SELECT, ACTION_REMOVE_IDX, ACTION_SELECT_IDX
from scp_greedy import greedy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

SWAP = {"remove": 3, "select": 2}

num_features = 9 # was 8
# num_features = 10 

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
#hello did this work please say it did
class ValueModel(nn.Module):
    def __init__(self, feature_size, seed = 42):
        super().__init__()
        # self.layer1 = nn.Linear(feature_size, 20) 
        self.layer1 = nn.Linear(num_features, 20) 
        self.layer2 = nn.Linear(20, 40) 
        self.layer3 = nn.Linear(40, 20) 
        self.output = nn.Linear(20, 2) 

        #torch.nn.init.xavier_uniform(self.layer1.weight)

        self.relu = torch.nn.PReLU()

        # LP Initialisation
        # w1 = torch.zeros(self.layer1.weight.shape)
        # w1[0,0] = 1
        # self.layer1.weight = torch.nn.Parameter(w1)
        # self.layer1.bias = nn.Parameter(torch.zeros(self.layer1.bias.shape))

        # w2 = torch.zeros(self.layer2.weight.shape)
        # w2[0,0] = 1
        # self.layer2.weight = torch.nn.Parameter(w2)
        # self.layer2.bias = torch.nn.Parameter(torch.zeros(self.layer2.bias.shape))

        # wo = torch.zeros(self.output.weight.shape)
        # wo[0,0] = 1
        # wo[1,0] = -1
        # self.output.weight = torch.nn.Parameter(wo)

        # bo = torch.zeros(self.output.bias.shape)
        # bo[1] = 1
        # self.output.bias = torch.nn.Parameter(bo)


    def forward(self, features):
        value = self.relu(self.layer1(features[0]))
        value = self.relu(self.layer2(value))
        value = self.relu(self.layer3(value))
        value = self.output(value)
        return value


class Agent:
    def __init__(self, num_squares, learning_rate=0.001, gamma=.9):
        self.memory = Memory(num_squares, num_features)
        self.action_value_func = ValueModel(num_features)
        self.target_action_value_func = ValueModel(num_features)

        self.loss_func = nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.action_value_func.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate
        self.gamma = gamma

    def compute_loss(self, sample):
        state, action, next_state, reward, done, set_cover_idx_valid, set_uncover_idx = sample

        state = torch.tensor(state).type(torch.float).detach()
        next_state = torch.tensor(next_state).type(torch.float).detach()

        curr_Q = self.action_value_func([state])
        curr_Q_action = torch.empty(self.memory.batch_size)

        for i in range(self.memory.batch_size):
            line, action_type = action[i]
            if int(action_type) == ACTION_REMOVE:
                action_type_idx = ACTION_REMOVE_IDX
            else:
                action_type_idx = ACTION_SELECT_IDX
            curr_Q_action[i] = curr_Q[i, int(line), action_type_idx]

        next_Q = self.target_action_value_func([next_state])

        # pick next action maximising Q for batch_size
        max_next_Q = torch.empty(self.memory.batch_size)
        for i in range(self.memory.batch_size):
            line, action_type = pick_action(next_Q[i], set_cover_idx_valid[i], set_uncover_idx[i], epsilon=0)
            if action_type == ACTION_REMOVE:
                action_type_idx = ACTION_REMOVE_IDX
            elif action_type == ACTION_SELECT:
                action_type_idx = ACTION_SELECT_IDX
            else:
                max_next_Q[i] = 0
                continue

            max_next_Q[i] = (next_Q[i, line, action_type_idx])
        expected_Q = reward + (1 - done) * self.gamma * max_next_Q

        loss = self.loss_func(curr_Q_action, expected_Q.detach())
        return loss

    def update_model(self):
        self.action_value_func.train()
        sample = self.memory.sample()
        if sample is None:
            return

        loss = self.compute_loss(sample)

        self.optimiser.zero_grad()
        loss.backward()         
        self.optimiser.step()  

def train(training_name, instances, instances_test, init_algorithm, is_final_state, num_episodes=1, agent=None):
    print("in training")
    lines_squares_matrix = instances[0]
    num_squares = lines_squares_matrix.shape[0]
    num_lines = lines_squares_matrix.shape[1]
    # taking the sum of all the 1s in each column, (each square how many lines stab it), 
    # then look at a particualr line (row), i look at all the values that are 1 in that row, 
    # for all values that are 1 in that row look at correspnding l_i, 
    # the compute the minumum number of lines, then mean, then take 10/25 percentile. 

    # '''Min_set_cover returns the indexs of the optmimal lines that stab all the squares'''
    # min_cover = min_set_cover(lines_squares_matrix)
    # print(lines_squares_matrix)
    # min_num_lines = len(min_cover)
    # print((min_cover))
    # print("Minimum number of lines ", len(min_cover))


    testing = True
    if agent is None:
        agent = Agent(num_lines)
        testing = False
        start_train = time()
        # training_name = training_name + "_" + str(round(start_train))
        filehandle = open("./output/" + training_name , "w")
        filehandle.write("Init Algorithm: %s\n" % init_algorithm.__name__)
        filehandle.write("Train Size: %i | %s \n" % (len(instances), str(instances[0].shape)))
        filehandle.write("Test Sizes: %s\n" % str([(i[0], len(i[1]), str(i[1][0][0].shape)) for i in instances_test]))
        filehandle.write("Termination Function: %s\n" % is_final_state.__name__)
        filehandle.write("Model: \n%s\n" % str(agent.action_value_func))
    torch.save(agent.action_value_func.state_dict(), './models/' + training_name )
    agent.action_value_func.eval()
    

    epsilon = 0.9
    decay_epsilon = 0.999
    min_epsilon = 0.1

    min_greedy_ratio = {}
    total_steps = 0
    rewards = []
    greedy_solutions_cache = {}
    filehandle1 = open("./output/rewards", "w")
    for episode in range(num_episodes):
        print("starting new episode----------------------===================================-0000000000000000000000-0-0-0-0-0-0-0")
        if not testing and episode % 100 == 0 and episode != 0:
            time_elapsed = time() - start_train # training time
            print("made it here")
            df = pd.DataFrame(rewards, columns=["episode_reward"])
            ax = df['episode_reward'].plot(color = 'lightgray')
            df['episode_reward'].rolling(50).mean().plot(color = 'black')
            ax.set_xlabel("Episode")
            plt.ylabel("Rolling Mean (10) Cumulative Return")
            plt.savefig("plots/plot1.png")
            print("made it here 2")
            filehandle.write("\nSteps[%s, %s]: %f\n" %(total_steps, episode, time_elapsed))
            for (name, ds) in instances_test:
                greedy_ratio_mean = test(filehandle, train, training_name, (name, ds), init_algorithm, is_final_state, agent)
                print("made it here 3")
                if name not in min_greedy_ratio or greedy_ratio_mean < min_greedy_ratio[name]:
                    # Save best model per validation dataset
                    print("made it to here4")
                    print("Save Model:", greedy_ratio_mean)
                    torch.save(agent.action_value_func.state_dict(), "./models/" + training_name) # + "_" + name
                    min_greedy_ratio[name] = greedy_ratio_mean
            plt.figure()
            plt.plot(time_elapsed)
            plt.savefig("plots/time_elapsed")
            start_train = time()

        
        current_instance_idx = episode//100 # or => random.choice(range(len(instances[0])))
        instance = instances[current_instance_idx]

        if "greedy" in training_name:
            adj_matrix, optimal_solution, line_adj_matrix, centers = instance # adj_matrix N lines x P points, line_adj_matrix N x N
        else:
            adj_matrix, optimal_solution, centers = instances 
            line_adj_matrix = adj_matrix # lines and squares are equivalent (same position) (FIX dataset)

        if epsilon * decay_epsilon > min_epsilon: 
            epsilon *= decay_epsilon

        steps = 1

        
        print("ilp solver")
        lp_optimal_sol, lp_obj_val = ilp_problem(adj_matrix)
        print(lp_obj_val)
        print(adj_matrix)

        min_set_cover_times = []
        '''Min_set_cover returns the indexs of the optmimal lines that stab all the squares'''
        min_set_cover_time = time()
        min_cover = min_set_cover(lines_squares_matrix)
        min_num_lines = len(min_cover)
        print((min_cover))
        print("Minimum number of lines ", len(min_cover))
        min_set_cover_time = time() - min_set_cover_time
        min_set_cover_times.append(min_set_cover_time)
        print("MIN TIME ---- ", min_set_cover_time)

        # print(adj_matrix)
        greedy_times = []
        greedy_time = time()
        if current_instance_idx not in greedy_solutions_cache: # If instance greedy solution not in cache, compute initial set cover
            greedy_solutions_cache[current_instance_idx] = np.array(init_algorithm(adj_matrix))
        # print(greedy_solutions_cache[current_instance_idx])
        # set_cover_idx = np.copy(greedy_solutions_cache[current_instance_idx])         
        set_cover_idx = np.copy(greedy_solutions_cache[current_instance_idx])         

        greedy_time = time() - greedy_time
        # print("GREEDY TIME ---- ", greedy_time)
        greedy_times.append(greedy_time)
        set_uncover_idx = np.array(list(set(range(num_lines)) - set(set_cover_idx))) # R' = S - R (complement)
        set_uncover_idx_cp = set_uncover_idx


        # plt.figure()
        # plt.plot(range(len(min_set_cover_times)), min_set_cover_times, c='b')
        # plt.plot(range(len(greedy_times)), greedy_times, c='r')
        # plt.savefig('plots/greedy_vs_min_set')

        greedy_set_cover = set_cover_idx
        greedy_cardinality = set_cover_idx.shape[0] #no of lines that all squares with greedy algorithm
        # print(greedy_cardinality)

        start_episode = time()
        L = adj_matrix.sum(axis=1)
        # each sqaure s is stabbed by L[s] lines
        S = adj_matrix.sum(axis=0) # each line l covers  P[d] points
        # list of lines l with associated list of squares stabbed
        covers = np.ones((num_lines, num_squares))
        covers_idx = []
        covers_length = np.zeros(num_lines)
        # number of neighbouring squares for each line l
        # i.e. a line l stabs N squares: sum the number of lines that cover those N squares
        neighbouring_lines = np.zeros(adj_matrix.shape[1])
        
        # num squares at less than 2, 4, 8 distance (i.e. num hops/lines until we reach it) d1 - d2 - d3 => d3 is at distance 2 
        print(num_lines, num_squares)
        for l in range(num_lines):
            cover = np.where(adj_matrix[:,l] == 1)[0]
            # print(cover)
            covers[l, cover] = 0
            covers_idx.append(cover)
            covers_length[l] = cover.shape[0] # degree
            neighbouring_lines[l] = sum(L[cover])

        '''Neighbouring lines is a list that keeps track of the number of squares the each line stabs (I think) '''
        # print(covers_idx)
        # print(neighbouring_lines) 

        #num_second_neighbouring_points = covers_length + [np.sum(covers_length[n]) for n in covers_idx] #TODO covers will be covers for lines when lines and points are not the same

        #neighbouring_lines = (neighbouring_lines - neighbouring_lines.min())/(neighbouring_lines.max() - neighbouring_lines.min())

        #num_second_neighbouring_points = (num_second_neighbouring_points - num_second_neighbouring_points.min())/(num_second_neighbouring_points.max() - num_second_neighbouring_points.min())

        # should be size one for each line
        lp_relaxation_lines = []
        lp_relaxation = lp_relaxation_ssp(adj_matrix)

        if testing: 
            for pos, i in enumerate(lp_relaxation):
                if i==1.0 or i==0.5:
                    lp_relaxation_lines.append(pos)
            print("-----------------------------0000000000000--------------------",lp_relaxation_lines)
        #changing num_second_neighbouring_points to 1 just for the moment
        state, set_cover_idx_valid, N = build_state(lp_relaxation, adj_matrix, covers, covers_idx, covers_length, neighbouring_lines, centers, set_cover_idx, min_num_lines)
        set_cover_idx_valid_cp = deepcopy(set_cover_idx_valid)
        # x, y = lp_solver(set_cover_idx, set_uncover_idx, SWAP)
        # print("lp_solver")
        
        # print(covers)
        # print(covers_idx)
        # print(set_cover_idx)
        # print(set_cover_idx_valid)
        # print(adj_matrix)
        # print(x)
        # print(y)
        cardinality = set_cover_idx.shape[0]
        best_cardinality = cardinality
        # print("best cardinality")
        # print(set_cover_idx)
        best_cover = greedy_set_cover
        best_step = 0 
        done = 0
        prev_cardinality = greedy_cardinality
        while not done:
            # Estimate value of state, action
            state_copy = np.copy(state)
            state = torch.tensor(state).type(torch.float)
            with torch.no_grad():
                action_estimates = agent.action_value_func([state])

            # Pick action
            if testing:
                action = pick_action(action_estimates, set_cover_idx_valid, set_uncover_idx, epsilon=0)
            else:
                action = pick_action(action_estimates, set_cover_idx_valid, set_uncover_idx, epsilon=epsilon)
            
            set_uncover_idx = set_uncover_idx_cp
            set_cover_idx_valid = set_cover_idx_valid_cp
            # Update state
            set_cover_idx, set_uncover_idx = take_action(set_cover_idx, set_uncover_idx, action)
            #changed num_second_neighbouring_points to one for the moment
            state, set_cover_idx_valid = update_state(lp_relaxation, adj_matrix, covers, covers_idx, covers_length, neighbouring_lines, centers, set_cover_idx, N, action, set_cover_idx_valid, min_num_lines)
            
            # print(state, set_cover_idx_valid)

            set_uncover_idx_cp = set_uncover_idx
            set_cover_idx_valid_cp = deepcopy(set_cover_idx_valid)
            if action[1] == ACTION_SELECT:
                set_cover_idx_valid.remove(action[0])
            else:
                set_uncover_idx = list(set_uncover_idx)
                set_uncover_idx.remove(action[0])
                
            
            # Receive reward
            reward = cardinality - set_cover_idx.shape[0]
            cardinality = set_cover_idx.shape[0]
            # print("CARD:", cardinality)

            # if sum(adj_matrix[:,set_cover_idx].sum(axis=1) == 0) > 0:
            #     # One point is not covered
            #     reward -= 10e4
        
            # print("\r" + "[%s,%s,%s]: %s" % (total_steps, episode, steps, cardinality), end='')
            # print("\r" + "[%s,%s,%s]: %s %s ==> %s" % (total_steps, episode, steps, cardinality,best_cardinality, str(sorted(set_cover_idx))), end='')
            if cardinality < greedy_cardinality:
                print("")

            if cardinality < best_cardinality:
                best_cardinality = cardinality
                best_cover = set_cover_idx
                best_step = steps
            
            print("\r" + "[%s,%s,%s]: card %s  best %s init %s opt %s==> %s\n" % (total_steps, episode, steps, cardinality,best_cardinality, greedy_cardinality, lp_obj_val , str(sorted(set_cover_idx))), end='')
            
            rewards.append(prev_cardinality-cardinality)
            prev_cardinality = cardinality
            if steps % (greedy_cardinality*2) == 0:
                # rewards.append(greedy_cardinality - cardinality)
                filehandle1.write("\nEpisode: %i\n" % episode)
                filehandle1.write("\nReward:\n")
                filehandle1.write(str(rewards))                
                episode_time = time() - start_episode

                done = 1
                if testing:
                    print("------------------testing----------------------")

                    optimal_cardinality = lp_obj_val
                    equal_greedy_solution = set(greedy_set_cover) == set(best_cover)
                    equal_greedy_cardinality = best_cardinality == greedy_cardinality
                    equal_optimal_cardinality = best_cardinality == optimal_cardinality
                    equal_greedy_optimal_cardinality = greedy_cardinality == optimal_cardinality
                    
                    # print()
                    # print("A", str(best_cardinality))
                    # print("G", equal_greedy_cardinality, greedy_cardinality, equal_greedy_solution)
                    # print("O", equal_optimal_cardinality, str(optimal_cardinality))

                    optimality_ratio = best_cardinality/optimal_cardinality
                    greedy_ratio = best_cardinality/greedy_cardinality

                    return equal_greedy_solution, equal_greedy_cardinality, equal_optimal_cardinality, \
                        optimality_ratio, greedy_ratio, greedy_cardinality, \
                             equal_greedy_optimal_cardinality, episode_time, greedy_time, best_step, min_num_lines, best_cardinality, optimal_cardinality

            if not testing:
                # Memorise s, s', action, reward for REPLAY
                # if cardinality < greedy_cardinality:
                #     agent.memory.push((state_copy, action, reward, state, 1, (set_cover_idx_valid, set_uncover_idx)))

                # reward depends on action type
                agent.memory.push((state_copy, action, action[1], state, done, (set_cover_idx_valid, set_uncover_idx)))
                
                # Perform gradient descent every C steps
                if total_steps % 4 == 0:
                    agent.update_model()
                    agent.action_value_func.eval()
                # Copy model to target model every X steps
                if total_steps % 100 == 0:
                    agent.target_action_value_func.load_state_dict(agent.action_value_func.state_dict())
            
            steps +=1
            total_steps +=1
            print("Total Steps:", total_steps)

'''method to see if the majority of the lines going through a certain square 
    are horzizontal or vertical'''
    #pseudo code for the moment
    # def hor_or_ver_square(l, s):
    #     if l is vertical and most lines through s is vertical :
    #         return 1
    #     elif l is hor and most line through s is hor: 
    #         return 1
    #     else :
    #         return 0

def minimise_optmimal_lines(optimal_lines, squares_covered):
    new_optimal_lines = []

    for idx, line  in enumerate(optimal_lines):
        new_optimal_lines.append(line)
        for pos, entry in enumerate(line):
            if entry == 1 and squares_covered[pos]==0:
               squares_covered[pos] = 1
        if all(i == 1 for i in squares_covered):
            print("all values are 1")
            break
    
    # new_optimal_lines, squares_covered = check_min_lines(new_optimal_lines, squares_covered)
    return new_optimal_lines, squares_covered

def percentile(lines):
    '''I want to use this method to get the percentile of the lines 
    Im going to do this to show more centre lines so the 25% percentile 
    will show the 75% lines closest to the centre'''
    # the difference of the 75th and the 25th percentile value gives you the Inter Quartile Range
    np.percentile(lines, [10, 25])


def build_state(lp_relaxation, adj_matrix, covers, covers_idx, covers_length, neighbouring_lines, centers, set_cover_idx, min_num_lines):
    """ Given an instance of an SCP build a state """
    state = np.zeros((adj_matrix.shape[1], num_features))
    # state = np.zeros((adj_matrix.shape[0], len(lp_relaxation)))
    print("hello")
    # if len(set_uncover_idx) != 0 and len(set_cover_idx) != 0:
    #     set_uncover = adj_matrix[:, set_uncover_idx]
    #     set_cover = adj_matrix[:,set_cover_idx]
    #     x, y = lp_solver(set_cover, set_uncover, SWAP)

    #     state[set_cover_idx,0] = -1 * x
    #     state[set_uncover_idx,0] = y

    # Lp relaxation for each line (i)
    # print(adj_matrix)
    # print(set_cover_idx)
    state[:,0] = lp_relaxation
    # it was 
    # state[:,0] = lp_relaxation
    state[:,1] = lp_relaxation
    # state[:,2] = centers[:, 1]

    # print(set_cover_idx)
    # aggregations on N
    N = adj_matrix.sum(axis=1) # each line l stabs a square by N[l] lines in current set cover
    # N is an array showing the amount of squears each line stabs 
    
    # # P_set_cover = adj_matrix[:,set_cover_idx].sum(axis=0) # each line
    #  l covers  P[d] points in current set cover

    # # P_all = np.zeros(adj_matrix.shape[1]) # number of lines in current cover set that cover points covered by line d
    # # for p in range(adj_matrix.shape[0]):
    # #     for l in range(adj_matrix.shape[0]):
    # #         if adj_matrix[p,d] == 1:
    # #             P_all[d] += N[p]

    # # D1: P1(3), P2(4), P3(1)
    '''Covers is opposite of adj matrix'''
    N_cover = ma.array(np.dot(np.ones((adj_matrix.shape[1], 1)),N.reshape((1, adj_matrix.shape[0]))).T, mask=covers)
    # print(N_cover)
    '''Percentiles'''

    # state[:,3] = np.min(N_cover, axis=1)
    # state[:,3] = normalise_column(state[:,3])
    # state[:,4] = np.percentile(N_cover.compressed(), 50)
    # state[:,4] = normalise_column(state[:,4])
    # state[:,5] = np.max(N_cover, axis=1)
    # state[:,5] = normalise_column(state[:,5])
    # state[:,6] = np.percentile(N_cover.compressed(), 75)
    # state[:,6] = normalise_column(state[:,6])
    # state[:,7] = np.percentile(N_cover.compressed(), 25)
    # state[:,7] = normalise_column(state[:,7])
    # state[:,8] = np.percentile(N_cover.compressed(), 90)
    # state[:,8] = normalise_column(state[:,8])
    # state[:,9] = min_num_lines


    ''' Current features 
        1. The lp relaxation of the initial set cover 
        2. The harmonic mean of the number of lines that stab each square in this set
        3. Binary variables {0, 1} indicating if it in the set cover
        4. max number of lines
        5-8. If the line is in the 50,75,25,10 percentiles
        9 . Binary variables {0, 1} indicating if it in the greedy set (not done yet)'''

    # print("Harmonic mean")
    h_mean = round(statistics.harmonic_mean(N), 2)
    # print(statistics.harmonic_mean(N))
    state[:,2] = h_mean # the harmonic mean of the number of lines that stab each square in the current set.
    state[:,3] = in_set_cover(adj_matrix, set_cover_idx)  # Binary variables {0, 1} indicating if it in the set cover 
    state[:,3] = normalise_column(state[:,3])
    state[:,4] = np.sum(adj_matrix, axis=0) # returns the number of squares each line stabs 
    state[:,5] = np.percentile(adj_matrix, 50, axis=0, interpolation='higher') # for each line returns 1 if it stabs more than 50% of the squares, 0 if less that 50% and 0.5 if it stabs exactly half the lines
    state[:,6] = np.percentile(adj_matrix, 75, axis=0, interpolation='higher') # for each line returns 1 if it stabs more than 25% of the squares, 0 if less that 25% and 0.25 if it stabs exactly 25% of the lines
    state[:,7] = np.percentile(adj_matrix, 25, axis=0, interpolation='higher') # for each line returns 1 if it stabs more than 75% of the squares, 0 if less that 75% and 0.75 if it stabs exactly 75% of the lines
    state[:,8] = np.percentile(adj_matrix, 10, axis=0, interpolation='higher') # if line is in 10th percentile so if it stabs 90% of the squares 
    # state[:,9] = min_num_lines

    # print(state)

    ''' The general formula for calculating a harmonic mean is: 
        Harmonic mean = n / (∑1/x_i)
        
        Where:

        n - the number of the values in a dataset
        x_i - the point in a dataset   '''
    
    # state[:,4] = covers_length / np.sum(1/N_cover, axis=1) #harmonic_mean_line
    # state[:,4] = normalise_column(state[:,4])
    # for l in range(adj_matrix.shape[1]):
    #     N_l = N[covers_o[d]]
    #     state[l,1] =  min(N_l) # min_line
    #     state[l,2] = N_l.shape[0] / np.sum(1/N_l) # harmonic_mean_line
    #     p_25 = np.percentile(N[cover], 25)
    #     p_75 = np.percentile(N[cover], 75)
    #     state[l,2] = p_25
    #     state[l,3] = p_75
    
    # state[:,-1] = neighbouring_lines
    # state[:,-1] = num_second_neighbouring_points

    # # std = N.std()

    # median = np.percentile(N, 50)
    # p_75 = np.percentile(N, 75)
    # p_25 = np.percentile(N, 25)
    # p_1 = np.percentile(N, 1)
    # p_99 = np.percentile(N, 99)

    # min_n = min(N)
    # max_n = max(N)

    # state[:,1] = min_n
    # state[:,3] = p_25
    # state[:,3] = p_1
    # state[:,3] = median
    # state[:,4] = p_75
    # state[:,6] = p_99
    # state[:,3] = adj_matrix.shape[0] / np.sum(1/N) # harmonic_mean
    # # state[:,3] = P / adj_matrix.shape[0]
    # state[set_cover_idx,3] = 1
    # state[:,-1] = (covers_length - covers_length.min())/(covers_length.max() - covers_length.min()) # degree

    set_cover_idx_valid = build_valid_remove_v3(N, covers_idx, set_cover_idx)
    return state, set_cover_idx_valid, N

def update_state(lp_relaxation, adj_matrix, covers, covers_idx, covers_length, neighbouring_lines, centers, set_cover_idx, N, action, set_cover_idx_valid, min_num_lines):
    state = np.zeros((adj_matrix.shape[1], num_features))
    
    # if len(set_uncover_idx) != 0 and len(set_cover_idx) != 0:
    #     set_uncover = adj_matrix[:, set_uncover_idx]
    #     set_cover = adj_matrix[:,set_cover_idx]
    #     x, y = lp_solver(set_cover, set_uncover, SWAP)

    #     state[set_cover_idx,0] = -1 * x
    #     state[set_uncover_idx,0] = y

    # it was 
    state[:,0] = lp_relaxation
    state[:,1] = lp_relaxation
    
    # aggregations on N # each point p is covered by N[p] lines in current set cover
    N[covers_idx[action[0]]] -= action[1]


    # # P_set_cover = adj_matrix[:,set_cover_idx].sum(axis=0) # each line l covers  P[d] points in current set cover

    # # P_all = np.zeros(adj_matrix.shape[1]) # number of lines in current cover set that cover points covered by line d
    # # for p in range(adj_matrix.shape[0]):
    # #     for l in range(adj_matrix.shape[0]):
    # #         if adj_matrix[p,d] == 1:
    # #             P_all[d] += N[p]

    # # D1: P1(3), P2(4), P3(1)
    N_cover = ma.array(np.dot(np.ones((adj_matrix.shape[1], 1)),N.reshape((1, adj_matrix.shape[0]))).T, mask=covers)

    # was
    # state[:,3] = np.min(N_cover, axis=1)
    # state[:,3] = normalise_column(state[:,3])
    # state[:,4] = np.percentile(N_cover.compressed(), 50)
    # state[:,4] = normalise_column(state[:,4])
    # state[:,5] = np.max(N_cover, axis=1)
    # state[:,5] = normalise_column(state[:,5])
    # state[:,6] = np.percentile(N_cover.compressed(), 75)
    # state[:,6] = normalise_column(state[:,6])
    # state[:,7] = np.percentile(N_cover.compressed(), 25) # little difference
    # state[:,7] = normalise_column(state[:,7])
    # state[:,8] = np.percentile(N_cover.compressed(), 90)
    # state[:,8] = normalise_column(state[:,8])
    # state[:,9] = min_num_lines
    # state[:,4] = covers_length / np.sum(1/N_cover, axis=1) # harmonic_mean_line
    # state[:,4] = normalise_column(state[:,4])
    h_mean = round(statistics.harmonic_mean(N), 2)

    state[:,2] = h_mean # the harmonic mean of the number of lines that stab each square in the current set.
    state[:,3] = in_set_cover(adj_matrix, set_cover_idx)  # Binary variables {0, 1} indicating if it in the set cover 
    state[:,3] = normalise_column(state[:,3])
    state[:,4] = np.sum(adj_matrix, axis=0) # returns the number of squares each line stabs 
    state[:,5] = np.percentile(adj_matrix, 50, axis=0, interpolation='higher') # for each line returns 1 if it stabs more than 50% of the squares, 0 if less that 50% and 0.5 if it stabs exactly half the lines
    state[:,6] = np.percentile(adj_matrix, 75, axis=0, interpolation='higher') # for each line returns 1 if it stabs more than 25% of the squares, 0 if less that 25% and 0.25 if it stabs exactly 25% of the lines
    state[:,7] = np.percentile(adj_matrix, 25, axis=0, interpolation='higher') # for each line returns 1 if it stabs more than 75% of the squares, 0 if less that 75% and 0.75 if it stabs exactly 75% of the lines
    state[:,8] = np.percentile(adj_matrix, 10, axis=0, interpolation='higher') # if line is in 10th percentile so if it stabs 90% of the squares 
    # state[:,9] = min_num_lines

    # # for l in range(adj_matrix.shape[1]):
    # #     N_d = N[covers_o[d]]
    # #     state[d,1] =  min(N_d) # min_line
    # #     state[d,2] = N_d.shape[0] / np.sum(1/N_d) # harmonic_mean_line
    #     # p_25 = np.percentile(N[cover], 25)
    #     # p_75 = np.percentile(N[cover], 75)
    #     # state[d,2] = p_25
    #     # state[d,3] = p_75
    
    # state[:,-1] = neighbouring_lines
    # state[:,-1] = num_second_neighbouring_points

    # # std = N.std()

    # # median = np.percentile(N, 50)
    # # p_75 = np.percentile(N, 75)
    # # p_25 = np.percentile(N, 25)
    # # p_1 = np.percentile(N, 1)
    # # p_99 = np.percentile(N, 99)


    # # min_n = min(N)
    # # max_n = max(N)

    # # state[:,1] = min_n
    # # state[:,3] = p_25
    # # state[:,3] = p_1
    # # state[:,3] = median
    # # state[:,4] = p_75
    # # state[:,6] = p_99
    # # state[:,3] = adj_matrix.shape[0] / np.sum(1/N) # harmonic_mean
    # # state[:,3] = P / adj_matrix.shape[0]
    # # state[set_cover_idx,3] = 1
    # state[:,-1] = (covers_length - covers_length.min())/(covers_length.max() - covers_length.min()) # degree


    set_cover_idx_valid = update_valid_remove_v2(action, covers_idx, N, set_cover_idx, set_cover_idx_valid)

    return state, set_cover_idx_valid

def normalise_column(col):
    """ Min Max normalisation [0,1]"""
    col_min = col.min()
    col_max = col.max()
    if col_min == col_max:
        return 0
    else:
        return (col - col_min)/(col_max - col_min)

def percentile_normalise_column(col):
    """ Returns 0 if not in percentile 1 if it is"""
    if col>0:
        return 1
    else: 
        return 0

def in_set_cover(adj_matrix, set_cover_idx):
    ans = [] 
    for idx, val  in enumerate(adj_matrix.T):
        if idx in set_cover_idx:
            ans.append(1)   
        else:
            ans.append(0)
    return ans

class Memory:
    """ Replay Memory """
    def __init__(self, num_lines, num_features, N=10000, batch_size=128):
        self.N = N
        self.batch_size = batch_size
        self.n = 0
        self.full = False

        self.memory_state = torch.empty(N, num_lines, num_features)
        self.memory_action = torch.empty(N, 2)
        self.memory_next_state = torch.empty(N, num_lines, num_features)
        self.memory_reward = torch.empty(N)
        self.memory_done = torch.empty(N)
        self.memory_set_cover_idx = np.empty(N, dtype=object)
        self.memory_set_uncover_idx = np.empty(N, dtype=object)


    def push(self, experience):
        state, action, reward, next_state, done, solution = experience

        if self.n == self.N:
            self.full = True
            self.n = 0

        set_cover_idx, set_uncover_idx = solution

        self.memory_state[self.n] = torch.Tensor(state).float()
        self.memory_action[self.n] = torch.Tensor(action)
        self.memory_next_state[self.n] = torch.Tensor(next_state).float()
        self.memory_reward[self.n] = reward
        self.memory_done[self.n] = done
        self.memory_set_cover_idx[self.n] = set_cover_idx
        self.memory_set_uncover_idx[self.n] = set_uncover_idx

        self.n += 1

    def sample(self):
        memory_size = self.N if self.full else self.n
        if memory_size >= self.batch_size:
            sample_idx = np.random.choice(np.arange(memory_size), self.batch_size)

            sample_state = self.memory_state[sample_idx]
            sample_action = self.memory_action[sample_idx]
            sample_next_state = self.memory_next_state[sample_idx]
            sample_reward = self.memory_reward[sample_idx]
            sample_done = self.memory_done[sample_idx]

            sample_set_cover_idx = self.memory_set_cover_idx[sample_idx]
            sample_set_uncover_idx = self.memory_set_uncover_idx[sample_idx]

            return sample_state, sample_action, sample_next_state, sample_reward, sample_done, sample_set_cover_idx, sample_set_uncover_idx
        return None