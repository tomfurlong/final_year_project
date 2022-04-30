
from cProfile import label
import os
os.environ["XPAUTH_PATH"] = "/home/dajwani/software/xpress-mp-server/bin/xpauth.xpr"
import time
import numpy as np
import matplotlib.pyplot as plt
import dataset_squares
from dataset import load_dataset
from minimal_set_cover import min_set_cover
from rl_feature import train as feature_train, ValueModel as feature_value_model
from rl_gnn import train as gnn_train, ValueModel as gnn_value_model
from rl_hybrid import train as hybrid_train, ValueModel as hybrid_value_model
from scp_greedy import greedy
from minimal_set_cover import min_set_cover
from utils import is_final_state_5greedy, is_final_state_2greedy, is_final_state_disks, is_final_state_disks_2, is_final_state_2disks, test_final, num_features

if __name__ == "__main__":
    # instances_500 =  dataset_cpp.load_dataset_pickle("fix_500_0.1_radius")
    # instances_1000 =  dataset_cpp.load_dataset_pickle("fix_1000_0.1")
    # instances_1500 =  dataset_cpp.load_dataset_pickle("fix_1500_0.1")
    # instances_2500 =  dataset_cpp.load_dataset_pickle("fix_2500_0.1")


    instances_squares = dataset_squares.load_dataset("squares")
    
    train_idx = 36
    val_idx = 72

    # print("train")
    # print(instances_squares[:train_idx])
    # print("val")
    # print(instances_squares[train_idx:val_idx])
    # print("test1")
    # print(instances_squares[val_idx:])

    # create more instances
    instances_train = instances_squares[:train_idx]
    instances_val = instances_squares[train_idx:val_idx]
    instances_test = instances_squares[val_idx:]

    ## TRAIN FEATURE Init Cover Min set cover
    train_times_min = []
    for instance in instances_train:
        start_time = time.time()
        feature_train("FEATURE_min_set", instance, [("val_500", instances_val[:10])], min_set_cover, is_final_state_2greedy, num_episodes=50)
        train_time = time.time() - start_time
        train_times_min.append(train_time)


    y=train_times_min
    x=range(len(train_times_min))


    ## TEST FEATURE with Min set cover init
    # training_name = "fix_500_0.1_radius_feature_1648479854" #to load model for testing
    training_name = "FEATURE_min_set"
    ##changing min set cover init
    test_final(feature_train, feature_value_model(num_features), training_name,  [("test_min_set_cover", instances_test)],  min_set_cover, is_final_state_2greedy)
    

    # Plots
    plt.figure()
    plt.title("Train Times")
    plt.ylabel("Time(s)")
    plt.xlabel("Number of instances")
    plt.plot(x, y, c="blue")
    plt.savefig("plots/Train_times_plot.png")


    # TRAIN FEATURE greedy init
    train_times_greedy = []
    for instance in instances_val:
        start_time = time.time()
        feature_train("FEATURE_Greedy", instance, [("val_500", instances_val[:10])], greedy, is_final_state_2greedy, num_episodes=50)
        train_time = time.time() - start_time
        train_times_greedy.append(train_time)


    y1=train_times_greedy
    x=range(len(train_times_greedy))


    ## TEST FEATURE grredy init
    # training_name = "fix_500_0.1_radius_feature_1648479854" #to load model for testing
    training_name = "FEATURE_Greedy"
    ##changing min set cover init
    test_final(feature_train, feature_value_model(num_features), training_name,  [("test_greedy", instances_test)],  greedy, is_final_state_2greedy)


    # Plots
    plt.figure()
    plt.title("Train Times Comparison")
    plt.ylabel("Time(s)")
    plt.xlabel("Number of instances")
    plt.plot(x, y, c="blue", label='Minimal set Cover')
    plt.plot(x,y1, c= "green", label='Greedy')
    plt.legend()
    plt.savefig("plots/Train_times_comparison_plot.png")


    '''Current features 
        1. The lp relaxation of the initial set cover 
        2. The harmonic mean of the number of lines that stab each square in this set
        3. Binary variables {0, 1} indicating if it in the set cover
        4. max number of lines
        5-8. If the line is in the 50,75,25,10 percentiles
        9 . Binary variables {0, 1} indicating if it in the greedy set (not done yet)'''

    '''Experiments
        1. Running time comparison RL vs ILP
        2. Running time comparison Training vs Test'''
        
    ## TRAIN GNN
    # gnn_train("greedy_gnn", instances_train, [("val_greedy", instances_val[:10])], greedy, is_final_state_2greedy, num_episodes=3000)
    ## TEST GNN
    # training_name = "greedy_gnn"
    # test_final(gnn_train, gnn_value_model(num_features-1), training_name, [("test_500", instances_test)],  min_set_cover, is_final_state_2greedy, has_embedding=True)

    ## TRAIN HYBRID
    # hybrid_train("greedy_hybrid", instances_greedy[:1] * 100, [("val_greedy", instances_greedy[:1])], greedy, is_final_state_2greedy, num_episodes=30000)
    ## TEST GNN
    # training_name = ""
    # test_final(hybrid_train, hybrid_value_model(num_features-1), training_name, [( "test_2500", instances_2500)],  min_set_cover, is_final_state_2greedy, has_embedding=True)
