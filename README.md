Code base for my final year project of using Reinforcement Learning to solve the Stabbing Square Problem.
To run , first run the dataset_square.py to generate instances of the squares and lines being represented by an adj_matrix
then, run the main.py 
Use python3.7

Files:
 - ilp.py: ILP formulation of the Stabbing Square Problem (SSP)
 - lp_ilp.py: LP relaxation of the Stabbing Square Problem
 - lp.py: LP relaxation of a SWAP of lines in the Stabbing Square Problem
 - main.py: main file to load datasets and initialise training agent
 - dataset_greedy.py: dataset generation where the greedy algorithm for SCP utilises O(log n) disks
 - dataset_squares.py dataset generation for lines and squares on a plane
 - minimal_set_cover.py: minimal set cover initialisation algorithm for SSP
 - scp_greedy.py: greedy algorithm for the stabbing sqaure problem
 - utils.py: helper functions
 - rl_feature.py: Reinforcement Learning (RL) Agent whose state is based on hand-crafted features
