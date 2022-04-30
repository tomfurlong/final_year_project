from re import A
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from time import time

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

num_features = 5

ACTION_SELECT = -1
ACTION_REMOVE = 1

ACTION_SELECT_IDX = 0
ACTION_REMOVE_IDX = 1

def test_final(train_fn, value_model, training_name, instances_test, init_algorithm, is_final_state, has_embedding=False):
    # filehandle = open("./output/" + training_name + "_test_new" + str(round(time())), "w")
    # filehandle=open("./output/testinginfo_datasets" + str(round(time())), "w")
    filehandle=open("./output/testinginfo_datasets" + training_name, "w")
    agent = AgentTest(value_model)

    agent.action_value_func.load_state_dict(torch.load("./models/" + training_name))
    if has_embedding:
        agent.embedding_model.load_state_dict(torch.load("./models/" + training_name + "_embedding"))

    for (name, ds) in instances_test:
        greedy_ratio_mean = test(filehandle, train_fn, training_name, (name, ds), init_algorithm, is_final_state, agent)
    


def test(filehandle, train_fn, training_name, instances, init_algorithm, is_final_state_fn, agent):
    print("in test")
    num_greedy_solutions = 0
    num_greedy_cardinality = 0
    num_optimal_cardinality = 0
    num_greedy_optimal_card_equal = 0

    optimality_ratios = 0
    greedy_ratios = 0
    total_episode_time = 0
    total_greedy_time = 0

    total_best_cardinality = 0
    total_greedy_cardinality = 0
    total_optimal_cardinality = 0

    total_min_num_lines = 0
    total_min_num_lines_by_lp_relax = 0
    best_steps = 0
    name = instances[0]
    instances = instances[1]
    lines_for_plots = []
    lines_for_plots_best_card = []
    lp_optimal_lines = []
    total_best_cardinality = 0
    total_init_ratio = 0
    total_opt_ratio  = 0



    test_times=[]
    total = len(instances)
    filehandle.write("Information\n")
    # filehandle.write("epi_time,gdy_time,opt_rati,gdy_rati,opt_card,gdy_card,gdy_equi,mean_opt,mean_gdy,mean_app,opt==gdy,best_step,dataset, minumum_num_lines_init_algorithm, best_cardinality, num_lines_from_opt_solution \n")
    # filehandle.write("epi_time, gdy_time, opt==gdy, best_step,"+ training_name +", best_cardinality, num_lines_from_opt_solution, Opt_sol \n")
    filehandle.write("Model,   Opt,  Init,  App,   App/init,   App/Opt\n")

    for instance in instances:
        start_time=time()
        equal_greedy_solution, equal_greedy_cardinality, equal_optimal_cardinality, optimality_ratio, greedy_ratio, greedy_cardinality, equal_greedy_optimal_cardinality, episode_time, greedy_time, best_step, min_num_lines, best_cardinality, optimal_cardinality = train_fn(training_name, instance, [], init_algorithm, is_final_state_fn, 1, agent)
        dif_in_min_lines = min_num_lines-best_cardinality
        # filehandle.write( "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i,%s,%i,%i, %i,      %i,    \n" % (episode_time, greedy_time, optimality_ratio, greedy_ratio, equal_optimal_cardinality, equal_greedy_cardinality, equal_greedy_solution, greedy_cardinality, best_cardinality, equal_greedy_optimal_cardinality, best_step, name, min_num_lines, dif_in_min_lines, best_cardinality, optimal_cardinality))
        # print("\n%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i,%s, %i\n" % (episode_time, greedy_time, optimality_ratio, greedy_ratio, equal_optimal_cardinality, equal_greedy_cardinality, equal_greedy_solution, optimal_cardinality, greedy_cardinality, best_cardinality, equal_greedy_optimal_cardinality, best_step, name, min_num_lines))
        filehandle.write("%s, %f,  %i,  %f,  %f,   %f  \n" % (training_name, optimal_cardinality, min_num_lines, best_cardinality, best_cardinality/min_num_lines, best_cardinality/optimal_cardinality))
        num_greedy_solutions += 1 if equal_greedy_solution else 0
        num_greedy_cardinality += 1 if equal_greedy_cardinality else 0
        num_optimal_cardinality += 1 if equal_optimal_cardinality else 0

        optimality_ratios += optimality_ratio
        greedy_ratios += greedy_ratio

        
        total_greedy_cardinality += greedy_cardinality
        total_best_cardinality += best_cardinality

        num_greedy_optimal_card_equal += 1 if equal_greedy_optimal_cardinality else 0
        total_episode_time += episode_time
        total_greedy_time += greedy_time

        total_init_ratio += best_cardinality/min_num_lines
        total_opt_ratio += best_cardinality/optimal_cardinality
        total_min_num_lines += min_num_lines
        total_best_cardinality += best_cardinality
        total_optimal_cardinality += optimal_cardinality
        lines_for_plots.append(min_num_lines)
        lines_for_plots_best_card.append(best_cardinality)
        lp_optimal_lines.append(optimal_cardinality)
        best_steps += best_step

        test_time = time() - start_time
        test_times.append(test_time)


    if training_name == "FEATURE_Greedy":
        title = "Greedy"
    else :
        title = "Minimal Set Cover" 
    plt.figure()
    # plt.title("Number of minumum lines comparison when using " + title + " initialisation algorithm")
    plt.xlabel("Each instance")
    plt.ylabel("Number of Minumum Lines")
    plt.plot(range(len(lines_for_plots)), lines_for_plots, c="green", label="Initial Minimum Lines")
    plt.plot(range(len(lines_for_plots)), lines_for_plots_best_card, c="blue", label="Minumum Lines using RL")
    plt.plot(range(len(lines_for_plots)), lp_optimal_lines, c="red", label="Minumum Lines Optimal Solution")
    plt.legend()
    plt.savefig("plots/average_min_lines_comp" + training_name + ".png")

    # plt.figure()
    # plt.title("Comparing Optimal solution to RL solution using " + title + " initialisation algorithm")
    # plt.xlabel("Each instance")
    # plt.ylabel("Number of Minumum Lines")
    # plt.plot(range(len(lines_for_plots)), lines_for_plots, c="green", label="Initial Minimum Lines")
    # plt.plot(range(len(lines_for_plots)), lines_for_plots_best_card, c="blue", label="Minumum Lines using RL")
    # plt.legend()
    # plt.savefig("plots/average_min_lines_comp" + training_name + ".png")

    plt.figure()
    plt.title("Test Times")
    plt.ylabel("Time(s)")
    plt.xlabel("Number of instances")
    plt.plot(range(len(test_times)), test_times, c="red")
    plt.savefig("plots/Test_times_plot" + training_name + ".png")


    greedy_pct = num_greedy_solutions / total
    greedy_card_pct = num_greedy_cardinality / total
    optimal_card_pct = num_optimal_cardinality / total
    optimal_ratio_mean = optimality_ratios / total
    greedy_ratio_mean = greedy_ratios / total
    mean_greedy_cardinality = total_greedy_cardinality / total
    mean_best_cardinality = total_best_cardinality / total
    greedy_optimal_card_equal = num_greedy_optimal_card_equal / total
    avg_episode_time = total_episode_time / total
    avg_greedy_time = total_greedy_time / total
    mean_best_step = best_steps / total
    mean_min_num_lines = total_min_num_lines/total
    mean_card = total_best_cardinality/total
    mean_optimal_cardinality = total_optimal_cardinality / total
    mean_opt_ratio = total_opt_ratio/total
    mean_init_ratio = total_init_ratio/total
    accuracy = mean_card/mean_optimal_cardinality
    # print(accuracy)
    # print(acc_min_lines_lp/acc_min_lines)
    filehandle.write("\nAverage Testing information\n")
    filehandle.write("%s, %f,  %i,  %f,  %f,   %f  \n" % (training_name, mean_optimal_cardinality, mean_min_num_lines, mean_card,mean_opt_ratio, mean_init_ratio))

    # filehandle.write("avg_time,gdy_time,opt_rati,gdy_rati,gdy_card,gdy_equi,mean_opt,mean_gdy,mean_app,opt==gdy,best_step,dataset, avg_min_num_lines, avg_min_lines_lp\n%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s,%f,%f,%f\n" % (avg_episode_time, avg_greedy_time, optimal_ratio_mean, greedy_ratio_mean, greedy_card_pct, greedy_pct, mean_greedy_cardinality, mean_best_cardinality, greedy_optimal_card_equal, mean_best_step, name, mean_min_num_lines, mean_card, mean_optimal_cardinality))
    filehandle.write("\nAverage difference from optimal solution\n%f" % (mean_card-mean_optimal_cardinality))
    filehandle.write("\nAccuracy of our model:  %f" % (accuracy))
    filehandle.flush()

    return greedy_ratio_mean


def is_final_state_5greedy(steps, num_disks, greedy_cardinality, set_cover_idx_valid, set_uncover_idx):
    return steps != 0 and steps % ((greedy_cardinality*5)-1) == 0 or len(set_cover_idx_valid) == 0 and len(set_uncover_idx) == 0

def is_final_state_2greedy(steps, num_disks, greedy_cardinality, set_cover_idx_valid, set_uncover_idx):
    return steps != 0 and steps % ((greedy_cardinality*2)-1) == 0 or len(set_cover_idx_valid) == 0 and len(set_uncover_idx) == 0

def is_final_state_3greedy(steps, num_disks, greedy_cardinality, set_cover_idx_valid, set_uncover_idx):
    return steps != 0 and steps % ((greedy_cardinality*3)-1) == 0 or len(set_cover_idx_valid) == 0 and len(set_uncover_idx) == 0

def is_final_state_10greedy(steps, num_disks, greedy_cardinality, set_cover_idx_valid, set_uncover_idx):
    return steps != 0 and steps % ((greedy_cardinality*10)-1) == 0 or len(set_cover_idx_valid) == 0 and len(set_uncover_idx) == 0

def is_final_state_disks_5(steps, num_disks, greedy_cardinality, set_cover_idx_valid, set_uncover_idx):
    return steps != 0 and steps % ((num_disks//5)-1) == 0 or len(set_cover_idx_valid) == 0 and len(set_uncover_idx) == 0

def is_final_state_disks_2(steps, num_disks, greedy_cardinality, set_cover_idx_valid, set_uncover_idx):
    return steps != 0 and steps % ((num_disks//2)-1) == 0 or len(set_cover_idx_valid) == 0 and len(set_uncover_idx) == 0

def is_final_state_disks_4(steps, num_disks, greedy_cardinality, set_cover_idx_valid, set_uncover_idx):
    return steps != 0 and steps % ((num_disks//4)-1) == 0 or len(set_cover_idx_valid) == 0 and len(set_uncover_idx) == 0

def is_final_state_disks(steps, num_disks, greedy_cardinality, set_cover_idx_valid, set_uncover_idx):
    return steps != 0 and steps % ((num_disks)-1) == 0 or len(set_cover_idx_valid) == 0 and len(set_uncover_idx) == 0

def is_final_state_2disks(steps, num_disks, greedy_cardinality, set_cover_idx_valid, set_uncover_idx):
    return steps != 0 and steps % ((num_disks*2)-1) == 0 or len(set_cover_idx_valid) == 0 and len(set_uncover_idx) == 0


def build_valid_remove_v2(N, adj_matrix, set_cover_idx):
    covered_by_one = np.where(N == 1)[0]

    invalid_disks = set({})

    for p in covered_by_one:
        invalid_disks = invalid_disks.union(set(set_cover_idx[np.where(adj_matrix[p, set_cover_idx] == 1)[0]]))

    valid_remove = list(set(set_cover_idx) - invalid_disks)

    return valid_remove

def build_valid_remove_v3(N, covers_idx, set_cover_idx):
    set_cover_idx_valid = []
    for i_disk in set_cover_idx:
        if not np.any(N[covers_idx[i_disk]] == 1):
            set_cover_idx_valid.append(i_disk)

    return set_cover_idx_valid

def update_valid_remove_v2(action, covers_idx, N, set_cover_idx, set_cover_idx_valid):
    if action[1] == -1: # SELECT disk
        invalid_disks = set(set_cover_idx) - set(set_cover_idx_valid)

        for i_disk in invalid_disks: # check if invalid disks (and new disk) are now valid
            if not np.any(N[covers_idx[i_disk]] == 1):
                set_cover_idx_valid.append(i_disk)

    else: # REMOVE disk
        set_cover_idx_valid.remove(action[0])
        remaining_valid_disks = set(set_cover_idx_valid)
        for i_disk in remaining_valid_disks:
            if np.any(N[covers_idx[i_disk]] == 1): # check if valid disks are now invalid. removed disk is invalid
                set_cover_idx_valid.remove(i_disk)

    return set_cover_idx_valid


def build_valid_remove(adj_matrix, set_cover_idx):
    """ Returns list of disk indices that can be removed while keeping all points covered """

    #adj_matrix[:,set_cover_idx].sum(axis=1) == 1

    invalid_remove = []

    # for each point, if only one disk covering it, then, disk cannot be removed
    for p in range(adj_matrix.shape[0]):
        sets_covering_p = []
        sets_covering_p_count = 0
        for d in set_cover_idx:
            if adj_matrix[p,d] == 1:
                sets_covering_p_count += 1
                sets_covering_p.append(d)
            
                if sets_covering_p_count == 2: # if already two disks covering point, then disk can be removed, no need to keep counting
                    break
        if sets_covering_p_count == 1:
            invalid_remove.append(sets_covering_p[0])

    valid_remove = list(set(set_cover_idx) - set(invalid_remove))

    return valid_remove

class Memory:
    """ Replay Memory """
    def __init__(self, num_disks, num_features, embedding_size=16, N=10000, batch_size=64):
        self.N = N
        self.batch_size = batch_size
        self.n = 0
        self.full = False

        self.memory_state = torch.empty(N, num_disks, num_features)
        self.memory_action = torch.empty(N, 2)
        self.memory_next_state = torch.empty(N, num_disks, num_features)
        self.memory_reward = torch.empty(N)
        self.memory_done = torch.empty(N)
        self.memory_set_cover_idx = np.empty(N, dtype=object)
        self.memory_set_uncover_idx = np.empty(N, dtype=object)
        self.memory_embeddings = torch.empty(N, num_disks, embedding_size)
        self.memory_embeddings_sum = torch.empty(N, num_disks, embedding_size) # the embeddings sum repeated num_disks times
        self.memory_cardinality = torch.empty(N, num_disks, 1) # the cardinality_improvement repeated num_disks times
        self.memory_next_cardinality = torch.empty(N, num_disks, 1)
        self.memory_episodes = torch.empty(N)


    def push(self, experience):
        state, action, reward, next_state, done, solution, embeddings_all, cardinality, next_cardinality, episode = experience

        if self.n == self.N:
            self.full = True
            self.n = 0

        set_cover_idx, set_uncover_idx = solution
        embeddings, embeddings_sum = embeddings_all

        self.memory_state[self.n] = torch.Tensor(state).float()
        self.memory_action[self.n] = torch.Tensor(action)
        self.memory_next_state[self.n] = torch.Tensor(next_state).float()
        self.memory_reward[self.n] = reward
        self.memory_done[self.n] = done
        self.memory_set_cover_idx[self.n] = set_cover_idx
        self.memory_set_uncover_idx[self.n] = set_uncover_idx
        self.memory_embeddings[self.n] = embeddings
        self.memory_embeddings_sum[self.n] = embeddings_sum
        self.memory_cardinality[self.n] = cardinality
        self.memory_next_cardinality[self.n] = next_cardinality
        self.memory_episodes[self.n] = episode

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
            sample_embeddings = self.memory_embeddings[sample_idx]
            sample_embeddings_sum = self.memory_embeddings_sum[sample_idx]
            sample_cardinality = self.memory_cardinality[sample_idx]
            sample_next_cardinality = self.memory_next_cardinality[sample_idx]
            sample_episodes = self.memory_episodes[sample_idx]

            sample_set_cover_idx = self.memory_set_cover_idx[sample_idx]
            sample_set_uncover_idx = self.memory_set_uncover_idx[sample_idx]

            return sample_state, sample_action, sample_next_state, sample_reward, sample_done, sample_set_cover_idx, sample_set_uncover_idx, sample_embeddings, sample_embeddings_sum, sample_cardinality, sample_next_cardinality, sample_episodes
        return None

class Embedding(nn.Module):
    def __init__(self, x_in, p_dim=16, seed = 42, T = 3):
        super().__init__()
        torch.manual_seed(seed)
        self.T = T
        self.p_dim = p_dim
        
        self.x_in = nn.Linear(x_in, p_dim) 
        self.u_in = nn.Linear(p_dim, p_dim) 
        
        self.x_out = nn.Linear(p_dim, p_dim) 
        self.u_out = nn.Linear(p_dim, p_dim) 

    def forward(self, x, u):
  
        x_embed = self.x_out(self.x_in(x))
        u_embed = self.u_out(self.u_in(u))
        values = x_embed + u_embed
        return values


def get_embeddings(state, adj_matrix, set_cover_idx, embedding_model):
    """ State == features """ 
    x = torch.zeros((adj_matrix.shape[1], 1)) # num disks
    x[set_cover_idx] = 1

    fc_features = torch.ones(adj_matrix.shape[1], embedding_model.p_dim).detach() *0.01   # num_disks x num_features or p_dim

    # u = torch.matmul(torch.tensor(adj_matrix).float(), torch.tensor(fc_features).float()) # neighbour count (* 0.01)
    # u = torch.matmul(torch.tensor(adj_matrix).float(), torch.tensor(state).float()) # neighbour sum of features, p_dim must be num_features

    embeddings = fc_features

    for _ in range(embedding_model.T):
        embeddings = embedding_model.forward(x, torch.matmul(torch.tensor(adj_matrix).float(), embeddings))

    embeddings = (embeddings-torch.mean(embeddings, dim=0))/torch.std(embeddings, dim=0)

    # embedding sum per node (repetaed N times)
    embeddings_sum = torch.matmul(torch.ones(adj_matrix.shape[1],1), embeddings.mean(dim = 0).view(1,-1))

    return embeddings, embeddings_sum


#def create_dataset():
    # X = uniform_random_points(12)
    # S = fixed_disks(X, 0.2)
    # adj_matrix = build_adj_matrix(X, S)

    # infile = open("scp_problem_instance",'rb')
    # adj_matrix = pickle.load(infile)
    # infile.close()

    # instances = [(adj_matrix, [0, 4, 5, 15])]

def pick_action(action_estimates, set_cover_idx, set_uncover_idx, epsilon=.8):
    s = None
    action_type = None

    pick_set_cover = True
    length_set_cover_idx = len(set_cover_idx)
    if length_set_cover_idx == 0:
        pick_set_cover = False
    pick_set_uncover = True
    length_set_uncover_idx = len(set_uncover_idx)
    if length_set_uncover_idx == 0:
        pick_set_uncover = False
    
    if np.random.rand() < epsilon: # exploration
        if pick_set_uncover or pick_set_cover:
            s = np.random.choice(list(set_cover_idx) + list(set_uncover_idx))

            if s in set_cover_idx:
                action_type = ACTION_REMOVE
            else:
                action_type = ACTION_SELECT

        return (s, action_type)

    else: # exploitation
        if pick_set_cover:
            remove_value, remove_idx = torch.max(action_estimates[set_cover_idx, ACTION_REMOVE_IDX], 0) # we pass parameter 'dim' to get the indices (as with argmax)
        if pick_set_uncover:
            select_value, select_idx = torch.max(action_estimates[set_uncover_idx, ACTION_SELECT_IDX], 0)

        if pick_set_cover and pick_set_uncover:
            if remove_value >= select_value:
                action_type = ACTION_REMOVE
                s = set_cover_idx[remove_idx]
            else:
                action_type = ACTION_SELECT
                s = set_uncover_idx[select_idx]
        elif not pick_set_cover and pick_set_uncover:
            action_type = ACTION_SELECT
            s = set_uncover_idx[select_idx]
        elif pick_set_cover and not pick_set_uncover:
            action_type = ACTION_REMOVE
            s = set_cover_idx[remove_idx]
    
        return (s, action_type)

def take_action(set_cover_idx, set_uncover_idx, action):
    if action[1] == ACTION_SELECT:
        set_cover_idx = np.array(list(set_cover_idx) + [action[0]])
        set_uncover_idx = np.array(list((set(set_uncover_idx) - {action[0]})))
    
    elif action[1] == ACTION_REMOVE:
        set_cover_idx = np.array(list((set(set_cover_idx) - {action[0]})))
        set_uncover_idx = np.array(list(set_uncover_idx) + [action[0]])

    return set_cover_idx, set_uncover_idx

class AgentTest:
    def __init__(self, value_model):
        self.action_value_func = value_model
        self.embedding_model = Embedding(1)
