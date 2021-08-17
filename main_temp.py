from Version13_dynamicProgramming_multiAgent import *
from helper_functions import *

save_plans = True

# # =================================================================
### Root RRT ###

# Parameters
x0_1 = Node(np.array([0, 0]))
x0_2 = Node(np.array([0, 0]))
starts = [x0_1, x0_2]
Xi = [[0, 5], [0, 5]]
Delta = 0.5
Q = 0.5 * Delta * 1e2 * np.eye(2)
QN = 1e4 * np.eye(2)
xg1 = np.array([4.5, 4.5])
xg2 = np.array([0.5, 4.5])
goal_states = [xg1, xg2]
gamma = 10000 * Delta
eta = 8 * Delta
Theta1 = np.array([[1, 0], [0, 0]])
Theta2 = np.array([[0, 0], [0, 1]])
Theta12 = np.array([[0.9, 0], [0, 0.1]])
Theta22 = np.array([[0.1, 0], [0, 0.9]])
Omega = np.eye(2)
b0 = np.array([0.5, 0.5])
N = 1000
N_subtrees = 5

observation_area1 = ObservationArea([[4, 5], [4, 5]], [Theta1, Theta2])
observation_area2 = ObservationArea([[0, 1], [4, 5]], [Theta1, Theta2])

# Define agent parameters
obstacles_agent_1 = [[[2.5, 3], [2, 5]]]
obstacles_agent_1_plan = [[[2, 3.5], [1.5, 5]]]
obstacles_agent_2 = None

obstacles = {0: obstacles_agent_1_plan, 1: obstacles_agent_2}

observation_areas_agent_1 = [observation_area1, observation_area2]
observation_areas_agent_2 = [observation_area1, observation_area2]

observation_areas = {0: observation_areas_agent_1, 1: observation_areas_agent_2}

# Create the root RRT
RRT_root = RRT(starts=starts, Xi=Xi, Delta=Delta, Q=Q, QN=QN, goal_states=goal_states, Omega=Omega, v0=b0, star=True, gamma=gamma, eta=eta,
                obstacles=obstacles,
                observation_areas=observation_areas, N_subtrees=N_subtrees)

### Mixed Observable RRT model ###
model = Model(RRT_root, N)

# # =================================================================
# Uncomment the following to run and plot

model, best_plan = run_MORRT(model)
# plot_environment(model)
plan_ends = flatten_plan(best_plan)
# plot_agent_plans(model, best_plan)


## Save plans as numpy arrays
if save_plans:
    # Agent 1
    path_e1_Agent1 = return_nodes_to_follow(return_subpath(plan_ends[0][0], 0, agent=0), agent=0) + return_nodes_to_follow(return_subpath(plan_ends[0][0], 1, agent=0), agent=0)
    path_e2_Agent1 = return_nodes_to_follow(return_subpath(plan_ends[0][1], 0, agent=0), agent=0) + return_nodes_to_follow(return_subpath(plan_ends[0][1], 1, agent=0), agent=0)
    path_e1_Agent1 = path_to_array(path_e1_Agent1)
    path_e2_Agent1 = path_to_array(path_e2_Agent1)

    pathForObs_e1 = open("path_e1_Agent1.txt", "w")
    for row in path_e1_Agent1:
        row = row.reshape(1, 2)
        np.savetxt(pathForObs_e1, row, delimiter=',')
    pathForObs_e1.close()

    pathForObs_e2 = open("path_e2_Agent1.txt", "w")
    for row in path_e2_Agent1:
        row = row.reshape(1, 2)
        np.savetxt(pathForObs_e2, row, delimiter=',')
    pathForObs_e2.close()

    # Agent 2
    path_e1_Agent2 = return_nodes_to_follow(return_subpath(plan_ends[1][0], 0, agent=1), agent=1) + return_nodes_to_follow(return_subpath(plan_ends[1][0], 1, agent=1), agent=1)
    path_e2_Agent2 = return_nodes_to_follow(return_subpath(plan_ends[1][1], 0, agent=1), agent=1) + return_nodes_to_follow(return_subpath(plan_ends[1][1], 1, agent=1), agent=1)
    path_e1_Agent2 = path_to_array(path_e1_Agent2)
    path_e2_Agent2 = path_to_array(path_e2_Agent2)

    pathForObs_e1 = open("path_e1_Agent2.txt", "w")
    for row in path_e1_Agent2:
        row = row.reshape(1, 2)
        np.savetxt(pathForObs_e1, row, delimiter=',')
    pathForObs_e1.close()

    pathForObs_e2 = open("path_e2_Agent2.txt", "w")
    for row in path_e2_Agent2:
        row = row.reshape(1, 2)
        np.savetxt(pathForObs_e2, row, delimiter=',') 
    pathForObs_e2.close()
