import numpy as np
from utils import system
import pdb
import matplotlib.pyplot as plt
#from ftocp import FTOCP
from nlp import NLP
from matplotlib import rc
from numpy import linalg as la
from mpac_cmd import *
from Version13_dynamicProgramming_multiAgent import *

### Agent=0 is quadruped. Agent=1 is drone

# Set precomputed to True if plan is precomputed and stored in .txt file
precomputed = True

# First observation: 0 or 1 (corresponding to e=1 and e=1, respectively)
obs_choice = 0

# if simulation is True we use quadruped and drone. otherwise, we use euler integration etc. for state updates
simulation = True

# =====================================
# Stand and get initial position

if simulation:
    stand_idqp()
    data = get_tlm_data()
    x = data["q"][0]
    y = data["q"][1]
    theta = data["q"][5]



# =====================================
### Root RRT Parameters ###
if simulation:
    x0_1 = Node(np.array([x, y]))  # Without theta, for high level plan
    x0_2 = Node(np.array([0, 0]))  # Without theta, for high level plan
else:
    x0_1 = Node(np.array([0, 0]))  # Without theta, for high level plan
    x0_2 = Node(np.array([0, 0]))  # Without theta, for high level plan
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

model = Model(RRT_root, N)

if not precomputed:
    model, best_plan = run_MORRT(model)
    plan_ends = flatten_plan(best_plan)
    plan_node_agent1 = plan_ends[0][obs_choice]  # Hard coded, aka "pre-defined observations" for now
    plan_node_agent2 = plan_ends[1][obs_choice]  # Hard coded, aka "pre-defined observations" for now
    print('end1: ' + str(plan_node_agent1.state))
    print('end2: ' + str(plan_node_agent2.state))






# =============================
# Initialize system parameters for two agents
if not simulation:
    x0_agent1 = np.append(x0_1.state, 0)  # With theta, for robot dynamics
    x0_agent2 = np.append(x0_1.state, 0)  # With theta, for robot dynamics
else:
    x0_agent1 = np.append(x0_1.state, theta)  # With theta, for robot dynamics
    x0_agent2 = np.append(x0_1.state, 0)  # With theta, for robot dynamics



dt = 0.1  # Discretization time
sys_agent1 = system(x0_agent1, dt, agent=0)  # initialize system object
sys_agent2 = system(x0_agent2, dt, agent=1)  # initialize system object

maxTime = 100  # Simulation time


# Initialize mpc parameters
N = 15
n = 3
d = 2
Q = 1 * np.eye(n)
R = 1 * np.eye(d)
dR = 0.1 * np.eye(d)
Qf = 1000 * np.eye(n)
# Remove cost of heading angle since high level plan is in xy-space
Q[n - 1, n - 1] = 0
Qf[n - 1, n - 1] = 0

# Define obstacle ellipse
obstacle = obstacles_agent_1[0]
el_x = (obstacle[0][1]+obstacle[0][0])/2
el_y = (obstacle[1][1]+obstacle[1][0])/2
el_ax = (obstacle[0][1]-obstacle[0][0])/2
el_ay = (obstacle[1][1]-obstacle[1][0])/2
ellipse=[el_x, el_y, el_ax, el_ay]

# =================================================================
# ======================== Subsection: Nonlinear MPC ==============
# First solve the nonlinear optimal control problem as a Non-Linear Program (NLP)
printLevel = 1
xub_agent1 = np.array([Xi[0][1], Xi[1][1], 2 * np.pi])
uub_agent1 = np.array([0.15, 0.15])
xub_agent2 = np.array([Xi[0][1], Xi[1][1], 2 * np.pi])
uub_agent2 = np.array([0.15, 0.15])





sys_agent1.reset_IC()  # Reset initial conditions
sys_agent2.reset_IC()  # Reset initial conditions
xPredNLP_agent1 = []
xPredNLP_agent2 = []
eps = 0.1  # TODO: Change to 0.1?



if precomputed:
    # Load pre-computed paths

    # Agent 1
    path_temp_agent1 = open("e" + str(obs_choice+1) + "_Agent1.txt", "r")
    path_agent1 = []
    for row in path_temp_agent1.read().split('\n'):
        row = row.strip()
        if row:
            row = [float(x) for x in row.split(',')]
            path_agent1.append(row)
    path_temp_agent1.close()
    path_agent1 = np.array(path_agent1)
    path_agent1 = array_to_path(path_agent1, RRT_root)

    # Agent 2
    path_temp_agent2 = open("e" + str(obs_choice+1) + "_Agent2.txt", "r")
    path_agent2 = []
    for row in path_temp_agent2.read().split('\n'):
        row = row.strip()
        if row:
            row = [float(x) for x in row.split(',')]
            path_agent2.append(row)
    path_temp_agent2.close()
    path_agent2 = np.array(path_agent2)
    path_agent2 = array_to_path(path_agent2, RRT_root)

    max_hierarchy = 0
else:
    max_hierarchy = 1


hierarchy = 0
while hierarchy <= max_hierarchy:  # TODO: Change to 2 for two observations (or np.inf since we break loop when arriving at end node)
    print('Hierarchy: ' + str(hierarchy))
    # get_plan_node()  # TODO: For now we are fixing observations before-hand

    if not precomputed:
        path_agent1 = return_subpath(plan_node_agent1, hierarchy, agent=0)
        path_agent2 = return_subpath(plan_node_agent2, hierarchy, agent=1)
        goal_node_temp_agent1 = get_goal(path_agent1, agent=0)
        goal_node_temp_agent2 = get_goal(path_agent2, agent=1)

    else:
        goal_node_temp_agent1 = path_agent1[0]
        goal_node_temp_agent2 = path_agent2[0]

    node_number_agent1 = 0
    node_number_agent2 = 0


    goal_agent1 = goal_node_temp_agent1.state.reshape(2, ).copy()
    goal_agent2 = goal_node_temp_agent2.state.reshape(2, ).copy()
    goal_agent1 = np.append(goal_agent1, np.pi)
    goal_agent2 = np.append(goal_agent2, np.pi)
    print('goal1: ' + str(goal_agent1))
    print('goal2: ' + str(goal_agent2))

    # goal = all_goals_test[0] # FOR TESTING

    nlp_agent1 = NLP(N, Q, R, dR, Qf, goal_agent1, dt, xub_agent1, uub_agent1, printLevel, 0, ellipse)
    nlp_agent2 = NLP(N, Q, R, dR, Qf, goal_agent2, dt, xub_agent2, uub_agent2, printLevel, 1, ellipse)

    xt_agent1 = sys_agent1.x[-1]
    xt_agent2 = sys_agent2.x[-1]
    ut_agent1 = nlp_agent1.solve(xt_agent1)
    ut_agent2 = nlp_agent2.solve(xt_agent2)

    dists = np.array([np.linalg.norm(sys_agent1.x[-1][:-1] - goal_agent1[:-1]), np.linalg.norm(sys_agent1.x[-1][:-1] - goal_agent1[:-1])])
    finished = np.array([False, False]).reshape(2,1)

    i=1 # FOR TESTING
    while not finished.all():
        # for _ in range(maxTime):
        xt_agent1 = sys_agent1.x[-1]
        xt_agent2 = sys_agent2.x[-1]
        ut_agent1 = nlp_agent1.solve(xt_agent1, verbose=False)  # compute control input at time t
        ut_agent2 = nlp_agent2.solve(xt_agent2, verbose=False)  # compute control input at time t
        xPredNLP_agent1.append(nlp_agent1.xPred)  # store predicted trajectory at time t
        xPredNLP_agent2.append(nlp_agent2.xPred)  # store predicted trajectory at time t
        sys_agent1.applyInput(ut_agent1)
        sys_agent2.applyInput(ut_agent2)

        dists = np.array([np.linalg.norm(sys_agent1.x[-1][:-1] - goal_agent1[:-1]), np.linalg.norm(sys_agent1.x[-1][:-1] - goal_agent1[:-1])])
        i+=1
        #if i > 2500:
           #break
        # print(i)
        if (dists < eps).any():
            # Agent 1
            if dists[0] < eps:
                if goal_node_temp_agent1 == path_agent1[-1]:
                    finished[0] = True
                    # Stop Agent to wait for other to reach goal state (not optimal solution)
                    if simulation:
                        walk_mpc_idqp()
                    else:
                        # nlp_agent1 = NLP(N, Q, R, dR, Qf, goal_agent1, dt, xub_agent1, np.array([0, 0]), printLevel, 0, ellipse)
                        pass
                else:
                    node_number_agent1 += 1
                    if not precomputed:
                        start_index = path_agent1.index(goal_node_temp_agent1)
                        goal_node_temp_agent1 = get_goal(path_agent1[start_index:], agent=0)
                    else:
                        goal_node_temp_agent1 = path_agent1[node_number_agent1]
                    goal_agent1 = goal_node_temp_agent1.state.reshape(2, ).copy()
                    print('goal1: ' + str(goal_agent1))
                    goal_agent1 = np.append(goal_agent1, np.pi)
                    nlp_agent1 = NLP(N, Q, R, dR, Qf, goal_agent1, dt, xub_agent1, uub_agent1, printLevel, 0, ellipse)
            # Agent 2
            if dists[1] < eps:
                if goal_node_temp_agent2 == path_agent2[-1]:
                    finished[1] = True
                    # Stop Agent to wait for other to reach goal state (not optimal solution)
                    nlp_agent2 = NLP(N, Q, R, dR, Qf, goal_agent2, dt, xub_agent2, np.array([0, 0]), printLevel, 1, ellipse)
                else:
                    node_number_agent2 += 1
                    if not precomputed:
                        start_index = path_agent2.index(goal_node_temp_agent2)
                        goal_node_temp_agent2 = get_goal(path_agent2[start_index:], agent=1)
                    else:
                        goal_node_temp_agent2 = path_agent2[node_number_agent2]
                    goal_agent2 = goal_node_temp_agent2.state.reshape(2, ).copy()
                    print('goal2: ' + str(goal_agent2))
                    goal_agent2 = np.append(goal_agent2, np.pi)
                    nlp_agent2 = NLP(N, Q, R, dR, Qf, goal_agent2, dt, xub_agent2, uub_agent2, printLevel, 1, ellipse)


    hierarchy += 1
    if simulation:  # stop agents
        walk_mpc_idqp()
        lie()
    # if not goal_node_temp.children:  # Plan is finished
        # break

for t in range(0, maxTime):  # Time loop
    xt_agent1 = sys_agent1.x[-1]
    xt_agent2 = sys_agent2.x[-1]
    ut_agent1 = nlp_agent1.solve(xt_agent1, verbose=False)  # compute control input at time t
    ut_agent2 = nlp_agent2.solve(xt_agent2, verbose=False)  # compute control input at time t
    xPredNLP_agent1.append(nlp_agent1.xPred)  # store predicted trajectory at time t
    xPredNLP_agent2.append(nlp_agent2.xPred)  # store predicted trajectory at time t
    sys_agent1.applyInput(ut_agent1)
    sys_agent2.applyInput(ut_agent2)

x_cl_nlp_agent1 = np.array(sys_agent1.x)
x_cl_nlp_agent2 = np.array(sys_agent2.x)

# for timeToPlot in [0, 10]:
 #    plt.figure()
 #    plt.plot(xPredNLP[timeToPlot][:, 0], xPredNLP[timeToPlot][:, 1], '--.b',
  #            label="Predicted trajectory at time $t = $" + str(timeToPlot))
  #   plt.plot(xPredNLP[timeToPlot][0, 0], xPredNLP[timeToPlot][0, 1], 'ok',
          #   label="$x_t$ at time $t = $" + str(timeToPlot))
  #   plt.xlabel('$x$')
   #  plt.ylabel('$y$')
   #  plt.xlim(-1, 12)
   #  plt.ylim(-1, 10)
   #  plt.legend()

### Plot environment and closed loop trajectories
plt.figure()
# Observation Area 1
x_min, x_max = observation_area1.region[0][0], observation_area1.region[0][1]
y_min, y_max = observation_area1.region[1][0], observation_area1.region[1][1]
rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='c', ec="c", alpha=0.5, label='Observation Areas')
plt.gca().add_patch(rectangle)

# Observation Area 2
x_min, x_max = observation_area2.region[0][0], observation_area2.region[0][1]
y_min, y_max = observation_area2.region[1][0], observation_area2.region[1][1]
rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='c', ec="c", alpha=0.5)
plt.gca().add_patch(rectangle)

# Obstacle
x_min, x_max = obstacle[0][0], obstacle[0][1]
y_min, y_max = obstacle[1][0], obstacle[1][1]
rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='k', ec="k", label='Obstacle')
plt.gca().add_patch(rectangle)

# Goal Regions
plt.plot(4.5, 4.5, 'o', color='r', label = 'Goal Regions')
plt.plot(0.5, 4.5, 'o', color='r')
# # plt.plot(0, 10, 'o', color='r', label='e=3')
plt.annotate('e=1', (4.6, 4.6))
plt.annotate('e=2', (0.6, 4.6))


# Plot closed loop trajectories

### TODO: Remember to include horizon length N in name
plt.plot(x_cl_nlp_agent1[:, 0], x_cl_nlp_agent1[:, 1], '-*b', label="Closed-loop trajectory, Agent 1 (Quadruped)")
plt.plot(x_cl_nlp_agent2[:, 0], x_cl_nlp_agent2[:, 1], '-*r', label="Closed-loop trajectory, Agent 2 (Drone)")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(Xi[0])
plt.ylim(Xi[1])
plt.legend()
plt.show()



# # =================================================================
# # =========== Subsection: Sequential Quadratic Programming ========
# # State constraint set X = \{ x : F_x x \leq b_x \}
# Fx = np.vstack((np.eye(n), -np.eye(n)))
# bx = np.array([15,15,15,15]*(2))

# # Input constraint set U = \{ u : F_u u \leq b_u \}
# Fu = np.vstack((np.eye(d), -np.eye(d)))
# bu = np.array([10, 0.5]*2)

# # Terminal constraint set
# Ff = Fx
# bf = bx

# printLevel = 1
# uGuess = [np.array([10, 0.1])]*N

# ftocp = FTOCP(N, Q, R, Qf, Fx, bx, Fu, bu, Ff, bf, dt, uGuess, goal, printLevel)
# ftocp.solve(x0)

# plt.figure()
# plt.plot(xPredNLP[0][:,0], xPredNLP[0][:,1], '-*r', label='Solution from the NLP')
# plt.plot(ftocp.xPred[:,0], ftocp.xPred[:,1], '--ob', label='Solution from one iteration of SQP')
# plt.title('Predicted trajectory')
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.xlim(-1,12)
# plt.ylim(-1,10)
# plt.legend()

# uGuess = []
# for i in range(0, ftocp.N):
# 	uGuess.append(ftocp.uPred[i,:]) # Initialize input used for linearization using the optimal input from the first SQP iteration
# ftocpSQP = FTOCP(N, Q, R, Qf, Fx, bx, Fu, bu, Ff, bf, dt, uGuess, goal, printLevel)
# ftocpSQP.solve(x0)

# plt.figure()
# plt.plot(xPredNLP[0][:,0], xPredNLP[0][:,1], '-*r', label='Solution from the NLP')
# plt.plot(ftocp.xPred[:,0], ftocp.xPred[:,1], '--ob', label='Solution from one iteration of SQP')
# plt.plot(ftocpSQP.xPred[:,0], ftocpSQP.xPred[:,1], '-.dk', label='Solution from two iterations of SQP')
# plt.title('Predicted trajectory')
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.xlim(-1,12)
# plt.ylim(-1,10)
# plt.legend()

# plt.show()

# # =================================================================
# # =========== Subsection: NMPC using an SQP Approach  =============
# sys.reset_IC() # Reset initial conditions
# xPred = []
# for t in range(0,maxTime): # Time loop
# 	xt = sys.x[-1]
# 	ut = ...
# 	ftocpSQP.uGuessUpdate()
# 	xPred.append(ftocpSQP.xPred)
# 	sys.applyInput(ut)

# x_cl = np.array(sys.x)
# u_cl = np.array(sys.u)

# for t in range(0, 6):
# 	plt.figure()
# 	plt.plot(xPredNLP[t][:,0], xPredNLP[t][:,1], '-*r', label='Predicted trajectory using NLP at time $t = $'+str(t))
# 	plt.plot(xPred[t][:,0], xPred[t][:,1], '--.b', label='Predicted trajectory using one iteration of SQPat time $t = $'+str(t))
# 	plt.plot(xPred[t][0,0], xPred[t][0,1], 'ok', label="$x_t$ at time $t = $"+str(t))
# 	plt.xlabel('$x_1$')
# 	plt.ylabel('$x_2$')
# 	plt.xlim(-1,12)
# 	plt.ylim(-1,10)
# 	plt.legend()


# plt.figure()
# for t in range(0, maxTime):
# 	if t == 0:
# 		plt.plot(xPred[t][:,0], xPred[t][:,1], '--.b', label='Predicted trajectory using one iteration of SQP')
# 	else:
# 		plt.plot(xPred[t][:,0], xPred[t][:,1], '--.b')

# plt.plot(x_cl[:,0], x_cl[:,1], '-*r', label='Closed-loop trajectory using one iteration of SQP')
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.xlim(-1,12)
# plt.ylim(-1,10)
# plt.legend()


# plt.figure()
# plt.plot(x_cl_nlp[:,0], x_cl_nlp[:,1], '-*r', label='Closed-loop trajectory using NLP')
# plt.plot(x_cl[:,0], x_cl[:,1], '-ob', label='Closed-loop trajectory using one iteration of SQP')
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.xlim(-1,12)
# plt.ylim(-1,10)
# plt.legend()
# plt.show()
