#!/usr/bin/env python3
import serial
import struct
import time
import datetime
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Define global variables for drone
x = y = z = yaw = 0

def vrpnCallback(data):
    x = data.pose.position.x
    y = data.pose.position.y
    z = data.pose.position.z
    (roll, pitch, yaw) = euler_from_quaternion ([data.pose.orientation.x,data.pose.orientation.y,data.pose.orientation.z,data.pose.orientation.w])
    writeToXbee([x,y,z,yaw])
    rospy.loginfo(yaw)

def writeToXbee(data):
    # print(writeToXbee([1.1, 1.1, 1.1]))
    # data = [1.0, 1.0, 1.0, 1.0]
    f = len(data)
    # print(len(data))

    length = '22' # update this to work with any size message
    header = ['7E', '00', length, '10', '01', '00', '13', 'A2', '00', '41', 'B1', '91', '99', 'FF', 'FE', '00', '00','31','32']
    message = []
    for i in range(0, len(header)):
        message.append(bytearray.fromhex(header[i]))

    for i in range(len(data)):
        message.append(bytearray(struct.pack("f", data[i])))

    end = ['3C','3B']
    for i in range(0, len(end)):
        message.append(bytearray.fromhex(end[i]))

    output = bytearray()
    output = message[0]
    for i in range(1,len(message)):
        output = output + message[i]

    csum = sum(output[3:])
    low = 255 - (csum & 0xff)

    output = output + bytearray(low.to_bytes(1, 'big'))
    # print(low)
    # print(output.hex())

    # print('test writing')


    num = ser.write(output)

    # print(f'number of bytes written: {num}')


global ser
ser = serial.Serial("/dev/ttyUSB0",115200)
rospy.init_node('interface', anonymous=True)
rospy.Subscriber("/vrpn_client_node/RigidBody04/pose", PoseStamped, vrpnCallback)

while not rospy.is_shutdown():
    rospy.spin()

ser.close()



####### Sim code
import numpy as np
from utils import system
import pdb
import matplotlib.pyplot as plt
#from ftocp import FTOCP
from nlp import NLP
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.misc import derivative
from mpac_cmd import *
import time
from helper_functions import *
import os
from datetime import datetime


# Set precomputed to True if plan is precomputed and stored in .txt file
precomputed = True

# First observation: 0 or 1 (corresponding to e=1 and e=1, respectively)
obs_choice = 0

# If save_data is true text files closed loop trajectory, control inputs, solver time, and predicted trajectories are saved
save_data = False

# If simulation is True we use quadruped or drone. otherwise, we use euler integration for state updates etc.
simulation = False



## Import either singleAgent or multiAgent, since they share functions
from Version12_generalizedPlan_oneAgent import *

# =====================================
# Stand and get initial position
if simulation == 'quadruped':
    stand_idqp()
    data = get_tlm_data()
    x_quad = data["q"][0]
    y_quad = data["q"][1]
    theta_quad = data["q"][5]

# =====================================
### Root RRT Parameters ###
if not simulation:
    x0 = Node(np.array([0, 0]))
else:
    x0 = Node(np.array([x_quad, y_quad]))  # Start Node without theta, for high level plan
    
Xi = [[0, 5], [0, 5]]  # Constraint set
xy_cords = [[0, 1]]  # Indices of xy-coordinates
Delta = 0.5  # Incremental distance in RRT
Q = 0.5 * Delta * 1e3 * np.eye(2)  # Stage cost
QN = 1e4 * np.eye(2)  # Terminal cost
xg1 = np.array([4.5, 4.5])  # First partially observable goal state
xg2 = np.array([0.5, 4.5])  # Second partially observable goal state
goal_states = [xg1, xg2]
gamma = 10000  # RRT* radius parameter
eta = 4 * Delta  # RRT* radius parameter
Theta1 = np.array([[1, 0], [0, 0]])  # Observation accuracy matrix
Theta2 = np.array([[0, 0], [0, 1]])  # Observation accuracy matrix
Omega = np.eye(2)  # Partially observable environment transition matrix
b0 = np.array([1 / 2, 1 / 2])  # Initial belief
obstacles = [[[2, 3], [2, 5]]]  # Obstacles
observation_area1 = ObservationArea([[3, 5], [0, 1]], [Theta1, Theta2])  # First observation area
observation_area2 = ObservationArea([[-15, 15], [8, 11]], [Theta1, Theta2])  # Second observation area
observation_areas = [observation_area1]  # TODO: Add observation_area2 for experiment
N = 1000  # Number of nodes for final RRT
N_subtrees = 5  # Number of children of each RRT

# Create the root RRT
RRT_root = RRT(start=x0, Xi=Xi, Delta=Delta, Q=Q, QN=QN, goal_states=goal_states, Omega=Omega, v0=b0,
          star=True, gamma=gamma, eta=eta, obstacles=obstacles, observation_areas=observation_areas,
          N_subtrees=N_subtrees)


# =====================================
### Mixed Observable RRT Model ###
model = Model(RRT_root, N)

if not precomputed:
    model, best_plan = run_MORRT(model)  
    plan_list = flatten_list(best_plan)  
    plan_node = plan_list[obs_choice]  # Hard coded, aka "pre-defined observations" for now  

# =====================================
### Mid Level MPC ###
if not simulation:
    x0 = np.array([0, 0, 0])
else:
    x0 = np.array([x_quad, y_quad, theta_quad])
dt = 0.1  # Discretization time
sys = system(x0, dt, simulation)  # Including theta for mid-level MPC
maxTime = 10  # Simulation time

# Initialize mpc parameters
N = 15
N_MPC = N  # To avoid any confusion/mix-up with N_RRT
n = 3
d = 2
Q = 1 * np.eye(n)
R = 1 * np.eye(d)
dR = 0 * np.eye(d)
# R[1,1] = 1
Qf = 1000 * np.eye(n)
# Remove cost of heading angle since high level plan is in xy-space
Q[n - 1, n - 1] = 0
Qf[n - 1, n - 1] = 0

# =================================================================
# ======================== Subsection: Nonlinear MPC ==============
# First solve the nonlinear optimal control problem as a Non-Linear Program (NLP)


# path = plan_node.RRT.return_path(plan_node)
# plot(plan_node.RRT.xy_cords, path, color='r')



    


sys.reset_IC()  # Reset initial conditions
printLevel = 0
xPredNLP = []

eps = 0.5

    
# goal_node = Node(x0.copy())  # Temporary for first heading angle
xub = np.array([model.root.Xi[0][1], model.root.Xi[1][1], 2 * np.pi])
uub = np.array([0.15, 0.15])

# all_goals_test = [np.array([10, -10, 0]), np.array([5, -5, 0]), np.array([-5, -5, 0]), np.array([-5, 5, 0])] # FOR TESTING

if precomputed:
    # Load pre-computed path
    pathForObs = open("pathFor_e" + str(obs_choice+1) + ".txt", "r")
    path = []
    for row in pathForObs.read().split('\n'):
        row = row.strip()
        if row:
            row = [float(x) for x in row.split(',')]
            path.append(row)
    pathForObs.close()
    path = np.array(path)
    path = array_to_path(path, RRT_root)

    max_hierarchy = 0
else:
    max_hierarchy = 1

hierarchy = 0
while hierarchy <= max_hierarchy:  # TODO: Change to 2 for two observations (or np.inf since we break loop when arriving at end node)
    print('Hierarchy: ' + str(hierarchy))
    # get_plan_node()  # TODO: For now we are fixing observations before-hand

    if not precomputed:
        path = return_subpath(plan_node, hierarchy)  
        goal_node_temp = get_goal(path)

    else:
        goal_node_temp = path[0]

    node_number = 0

    
    goal = goal_node_temp.state.reshape(2, ).copy()
    goal = np.append(goal, np.pi)

    # goal = all_goals_test[0] # FOR TESTING

    nlp = NLP(N, Q, R, dR, Qf, goal, dt, xub, uub, printLevel)
    xt = sys.x[-1]
    ut = nlp.solve(xt)

    dist = np.linalg.norm(sys.x[-1][:-1] - goal[:-1])
    finished = False

    i=1 # FOR TESTING
    while not finished:
        # for _ in range(maxTime):
        xt = sys.x[-1]
        ut = nlp.solve(xt, verbose=False)  # compute control input at time t
        xPredNLP.append(nlp.xPred)  # store predicted trajectory at time t
        sys.applyInput(ut)

        # Compute state_next here for drone, due to ROS 
        if sys.simulation == 'drone':
            x_next = x
            y_next = y
            theta_next = yaw  # TODO: Correct?
            state_next = np.array([x_next, y_next, theta_next])  
            sys.x.append(state_next)

        dist = np.linalg.norm(sys.x[-1][:-1] - goal[:-1])
        i+=1
        #if i > 3000:
           #break
        # print(i)
        if dist < eps:
            if goal_node_temp == path[-1]:
                # if i == 4: # FOR TESTING
                finished = True
            else:
                node_number += 1
                print('New node')
                if not precomputed:
                    start_index = path.index(goal_node_temp)
                    goal_node_temp = get_goal(path[start_index:])
                else:
                    goal_node_temp = path[node_number]
                goal = goal_node_temp.state.reshape(2, ).copy()
                print('goal', goal)

                goal = np.append(goal, np.pi)
                # goal = all_goals_test[i] # FOR TESTING
                nlp = NLP(N, Q, R, dR, Qf, goal, dt, xub, uub, printLevel)
                # i += 1 # FOR TESTING

    # Stop robot until observation is made
    if simulation=='quadruped':
        walk_mpc_idqp()  
        stand_idqp()
        time.sleep(5)
        lie()

    hierarchy += 1

x_cl_nlp = np.array(sys.x)
u_cl_nlp = np.array(sys.u)





# Save data
if save_data:
    now = datetime.now()
 
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S_")

    # Closed Loop Trajectory
    np.savetxt('data/' + dt_string + 'x_cl_nlp.txt', x_cl_nlp) 

    # Control inputs
    np.savetxt('data/' + dt_string + 'u_cl_nlp.txt', u_cl_nlp) 

    # NLP Solver Time
    np.savetxt('data/' + dt_string + 'solverTime.txt', np.array(nlp.solverTime)) 

    # Predicted Trajectories
    pred_temp = open("data/" + dt_string + "xPredNLP.txt", "w")
    for row in xPredNLP:
        row = row.reshape(-1, n)
        np.savetxt(pred_temp, row, delimiter=',')
    pred_temp.close()


# plt.figure()
# for timeToPlot in np.arange(0, 11):
#     plt.figure()
#     plt.plot(xPredNLP[timeToPlot][:, 0], xPredNLP[timeToPlot][:, 1], '--.b',
#              label="Predicted trajectory at time $t = $" + str(timeToPlot))
#     plt.plot(xPredNLP[timeToPlot][0, 0], xPredNLP[timeToPlot][0, 1], 'ok',
#              label="$x_t$ at time $t = $" + str(timeToPlot))
#     plt.xlabel('$x$')
#     plt.ylabel('$y$')
#     plt.xlim(-15, 15)
#     plt.ylim(-15, 15)
#     plt.legend()

# plt.figure()
# for t in range(0, maxTime):
#     if t == 0:
#         plt.plot(xPredNLP[t][:, 0], xPredNLP[t][:, 1], '--.b', label='Predicted trajectory at time $t$')
#     else:
#         plt.plot(xPredNLP[t][:, 0], xPredNLP[t][:, 1], '--.b')


### Plot environment and closed loop trajectory
plt.figure()
# Observation Area
x_min, x_max = observation_area1.region[0][0], observation_area1.region[0][1]
y_min, y_max = observation_area1.region[1][0], observation_area1.region[1][1]
rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='c', ec="c", alpha=0.5, label='Observation Area')
plt.gca().add_patch(rectangle)

# Obstacle
x_min, x_max = obstacles[0][0][0], obstacles[0][0][1]
y_min, y_max = obstacles[0][1][0], obstacles[0][1][1]
rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='k', ec="k", label='Obstacle')
plt.gca().add_patch(rectangle)

# Goal Regions
plt.plot(4.5, 4.5, 'o', color='r', label = 'Goal Regions')
plt.plot(0.5, 4.5, 'o', color='r')
# # plt.plot(0, 10, 'o', color='r', label='e=3')
plt.annotate('e=1', (4.6, 4.6))
plt.annotate('e=2', (0.6, 4.6))


# Closed loop trajectory
plt.plot(x_cl_nlp[:, 0], x_cl_nlp[:, 1], '-*r', label="Closed-loop trajectory")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(model.root.Xi[0])
plt.ylim(model.root.Xi[1])
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(0,len(u_cl_nlp[:,0])), u_cl_nlp[:,0], label = "Velocity")
plt.legend()
plt.show()


plt.figure()
plt.plot(np.arange(0,len(u_cl_nlp[:,1])), u_cl_nlp[:,1], label = "Yaw rate")
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(0, len(nlp.solverTime)), nlp.solverTime, label = "Solver time")
plt.legend()
plt.show()















