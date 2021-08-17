import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation



def flatten_plan(agent_plans):
    """
    Flattens a nested array and returns an array of all objects in 'lst' that are not lists
    (for each agent)
    """
    flattened = {i: [] for i in agent_plans}

    def flatten_lsit_helper(agent, lst):
        for item in lst:
            if type(item) == list:
                flatten_lsit_helper(agent, item)
            else:
                flattened[agent].append(item)

    for agent in agent_plans:
        plan = agent_plans[agent]
        flatten_lsit_helper(agent, plan)

    return flattened


def plot_agent_plans(model, agent_plans):
    """
    Plots plans (all possible routes/paths) for each agent
    """
    agent_plan_nodes = flatten_plan(agent_plans)
    model.draw_plan(agent_plan_nodes, colors=['b', 'r', 'g'])


def plot_all_observations(tree):  # TODO: Must motify to work for multi-agent case...
    """
    Plots all observation nodes in model, starting at model.root and ending at tree
    """
    for node in tree.all_nodes:
        if node.observed:
            plt.plot(node.state[tree.xy_cords[0][0]], node.state[tree.xy_cords[0][1]], 'o', color='k')
    for child in tree.children:
        plot_all_observations(child)


def return_subpath(node, obs_number, agent):
    """
    Returns subpath for an end_node 'node', and hierarchy_number obs_number
    """
    node_temp = node.copy()
    RRT = node_temp.RRT
    while RRT.hierarchy_number != obs_number:
        node_temp = RRT.starts[agent]
        RRT = RRT.parent

    end_node = node_temp
    start_node = RRT.starts[agent]

    path = [end_node]
    curr_node = end_node.parent
    while curr_node != start_node.parent:
        path.append(curr_node)
        curr_node = curr_node.parent
    path.reverse()

    return path


def plot_path(path, style, color):
    """
    :param path: path consisting of nodes
    :param style: plot style, e.g., '--' or ':'
    :param color: color of path in plot
    """
    x_vals = []
    y_vals = []
    for node in path:
        x_vals.append(node.state[0])
        y_vals.append(node.state[1])
    plt.plot(x_vals, y_vals, style, color=color)
    plt.plot(x_vals[-1], y_vals[-1], 'o', color='k')


def animate_paths(agent_all_paths, colors, styles, fig, ax):
    agent_all_nodes = {i: [] for i in range(len(agent_all_paths))}
    agent_path_lengths = {i: [] for i in range(len(agent_all_paths))}

    for agent in range(len(agent_all_paths)):
        for path in agent_all_paths[agent]:
            agent_path_lengths[agent].append(len(path))
            agent_all_nodes[agent] += path

    agents_x_data = {i: [] for i in range(len(agent_all_paths))}
    agents_y_data = {i: [] for i in range(len(agent_all_paths))}

    agents_x_vals = {i: [] for i in range(len(agent_all_paths))}
    agents_y_vals = {i: [] for i in range(len(agent_all_paths))}
    for agent in agent_all_nodes:
        agent_path = agent_all_nodes[agent]
        for node in agent_path:
            agents_x_vals[agent].append(node.state[0])
            agents_y_vals[agent].append(node.state[1])

    # fig, ax = plt.subplots()
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)

    # line, = ax.plot(0, 0)
    # line, = ax.plot(0, 0)
    # ax.plot(0,0)

    def animation_frame(i):
        for agent in range(len(agent_all_paths)):
            agents_x_data[agent].append(agents_x_vals[agent][i])
            agents_y_data[agent].append(agents_y_vals[agent][i])
            ax.plot(agents_x_data[agent], agents_y_data[agent], styles[agent], color=colors[agent])
        # x_data.append(i*10)
        # y_data.append(i)
        # line.set_xdata(x_data)
        # line.set_ydata(y_data)
        # return line,

    return FuncAnimation(fig, func=animation_frame, frames=len(agents_x_vals[0]), interval=1000, repeat=False)


def animate_sub_paths(nodes, colors, styles):
    """
    Nodes consists of end_nodes from agents (must correspond to same observations)
    """
    fig, ax = plt.subplots()

    agent_all_paths = {i: [] for i in range(len(nodes))}

    RRT = nodes[0].RRT
    while RRT.hierarchy_number > 0:
        h = RRT.hierarchy_number
        for agent in range(len(nodes)):
            path = return_subpath(nodes[agent], h, agent)
            agent_all_paths[agent].append(path)
        RRT = RRT.parent
    # Add path for RRT.hierarchy_number = 0 and reverse list to make it in order from hierarchy_number=0 and up
    for agent in range(len(nodes)):
        path = return_subpath(nodes[agent], 0, agent)
        agent_all_paths[agent].append(path)
        agent_all_paths[agent].reverse()

    return animate_paths(agent_all_paths, colors, styles, fig, ax)
    ## Animate all paths
    # for i in range(len(agent_all_paths[0])):
    #     paths = []
    #     for agent in range(len(nodes)):
    #         paths.append(agent_all_paths[agent][i])
    #     print(paths)
    #     animate_paths(paths, colors, styles, fig, ax)

# animate_sub_paths([flattened[0][0], flattened[1][0]], ['r', 'r'], ['--', ":"])
# animate_paths([path1, path2], ['r', 'r'], ['--', ':'])

def run_MORRT(model):
    """
    Returns Mixed Observable RRT model as well as the best plan from the model
    """

    # Generate plans
    model.get_plan()

    # Find the best plan
    best_plan = model.best_plan()

    return model, best_plan

def get_goal(path, agent):
    """
    returns next goal node for MPC, given a "sub-path" to next observation
    """
    index = 0
    start_node = path[index]
    end_node = path[index]
    while not start_node.RRT.obstacle_between(start_node, path[index + 1], agent=agent):
        index += 1
        end_node = path[index]
        if index == len(path) - 1:
            break
    plt.plot(end_node.state[0], end_node.state[1], 'o', color='k')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    print('index', index)
    return end_node


def path_to_array(path):
    """
    :param path: path of nodes
    :return: path in the form of a numpy array with states 
    """
    arr = []
    for node in path:
        arr.append(node.state)
    arr = np.array(arr)
    print(arr)

    return arr

def array_to_path(arr, tree):
    """
    :param arr: path in the form of a numpy array with states
    :return: path of nodes
    """
    arr = arr.tolist()
    path = []
    for coords in arr:
        from Version13_dynamicProgramming_multiAgent import Node
        node_temp = Node(np.array(coords))
        node_temp.RRT = tree
        path.append(node_temp)

    return path

def return_nodes_to_follow(subpath, agent):
    """
    Returns nodes for hardware/sim to follow
    """
    indx = 0
    start = subpath[0]
    path = [start]
    end = subpath[0]

    if len(subpath) == 1:
        finished = True
    else:
        finished = False
    while not finished:
        indx += 1
        end = subpath[indx]
        if indx == len(subpath)-1:
            path.append(end)
            finished = True
        elif start.RRT.obstacle_between(start, subpath[indx+1], agent): 
            path.append(end)
            start = end

    return path



# plt.plot(12, 12, 'o', color='r')
# plt.plot(-12, 12, 'o', color='r')
# # plt.plot(0, 10, 'o', color='r', label='e=3')
# plt.annotate('e=1', (12.3, 12.3))
# plt.annotate('e=2', (-11.7, 12.3))
# plt.legend()
