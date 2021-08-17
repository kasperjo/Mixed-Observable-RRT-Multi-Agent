import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import scipy.special as func
import cProfile
import itertools
import time
from helper_functions import *

a = time.time()

np.random.seed(0)


class Model:
    def __init__(self, root, N_iters):
        """
        :param root: The initial Rapidly-Exploring Randon Tree. From Class RRT
        :param N_iters: Number of nodes to add for final RRT, when there are no more observations to be made
        """
        self.root = root
        self.N_iters = N_iters
        self.all_RRTs = [root]
        self.N_subtrees = self.root.N_subtrees
        self.N_agents = len(self.root.starts)

    def draw_plan(self, end_nodes, colors):
        """
        Draws a plan. One color for each agent

        end_nodes: dictionary with arrays of all possible plan end_nodes (for each agent)
        colors: N_agent different colors
        """
        for agent, nodes in enumerate(end_nodes.values()):
            for node in nodes:
                # self.root.draw_path_from_node(node, color=colors[agent], label='Agent ' + str(agent))  TODO: add back to this
                self.root.draw_path_from_node(self, node, color=colors[agent], agent=agent)

    def best_plan(self, tree=None):
        """
        This function returns the best plan using a depth first search and dynamical programming approach.
        The best plan is returned as a nested array with end nodes.
        self.get_plan() must be run first in order to generate the possible plans.
        """
        if tree is None:
            tree = self.root

        if not tree.children:

            agent_plans = {}
            agent_costs = {}

            for agent in range(len(tree.starts)):
                end_nodes, costs = tree.return_end_nodes(agent, only_lowest_costs=True)

                if tree == self.root:
                    agent_plans[agent] = end_nodes
                    agent_costs[agent] = costs

                else:
                    # sum the costs for each plan, and pair end nodes for each plan
                    costs_temp = []
                    plans = []

                    for i in range(tree.N_goal_states ** (tree.hierarchy_number - 1)):
                        cost_temp = costs[tree.N_goal_states * i:tree.N_goal_states * (i + 1)]
                        cost_temp_sum = np.sum(cost_temp)
                        costs_temp.append(cost_temp_sum)

                        plan_temp = end_nodes[tree.N_goal_states * i:tree.N_goal_states * (i + 1)]
                        plans.append(plan_temp)

                    costs = costs_temp

                    agent_plans[agent] = plans
                    agent_costs[agent] = costs

            if tree == self.root:
                return agent_plans
            else:
                return agent_plans, agent_costs

        else:  # tree has children
            agent_child_plans, agent_child_costs = {i: [] for i in range(len(tree.starts))}, {i: [] for i in
                                                                                              range(len(tree.starts))}
            for child in tree.children:
                child_plan, child_cost = self.best_plan(child)

                for agent in range(len(tree.starts)):
                    agent_child_plans[agent].append(child_plan[agent])
                    agent_child_costs[agent].append(child_cost[agent])

            agent_no_obs_ends, agent_no_obs_costs = {}, {}
            for agent in range(len(tree.starts)):
                no_obs_end, no_obs_cost = tree.return_end_nodes(agent, only_lowest_costs=True)
                agent_no_obs_ends[agent] = no_obs_end
                agent_no_obs_costs[agent] = no_obs_cost

            if tree == self.root:

                agent_best_child_costs = {i: np.inf for i in range(len(tree.starts))}
                agent_best_child_plans = {i: None for i in range(len(tree.starts))}

                for i in range(
                        len(agent_child_plans[0])):  # All agents have same length plan, so we can look at agent 0
                    child = tree.children[i]

                    all_agent_costs = {}

                    for agent in range(len(tree.starts)):
                        cost = agent_child_costs[agent][i]

                        obs_node = child.starts[agent]
                        cost_to_obs = obs_node.parent.path_costs.copy() + obs_node.parent.node_costs.copy()

                        if (len(cost_to_obs) != 1) or (len(cost) != 1):
                            print('Error1')

                        cost = np.array(cost) + np.array(cost_to_obs)

                        all_agent_costs[agent] = np.sum(cost)  # np.sum to remove list/array format

                    if np.sum(list(all_agent_costs.values())) < np.sum(
                            list(agent_best_child_costs.values())):  # Sum to remove list items
                        for agent in range(len(tree.starts)):
                            agent_best_child_costs[agent] = all_agent_costs[agent]
                        for agent in range(len(tree.starts)):
                            agent_best_child_plans[agent] = agent_child_plans[agent][i].copy()

                # Remove list so that cost is not [cost] for no_obs
                for agent in range(len(tree.starts)):
                    agent_no_obs_costs[agent] = np.sum(agent_no_obs_costs[agent])

                if np.sum(list(agent_best_child_costs.values())) < np.sum(list(agent_no_obs_costs.values())):
                    return agent_best_child_plans

                else:
                    return agent_no_obs_ends

            else:  # tree is not root

                agent_final_plans = {agent: [[] for _ in range(tree.N_goal_states ** (tree.hierarchy_number - 1))] for
                                     agent in range(len(tree.starts))}
                agent_final_costs = {agent: [[] for _ in range(tree.N_goal_states ** (tree.hierarchy_number - 1))] for
                                     agent in range(len(tree.starts))}

                for i in range(tree.N_goal_states ** (tree.hierarchy_number - 1)):
                    for j in range(tree.N_goal_states):

                        all_agent_costs_no_obs = {}
                        all_agent_plans_no_obs = {}

                        best_child_cost = {i: np.inf for i in range(len(tree.starts))}
                        best_child_plan = {}

                        for agent in range(len(tree.starts)):
                            end_no_obs, cost_no_obs = agent_no_obs_ends[agent][i + j], agent_no_obs_costs[agent][i + j]

                            all_agent_costs_no_obs[agent] = cost_no_obs
                            all_agent_plans_no_obs[agent] = end_no_obs

                        for k in range(len(agent_child_plans[0])):  # All agent has same len child plans
                            child = tree.children[k]
                            all_agent_costs = {}
                            for agent in range(len(tree.starts)):
                                # plan = agent_child_plans[agent][k][i + j]
                                cost = agent_child_costs[agent][k][i + j]

                                obs_node = child.starts[agent]
                                cost_to_obs = obs_node.parent.path_costs[0][i + j].copy() + \
                                              obs_node.parent.node_costs[0][
                                                  i + j].copy()

                                cost = np.array(cost) + np.array(cost_to_obs)

                                all_agent_costs[agent] = np.sum(cost)

                            if np.sum(list(all_agent_costs.values())) < np.sum(list(best_child_cost.values())):
                                best_child_cost = all_agent_costs

                                for agent in range(len(tree.starts)):
                                    best_child_plan[agent] = agent_child_plans[agent][k][i + j]

                        if np.sum(list(best_child_cost.values())) < np.sum(list(all_agent_costs_no_obs.values())):
                            for agent in range(len(tree.starts)):
                                agent_final_costs[agent][i].append(best_child_cost[agent])
                                agent_final_plans[agent][i].append(best_child_plan[agent])
                        else:
                            for agent in range(len(tree.starts)):
                                agent_final_costs[agent][i].append(all_agent_costs_no_obs[agent])
                                agent_final_plans[agent][i].append(all_agent_plans_no_obs[agent])

                ## "Merge" final_costs
                agent_costs_temp = {i: [] for i in range(len(tree.starts))}

                for agent in range(len(tree.starts)):
                    # agent_costs_temp[agent].append(np.sum(agent_final_costs[agent])) TODO: Remove this line? This is wrong?
                    for cost in agent_final_costs[agent]:
                        agent_costs_temp[agent].append(np.sum(cost))

                agent_final_costs = agent_costs_temp

                return agent_final_plans, agent_final_costs

    # def plot_plan(self, end_nodes):  TODO: Remove?
    #     self.root.draw_region(obstacles)
    #     self.root.draw_region(observation_areas)
    #
    #     paths = []
    #
    #     for end_node in end_nodes:
    #         paths.append(self.root.return_path(end_node))
    #
    #     for path in paths:
    #         plot(self.root.xy_cords, path, 'm')
    #
    #     plt.xlim(self.root.Xi[0])
    #     plt.ylim(self.root.Xi[1])

    def build_tree(self, tree):
        """
        get_plan() helper function
        """
        if len(tree.observed_areas[0]) == len(tree.observation_areas[
                                                  0]):  # Since all agents make the same observation it is sufficient to look at agent 0
            for agent in range(len(tree.starts)):
                for _ in trange(self.N_iters):
                    tree.add_node(agent)
        else:  # Grow tree until enough observations have been made
            # while len(tree.observations[agent]) < tree.N_subtrees:
            #     tree.add_node(agent)

            best_ends_for_obs_all_agents = []
            for agent in range(self.N_agents):
                obs_nodes_temp = tree.find_routes_to_observations(
                    agent)  # This is a dictionary. Note that this function builds a tree and finds best path to each observation area
                obs_nodes = []  # Convert to list
                for node in obs_nodes_temp.values():
                    obs_nodes.append(node)
                obs_nodes.append(tree.starts[agent])  # Also allow for agent not moving
                best_ends_for_obs_all_agents.append(obs_nodes)

            all_comb_observations = list(itertools.product(*best_ends_for_obs_all_agents))

            for obs_node_comb in all_comb_observations:
                if (not None in obs_node_comb) and (not self.too_short(obs_node_comb)):
                    self.get_child_plan(tree, obs_node_comb)

    def too_short(self, node_list):
        """
        get_plan() helper function

        To not allow observation directly
        """
        for node in node_list:
            if node.path_length < 1:
                return True
        return False

    def get_plan(self, tree=None):
        """
        Creates a sample of plans
        """
        if tree is None:
            tree = self.root

        self.build_tree(tree)

    def get_child_plan(self, tree, start_nodes):
        """
        get_plan() helper function
        """
        start_nodes = list(start_nodes)
        min_path_length = np.inf
        min_path_node = None
        for node in start_nodes:
            if node.path_length < min_path_length:
                min_path_length = node.path_length
                min_path_node = node

        observed_area = min_path_node.observation_area

        for i, node in enumerate(start_nodes):
            if node != min_path_node:
                # Update cost and belief vectors
                node_temp = node.copy()
                node_temp = get_node_with_path_length(min_path_length, node_temp)
                node_temp.path_costs, node_temp.terminal_costs, node_temp.node_costs = tree.compute_costs(node_temp,
                                                                                                          observation=True,
                                                                                                          area=min_path_node.observation_area)

                node_temp.observed = True
                start_nodes[i] = node_temp

        # for node in tree.starts:  # TODO: (REMOVE?) This is to fix self.compute_costs bug for this scenario
        #     if node in start_nodes:
        #         return None

        # Only count the cost from observation
        for i, node in enumerate(start_nodes):
            node.path_costs = np.zeros(node.path_costs.shape).copy()
            node.path_length = 0

        RRT_temp = RRT(start_nodes, tree.Xi, tree.Delta, tree.Q, tree.QN, tree.goal_states, tree.Omega,
                       min_path_node.vs, star=tree.star, gamma=tree.gamma,
                       eta=tree.eta,
                       obstacles=tree.obstacles,
                       observation_areas=tree.observation_areas.copy(), N_subtrees=tree.N_subtrees)

        for agent in range(len(RRT_temp.starts)):
            ## If observation is perfect, we do not want to make more observations since contradictiong observations would not make sense
            if observed_area.perfect_obs:
                RRT_temp.observed_areas[agent] = tree.observation_areas[agent].copy()
            else:
                RRT_temp.observed_areas[agent] = tree.observed_areas[agent].copy()
                RRT_temp.observed_areas[agent].append(observed_area)  # TODO: add back to this...
            # RRT_temp.observed_areas[agent] = tree.observation_areas[agent].copy()

        # Set initialized to true, since we have vs
        RRT_temp.initialized = True

        RRT_temp.hierarchy_number = tree.hierarchy_number + 1
        self.get_plan(RRT_temp)

        RRT_temp.parent = tree

        self.all_RRTs.append(RRT_temp)
        tree.children.append(RRT_temp)


class Node:
    """Node of Mixed Observable Rapidly-Exploring Random Tree"""

    def __init__(self, state, parent=None, children=None, RRT=None, path_length=0):
        """
        :param state: state of the node
        :param parent: parent node
        :param children: array of child nodes
        :param RRT: the RRT that the node belongs to
        :param path_length: the length of a path starting at Node.RRT.start and ending at node
        """
        if children is None:
            children = []
        self.vs = []  # An array of all unnormalized belief vectors
        self.RRT = RRT
        self.dim = state.reshape(-1, ).shape[0]
        self.state = state.reshape(self.dim, 1)
        self.parent = parent
        self.children = children

        if not self.vs:
            self.path_costs = np.zeros((1, 1))
            self.node_costs = np.zeros((1, 1))
            self.terminal_costs = np.zeros((1, 1))
        else:
            self.path_costs = np.zeros((1, RRT.N_goal_states * len(self.vs)))
            self.node_costs = np.zeros((1, RRT.N_goal_states * len(self.vs)))
            self.terminal_costs = np.zeros(
                (1, RRT.N_goal_states * len(self.vs)))

        self.observed = False  # True if observation is made at node
        self.observation_node = None  # Keep track of which "Parent node" made the observation higher up in the tree. Does not include start node cost
        self.path_length = path_length
        self.observation_area = None  # Keep track of where observation was made

    def copy(self):
        """
        Returns copy of node object, so as to not alter the copied node's parameters
        """
        node_new = Node(self.state.copy(), self.parent, self.children.copy(), self.RRT, self.path_length)
        node_new.vs = self.vs.copy()
        node_new.RRT = self.RRT
        node_new.observed = self.observed
        node_new.observation_node = self.observation_node
        node_new.observation_area = self.observation_area

        return node_new


def get_node_with_path_length(k, end_node):
    """
    Returns node, with path_length k, higher up in tree
    """
    if k > end_node.path_length:
        return None
    elif k == end_node.path_length:
        return end_node
    else:
        node_temp = end_node
        while k < node_temp.path_length:
            node_temp = node_temp.parent
        return node_temp.copy()


class RRT:
    """RRT class. Separate RRTs are built simultaneously for each agent"""

    def __init__(self, starts, Xi, Delta, Q, QN, goal_states, Omega, v0, star=True, gamma=None,
                 eta=None, obstacles=None,
                 observation_areas=None, hierarchy_number=0, N_subtrees=1):
        """
        :param starts: root of RRT. A list of nodes belonging to class Node (one start node for each agent)
        :param Xi: array on form [[x1_min, x1_max],...,[xn_min,xn_max]], defining state constraints of nodes in RRT
        :param Delta: incremental distance of RRT
        :param Q: quadratic state cost
        :param QN: quadratic terminal cost
        :param goal_states: list of possible partially observable goal states, e.g., [xg1, xg2]. Store the state of a goal as a numpy column vector
        :param Omega: transition probability matrix of partially obsetvable environment states
        :param v0: initial belief vector(s). The belief is the same for all agents (aka, there is only one v0)
        :param star: use RRT* algorithm if star=True. Use standard RRT
        :param gamma: parameter for RRT* radius. Only applicable if star=True
        :param eta: max radius of RRT* ball. Radius then shrinks as a function of gamma. Only applicable if star=True
        :param obstacles: obstacles for agent. On form = [[[x_min, x_max], [y_min], y_max]], ...] square obstacles in xy-space
        :param observation_areas: array of ObservationArea Class objects, with all areas where the agent can make observations
        :param hierarchy_number: the depth in the tree of RRTs where self is. E.g. if self the parent of self has a parent which is the root RRT, then hierarchy_number=2 (root->parent->self)
        :param N_subtrees: number of child RRTs to initialize from observation nodes. Aka number of observations to make in RRT before initializing a new one
        """
        self.starts = {i: starts[i] for i in range(len(starts))}
        self.dim = starts[0].state.reshape(-1, ).shape[0]
        self.Xi = Xi
        self.Delta = Delta
        self.all_nodes = {i: [starts[i]] for i in range(len(starts))}
        self.all_edges = {i: [] for i in range(len(starts))}
        self.all_paths = {i: [] for i in range(len(starts))}
        self.dim = len(self.Xi)
        self.Q = Q
        self.QN = QN
        self.goal_states = []
        for g in goal_states:
            self.goal_states.append(g.reshape(self.dim, 1))  # store goal states as column vectors
        self.star = star
        self.gamma = gamma
        self.eta = eta
        self.Omega = Omega

        try:  # A single belief vector (prior to any observation)
            self.v0 = v0.reshape(v0.reshape(-1, ).shape[0], 1)  # Make v0 a column vector
        except:  # More than one belief vector (aftern an observation)
            vs_temp = []
            for v in v0:
                vs_temp.append(v.reshape(v.reshape(-1, ).shape[0], 1))  # Make v a column vector
            self.v0 = vs_temp

        self.initialized = False
        self.obs_made = {i: 0 for i in range(len(starts))}
        self.obstacles = obstacles
        self.xy_cords = [[0, 1]]
        self.observation_areas = observation_areas
        self.observations = {i: [] for i in range(len(starts))}  # On form [[observation_node, observation_area],...]
        self.children = []  # Keep track of children (from observations)
        self.observed_areas = {i: [] for i in
                               range(len(starts))}  # Each child RRT can not observe an already observed area
        self.hierarchy_number = hierarchy_number
        self.parent = None
        self.N_subtrees = N_subtrees  # Keep track of number of allowed observations
        self.N_goal_states = len(self.goal_states)

        self.best_ends = None
        self.lowest_costs = np.inf

    def find_routes_to_observations(self, agent):
        """
        Returns the shortest path to each observation area (for each agent separately)
        if a path is found. A path will be found to each observation area if
        N_subtrees is high enough and an agent is allowed to reach the area in its action space
        """
        best_ends_for_observations = {i: None for i in range(len(
            self.observation_areas[
                agent]))}  # We want to find best route to each observation area, given that we have found any
        best_costs = {i: np.inf for i in range(len(self.observation_areas[agent]))}
        while len(self.observations[agent]) < self.N_subtrees:
            self.add_node(agent)

        observations_made = {i: [] for i in range(
            len(self.observation_areas[agent]))}  # {i : [observation_nodes]} for each observation area

        for node in self.all_nodes[agent]:
            if node.observed:
                area = node.observation_area
                for area_index in range(len(self.observation_areas[agent])):
                    if area == self.observation_areas[agent][area_index]:
                        observations_made[area_index].append(
                            node)  # Append node to observation area "area_index" to keep track of which nodes were observed where

        for i in observations_made.keys():
            all_obs_nodes = observations_made[i]
            for node_temp in all_obs_nodes:
                cost_temp = np.sum(node_temp.path_costs.copy() + node_temp.terminal_costs.copy())
                if cost_temp < best_costs[i]:
                    best_costs[i] = cost_temp
                    best_ends_for_observations[i] = node_temp

        return best_ends_for_observations

    def add_node(self, agent):
        """
        Adds node to RRT
        """
        if not self.initialized:
            for start in self.starts.values():
                start.RRT = self
                start.cost = 0
                start.vs = [self.v0]
            self.initialized = True
        x_new = np.zeros((self.dim, 1))
        for i in range(self.dim):
            x_min = self.Xi[i][0]
            x_max = self.Xi[i][1]
            x_new[i, 0] = np.random.uniform(low=x_min, high=x_max)
        rand_node = Node(x_new)

        parent = self.find_nearest_node(rand_node, agent)
        # Do nothing more if obstacle is in the way
        if self.obstacle_between(rand_node, parent, agent):
            return None
        new_node = self.generate_new_node(parent, rand_node)
        # Update costs at new node
        new_node.parent = parent  # Temporarily set parent in order for return_node_number to work
        k, observation, area = self.get_observation(new_node, agent)
        new_node.RRT = self  # Need this in self.compute_costs
        new_node.path_costs, new_node.terminal_costs, new_node.node_costs = self.compute_costs(new_node, observation,
                                                                                               area)
        # RRT*
        if (self.star) and (
                observation is None):  # TODO: Do not alter observation node??? Could probably include, but must make sure to keep observation
            ## RRT-star
            neighbors = self.find_neighbors(new_node, agent)

            # Find best parent neighbor
            cost = np.sum(new_node.path_costs)
            for neighbor in neighbors:
                cost_temp = neighbor.path_costs.copy() + neighbor.node_costs.copy()
                if (np.sum(cost_temp) < cost) and (not neighbor.observed) and (
                        not self.obstacle_between(new_node, neighbor, agent)):
                    parent = neighbor
                    cost = np.sum(cost_temp)

            # Do nothing more if obstacle is in the way
            if self.obstacle_between(new_node, parent, agent):
                return None

            # Update costs at new node
            new_node.parent = parent  # Temporarily set parent in order for return_node_number to work
            new_node.path_costs, new_node.terminal_costs, new_node.node_costs = self.compute_costs(new_node,
                                                                                                   observation,
                                                                                                   area)

        # Update hierarchy
        parent.children.append(new_node)
        self.all_nodes[agent].append(new_node)
        self.all_edges[agent].append([parent, new_node])

        new_node.path_length = new_node.parent.path_length + 1

        # RRT*
        if (self.star) and (
                observation is None):
            ## RRT-star
            for neighbor in neighbors:
                curr_costs = neighbor.path_costs.copy()
                new_costs_temp = new_node.path_costs.copy() + new_node.node_costs.copy()
                if (np.sum(new_costs_temp) < np.sum(curr_costs)) and (not neighbor.observed) and (
                        not neighbor.parent.observed) and not self.obstacle_between(new_node,
                                                                                    neighbor,
                                                                                    agent):
                    # Change hierarchy
                    neighbor.parent.children.remove(neighbor)
                    self.all_edges[agent].remove([neighbor.parent, neighbor])
                    neighbor.parent = new_node
                    self.all_edges[agent].append([new_node, neighbor])
                    neighbor.path_costs, neighbor.terminal_costs, neighbor.node_costs = self.compute_costs(neighbor,
                                                                                                           None,
                                                                                                           None)
                    new_node.children.append(neighbor)
                    neighbor.path_length = neighbor.parent.path_length + 1

        # Track nodes since reset
        if observation is not None:
            self.obs_made[agent] += 1
            new_node.observed = True
            new_node.observation_area = area
            if len(self.observations[agent]) < self.N_subtrees:
                self.observations[agent].append([new_node, area])

        boolean, obs_node = self.observation_in_path(new_node, agent)
        new_node.observation_node = obs_node

    def find_neighbors(self, node, agent):
        """
        Returns an array of node neighbors for RRT* algorithm
        """
        neighbors = []
        n_nodes = len(self.all_nodes[agent])
        for node_temp in self.all_nodes[agent]:
            Vd = np.pi ** (self.dim / 2) / func.gamma(self.dim / 2 + 1)
            radius = min((self.gamma / Vd * np.log(n_nodes) / n_nodes) ** (1 / self.dim), self.eta)
            if np.linalg.norm(node_temp.state - node.state) < radius:
                neighbors.append(node_temp)
        return neighbors

    def return_end_nodes(self, agent, only_lowest_costs=False):
        """
        :param lowest_cost: if True, only return end_nodes with lowest cost of one of outfalls
        :return: end nodes of RRT. Aka, returns nodes that do not have any children
        """
        end_nodes = []
        best_ends = [None for _ in range(self.N_goal_states ** self.hierarchy_number)]
        lowest_costs = [np.inf for _ in range(self.N_goal_states ** self.hierarchy_number)]

        for node in self.all_nodes[agent]:
            if (not node.children) and (not self.observation_in_path(node, agent)[0]):
                if only_lowest_costs:
                    for i in range(len(lowest_costs)):
                        cost_temp = node.path_costs[0, i].copy() + node.terminal_costs[0, i].copy()
                        # xg = self.goal_states[i % 2] # TODO: Fix this. What is best way...?
                        # if (cost_temp < lowest_costs[i]) and (not self.obstacle_between(node, Node(xg), agent)):  # Only consider end nodes that do not have an obstacle between
                        if cost_temp < lowest_costs[i]:
                            lowest_costs[i] = cost_temp
                            best_ends[i] = node
                else:
                    end_nodes.append(node)
        if only_lowest_costs:
            return best_ends, lowest_costs
        else:
            return end_nodes, lowest_costs

    def get_observation(self, node, agent):
        """
        :return: -1, True, observation area (if observation is made at node)
                 -1, None, None if no observation is made
        -1 is just a 'left-over' parameter from the single agent case. It is not important here...
        """
        observed, area = self.is_inside(node, self.observation_areas, agent)
        if observed:
            if (not self.observation_in_path(node, agent)[0]) and (
                    area not in self.observed_areas[agent]):
                for observation in self.observations[agent]:
                    node_temp = observation[0]
                    if np.linalg.norm(node.state - node_temp.state) < 0:
                        return -1, None, None
                print('Observation made')
                return -1, True, area
            else:
                return -1, None, None
        else:
            return -1, None, None

    def observation_in_path(self, node, agent):
        """
        Returns True if there is an observation at a previous node in the path starting at self.starts[agent]
        """
        node_temp = node
        while node_temp != self.starts[agent]:
            if node_temp.observed:
                return True, node_temp
            node_temp = node_temp.parent
        return False, None

    def get_C(self, observation, area):
        """
        Helper function for updating unnormalized belief vectors
        """
        if observation:
            C = []
            for Theta in area.Thetas:
                C.append(Theta @ self.Omega)
        else:
            C = [self.Omega]
        return C

    def get_vs(self, node, C):
        """
        Returns unnormalized belief vectors
        """
        vs_parent = node.parent.vs.copy()
        vs = []

        for v in vs_parent:
            for c in C:
                vs.append(c @ v)
        return vs

    def compute_costs(self, node, observation=None, area=None):
        """Computes cost at node (not include terminal cost) cost_internal,
        as well as terminal cost cost_terminal
        """
        if (node.RRT.hierarchy_number == 0) and (
                node in node.RRT.starts.values()):  # in case an observation is made immediately (root node has no parent)
            return node.path_costs, node.terminal_costs, node.node_costs

        C = self.get_C(observation, area)
        node.vs = self.get_vs(node, C)

        # Compute node, and terminal costs
        h = []
        hN = []
        for i in range(self.N_goal_states):
            h.append(self.cost_h(node, self.goal_states[i]))
            hN.append(self.cost_hN(node, self.goal_states[i]))

        path_costs = node.parent.path_costs.copy() + node.parent.node_costs.copy()

        N_vs = len(node.vs)
        node_costs = []
        terminal_costs = []
        for i in range(N_vs):
            node_costs.append(np.dot(h, node.vs[i]))
            terminal_costs.append(np.dot(hN, node.vs[i]))
        node_costs = np.array(node_costs).reshape((1, N_vs))
        terminal_costs = np.array(terminal_costs).reshape((1, N_vs))

        if path_costs.shape[1] == node_costs.shape[1] / self.N_goal_states:
            path_costs_temp = np.zeros(node_costs.shape)
            for i in range(int(node_costs.shape[1] / self.N_goal_states)):
                for j in range(self.N_goal_states):
                    path_costs_temp[0, self.N_goal_states * i + j] = path_costs[0, i].copy() + node_costs[
                        0, self.N_goal_states * i + j].copy()
            path_costs = path_costs_temp.copy()

        return path_costs, terminal_costs, node_costs

    def find_nearest_node(self, rand_node, agent):
        """
        Returns the RRT-node closest to the node rand_node
        """
        nearest = None
        distance = np.inf
        for node in self.all_nodes[agent]:
            dist_temp = np.linalg.norm(node.state - rand_node.state)
            if dist_temp < distance:
                nearest = node
                distance = dist_temp

        return nearest

    def is_inside(self, node, constraint, agent):
        """
        :param constraint: obstacles or observation_areas
        :param node:
        :param agent: the index of agent to check constraint for
        :return: True if node is inside of constraint. Also returns the specific area which the node is inside
        """
        for cords in self.xy_cords:
            x = node.state[cords[0]]
            y = node.state[cords[1]]
            if constraint == self.obstacles:
                if not self.obstacles[agent]:
                    return False, None
                for area in self.obstacles[agent]:
                    if (area[0][0] <= x <= area[0][1]) and (area[1][0] <= y <= area[1][1]):
                        return True, area
            elif constraint == self.observation_areas:
                if not self.observation_areas[agent]:
                    return False, None
                if self.observation_areas[agent] is not None:
                    for observation_area in self.observation_areas[agent]:
                        if (observation_area.region[0][0] <= x <= observation_area.region[0][1]) and (
                                observation_area.region[1][0] <= y <= observation_area.region[1][1]):
                            return True, observation_area
        return False, None

    def obstacle_between(self, node1, node2, agent):
        """
        Checks if there is an obstacle between node1 and node1. Returns True/False
        """
        if self.obstacles[agent] is None:
            return False
        if self.is_inside(node1, self.obstacles, agent)[0] or self.is_inside(node2, self.obstacles, agent)[0]:
            return True

        for cords in self.xy_cords:
            x1 = node1.state[cords[0]]
            y1 = node1.state[cords[1]]
            x2 = node2.state[cords[0]]
            y2 = node2.state[cords[1]]
            p1 = Point(x1, y1)
            q1 = Point(x2, y2)
            for obstacle in self.obstacles[agent]:
                x_min = obstacle[0][0]
                x_max = obstacle[0][1]
                y_min = obstacle[1][0]
                y_max = obstacle[1][1]
                p2 = Point(x_min, y_min)
                q2 = Point(x_min, y_max)
                if doIntersect(p1, q1, p2, q2):
                    return True
                p2 = Point(x_min, y_max)
                q2 = Point(x_max, y_max)
                if doIntersect(p1, q1, p2, q2):
                    return True
                p2 = Point(x_max, y_max)
                q2 = Point(x_max, y_min)
                if doIntersect(p1, q1, p2, q2):
                    return True
                p2 = Point(x_max, y_min)
                q2 = Point(x_min, y_min)
                if doIntersect(p1, q1, p2, q2):
                    return True
            return False

    def generate_new_node(self, parent, rand_node):
        """
        Creates new RRT node
        """
        dist = np.linalg.norm(parent.state - rand_node.state)
        if dist < self.Delta:  # In case rand_node is very close to parent
            new_state = rand_node.state
        else:
            new_state = parent.state + (rand_node.state - parent.state) / dist * self.Delta
        new_node = Node(new_state)
        return new_node

    def draw_tree(self, agent, color='b'):
        """
        Draws the RRT of agent in a plot
        """
        for edge in self.all_edges[agent]:
            parent, child = edge
            for cords in self.xy_cords:
                plt.plot([parent.state[cords[0]], child.state[cords[0]]],
                         [parent.state[cords[1]], child.state[cords[1]]], c=color)
        plt.xlim(self.Xi[0])
        plt.ylim(self.Xi[1])
        plt.show()

    def cost_h(self, node, xg):
        """
        Helper function for stage cost
        """
        h = (node.state - xg).T @ self.Q @ (node.state - xg)
        return float(h)

    def cost_hN(self, node, xg):
        """
        Helper function for terminal cost
        """
        hN = (node.state - xg).T @ self.QN @ (node.state - xg)
        return float(hN)

    @staticmethod
    def return_path(end_node):
        """
        :return: nodes ordered in path, starting at start of root RRT and ending at 'end_node'
        """
        path = [end_node]
        curr_node = end_node
        while curr_node.parent != None:
            curr_node = curr_node.parent
            path.append(curr_node)
        path.reverse()
        return path

    def draw_path_from_node(self, model, end_node, color='b', label=None, style='--', agent=0):
        path = self.return_path(end_node)
        plot(model, path, style=style, color=color, label=label, agent=agent)

    def draw_region(self, constraint, agent):
        """
        Draws all regions in constraint set in a plot

        :param constraint: obstacles or observation_areas
        :param agent: index of agent for whom the constraints should be drawn
        """
        if (constraint == self.obstacles) and (self.obstacles[agent] is not None):
            for area in self.obstacles[agent]:
                x_min, x_max = area[0][0], area[0][1]
                y_min, y_max = area[1][0], area[1][1]
                rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='k', ec="k")
                plt.gca().add_patch(rectangle)
        elif (constraint == self.observation_areas) and (self.observation_areas[agent] is not None):
            for observation_area in self.observation_areas[agent]:
                x_min, x_max = observation_area.region[0][0], observation_area.region[0][1]
                y_min, y_max = observation_area.region[1][0], observation_area.region[1][1]
                rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fc='c', ec="c", alpha=0.5)
                plt.gca().add_patch(rectangle)

        plt.xlim(self.Xi[0])
        plt.ylim(self.Xi[1])

    # def plot(self, path, color, label): TODO: Remove?
    #     all_x_vals = [[] for _ in range(len(self.xy_cords))]
    #     all_y_vals = [[] for _ in range(len(self.xy_cords))]
    #     for node in path:
    #         for i in range(len(xy_cords)):
    #             all_x_vals[i].append(node.state[self.xy_cords[i][0]])
    #             all_y_vals[i].append(node.state[self.xy_cords[i][1]])
    #     for i in range(len(xy_cords)):
    #         plt.plot(all_x_vals[i], all_y_vals[i], c=color, label=label)
    #         plt.plot(all_x_vals[i][-1], all_y_vals[i][-1], '*', c='g')

    def return_node_number(self, node, agent):  # TODO: Do not need this due to node.path_length
        """
        :param node: Tree node
        :param agent: index for agent
        :return: The number 'k' where 'node' is the k:th node in path, i.e,
        the time step k, used in cost update equation
        """
        if node == self.starts[agent]:
            return 0
        else:
            k = 1
            node_temp = node.parent
            while node_temp != self.starts[agent]:
                k += 1
                node_temp = node_temp.parent
            return k


class ObservationArea:
    def __init__(self, region, Thetas):
        """
        :param region: Regions on form [[-x_min, x_max], [y_min, y_max]]
        :param Thetas: List of Thetas for region, [Theta1, Theta2,...], corresponding to noise in observing goal states
        """
        self.region = region
        self.Thetas = Thetas

        # See if observation is perfect
        perfect_theta = np.zeros(Thetas[0].shape)
        perfect_theta[0][0] = 1
        if (Thetas[0] == perfect_theta).all():
            self.perfect_obs = True
        else:
            self.perfect_obs = False


def plot(model, path, style='--', color='b', label=None, agent=0):
    """
    :param model: model of Class model
    :param path: Node path
    :param color: Color of path
    :param label: Label of plotted path
    :return:
    """
    tree = model.root
    all_x_vals = [[] for _ in range(len(tree.xy_cords))]
    all_y_vals = [[] for _ in range(len(tree.xy_cords))]
    observed = 1
    for node in path:
        for i in range(len(tree.xy_cords)):
            all_x_vals[i].append(node.state[tree.xy_cords[i][0]])
            all_y_vals[i].append(node.state[tree.xy_cords[i][1]])
            if node.observed:
                plt.plot(node.state[tree.xy_cords[i][0]], node.state[tree.xy_cords[i][1]], 'o', color='k')
                plt.annotate('o: ' + str(observed), (node.state[tree.xy_cords[i][0]], node.state[tree.xy_cords[i][1]]))
                observed += 1
    for i in range(len(tree.xy_cords)):
        plt.plot(all_x_vals[i], all_y_vals[i], style, c=color, label=label)
        plt.plot(all_x_vals[i][-1], all_y_vals[i][-1], '*', c='g')

    plt.xlim(tree.Xi[0])
    plt.ylim(tree.Xi[1])
    plt.legend()
    plt.show()


######### Code from Geeksforgeeks #######################------------------------------------------------------------
# A Python3 program to find if 2 given line segments intersect or not

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Given three colinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):

        # Clockwise orientation
        return 1
    elif (val < 0):

        # Counterclockwise orientation
        return 2
    else:

        # Colinear orientation
        return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1, q1, p2, q2):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True

    # Special Cases

    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True

    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True

    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True

    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True

    # If none of the cases
    return False


# This code is contributed by Ansh Riyal
######### Code from Geeksforgeeks #######################------------------------------------------------------------


def run():
    N = 1000
    # RRT1 = RRT(starts, Xi, Delta, A, B, Q, QN, goal_states, Omega, b0, Nb=Nb, star=True, gamma=gamma, eta=eta,
    #            N_reset=N_reset, reset_on_obs=False, N_obs=N_obs, obstacles=obstacles, xy_cords=xy_cords,
    #            observation_areas=observation_areas, N_subtrees=2)
    #model = Model(RRT1, N)
    #model.get_plan()
    #best_plan = model.best_plan()
    # best_trees, best_ends, best_hierarchy, lowest_cost = model.get_best_plan()

    # best_ends, best_hierarchy_number = model.return_best_plan()
    # RRT1.draw_region(obstacles, 0)
    # RRT1.draw_region(observation_areas, 0)

    # if observation_made:
    #     plot(RRT1.xy_cords, plan[0], 'm', 'Observation: 1')
    #     plot(RRT1.xy_cords, plan[1], 'r', 'Observation: 2')
    #     # plt.plot(end[0].state[0], end[0].state[1], '*', color='g')  # TODO: hard coded end node
    #     # plt.plot(end[1].state[0], end[1].state[1], '*', color='g')  # TODO: hard coded end node
    # else:
    #     plot(RRT1.xy_cords, plan, 'y', 'No observation')
    #     plt.plot(end.state[0], end.state[1], '*', color='g')  # TODO: hard coded end node
    # return model, best_ends, best_hierarchy_number
    return None, None


#model, best_plan = run()
#plot_agent_plans(model, best_plan)
# model.draw_plan(best_ends, colors = ['b', 'r'])
# model, best_ends, best_hierarchy_number = run()
# model.plot_plan(best_ends)
# plt.plot(model.root.start.state[0], model.root.start.state[1], 'o')
plt.plot(12, 12, 'o', color='r')
plt.plot(-12, 12, 'o', color='r')
# # plt.plot(0, 10, 'o', color='r', label='e=3')
plt.annotate('e=1', (12.3, 12.3))
plt.annotate('e=2', (-11.7, 12.3))
# plt.legend()

b = time.time()
# print(b - a)

# def plot_first_part(agent, node, plot_style, color):
#     RRT = node.RRT
#     while RRT.hierarchy_number != 0:
#         node = RRT.starts[agent]
#         RRT = RRT.parent
#
#     RRT.draw_path_from_node(node, color=color, label=None, style=plot_style)
#
#
# def plot_second_part(agent, node, plot_style, color):
#     RRT = node.RRT
#     RRT.draw_path_from_node(node, color=color, label=None, style=plot_style)
