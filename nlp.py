from casadi import *
from numpy import *
from scipy.integrate import odeint
import pdb
import itertools
import numpy as np
from cvxpy import *
import time


class NLP(object):
    """ Non-Linear Program
	"""

    def __init__(self, N, Q, R, dR, Qf, goal, dt, bx, bu, printLevel, agent, ellipse, avoid_obs=False):
        # Define variables
        self.N = N
        self.n = Q.shape[1]
        self.d = R.shape[1]
        self.bx = bx
        self.bu = bu
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.dR = dR
        self.goal = goal
        self.dt = dt
        self.ellipse = ellipse
        self.agent = agent
        self.avoid_obs = avoid_obs

        self.bx = bx
        self.bu = bu

        self.printLevel = printLevel

        print("Initializing FTOCP")
        self.buildFTOCP()
        self.solverTime = []
        print("Done initializing FTOCP")

    def solve(self, x0, verbose=False):
        # Set initial condition + state and input box constraints
        self.lbx = x0.tolist() + (-self.bx).tolist() * (self.N) + (-self.bu).tolist() * self.N
        self.ubx = x0.tolist() + (self.bx).tolist() * (self.N) + (self.bu).tolist() * self.N

        if self.agent==0 and self.avoid_obs:  # Obstacle constraint
            self.lbx = self.lbx + [1] * (self.N-1) # + [-1000] * self.n
            self.ubx = self.ubx + [100000] * (self.N-1)  # + [1000] * self.n 

        # Solve nonlinear programm
        start = time.time()
        sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)
        end = time.time()
        self.solverTime.append(end - start)

        # Check if the solution is feasible
        if (self.solver.stats()['success']):
            self.feasible = 1
            x = sol["x"]
            self.xPred = np.array(x[0:(self.N + 1) * self.n].reshape((self.n, self.N + 1))).T
            self.uPred = np.array(
                x[(self.N + 1) * self.n:((self.N + 1) * self.n + self.d * self.N)].reshape((self.d, self.N))).T
            self.mpcInput = self.uPred[0][0]

            if self.printLevel >= 2:
                print("xPredicted:")
                print(self.xPred)
                print("uPredicted:")
                print(self.uPred)

            if self.printLevel >= 1: print("NLP Solver Time: ", self.solverTime[-1], " seconds.")

        else:
            self.xPred = np.zeros((self.N + 1, self.n))
            self.uPred = np.zeros((self.N, self.d))
            self.mpcInput = []
            self.feasible = 0
            print("Unfeasible")

        return self.uPred[0]

    def buildFTOCP(self):
        # Define variables
        N = self.N
        n = self.n
        d = self.d

        # Define variables
        X = SX.sym('X', n * (self.N + 1))
        U = SX.sym('U', d * self.N)
        if self.avoid_obs:
            slackObs = SX.sym('X', (self.N-1))

        # Define dynamic constraints
        self.constraint = []
        for i in range(0, self.N):
            X_next = self.dynamics(X[n * i:n * (i + 1)], U[d * i:d * (i + 1)])
            for j in range(0, self.n):
                self.constraint = vertcat(self.constraint, X_next[j] - X[n * (i + 1) + j])

        # Obstacle constraint for quadruped
        if self.avoid_obs:
            if self.agent==0:
                for i in range(1, N):
                    self.constraint = vertcat(self.constraint, ((X[n*i+0] -  self.ellipse[0])**2/self.ellipse[2]**2) + ((X[n*i+1] - self.ellipse[1])**2/self.ellipse[3]**2) - slackObs[i-1])

            # Defining Cost (We will add stage cost later)
        self.cost = 0
        for i in range(0, self.N):
            self.cost = self.cost + (X[n * i:n * (i + 1)] - self.goal).T @ self.Q @ (X[n * i:n * (i + 1)] - self.goal)
            self.cost = self.cost + U[d * i:d * (i + 1)].T @ self.R @ U[d * i:d * (i + 1)]

            if i < self.N-1:
                ii = i + 1
                self.cost = self.cost + (U[d * i:d * (i + 1)]-U[d * ii:d * (ii + 1)]).T @ self.dR @ U[d * ii:d * (ii + 1)]

        self.cost = self.cost + (X[n * self.N:n * (self.N + 1)] - self.goal).T @ self.Qf @ (
                X[n * self.N:n * (self.N + 1)] - self.goal)




        # Set IPOPT options
        # opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive","ipopt.mu_init":1e-5,"ipopt.mu_min":1e-15,"ipopt.barrier_tol_factor":1}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
        opts = {"verbose": False, "ipopt.print_level": 0,
                "print_time": 0}  # \\, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
        
        if self.agent==0 and self.avoid_obs:
            nlp = {'x':vertcat(X,U, slackObs), 'f': self.cost, 'g': self.constraint}
        else:
            nlp = {'x': vertcat(X, U), 'f': self.cost, 'g': self.constraint}
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)

        # Set lower bound of inequality constraint to zero to force n*N state dynamics
        self.lbg_dyanmics = [0] * (n * self.N)
        self.ubg_dyanmics = [0] * (n * self.N)

        if self.agent==0 and self.avoid_obs:  # Add obstacle constraint for quadruped
            self.lbg_dyanmics = self.lbg_dyanmics + [0*1.0]*(N-1) #+ [0]*n
            self.ubg_dyanmics = self.ubg_dyanmics + [0*100000000]*(N-1) #+ [0]*n
            

    def dynamics_model(self, x, t, u):
        x_t = x[0]
        y_t = x[1]
        theta_t = x[2]

        dxdt = u[0] * np.cos(theta_t)
        dydt = u[0] * np.sin(theta_t)
        dthetadt = u[1]

        return [dxdt, dydt, dthetadt]



    ### Replace this with real world dynamics...
    def dynamics(self, x, u):
        # state x = [x,y, theta]

        x_next = x[0] + self.dt * cos(x[2]) * u[0]
        y_next = x[1] + self.dt * sin(x[2]) * u[0]
        theta_next = x[2] + self.dt * u[1]

        #xnext = odeint(self.dynamics_model, x, [0, self.dt], args=(u,))


        state_next = [x_next, y_next, theta_next]

        return state_next
