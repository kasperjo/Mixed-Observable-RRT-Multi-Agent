import numpy as np
import pdb
from mpac_cmd import *


class system(object):
    """docstring for system"""

    def __init__(self, x0, dt, agent, simulation):
        self.x = [x0]
        self.u = []
        self.w = []
        self.x0 = x0
        self.dt = dt
        self.agent = agent
        self.simulation = simulation

    def applyInput(self, ut):
        self.u.append(ut)

        xt = self.x[-1]

        if self.agent == 1:  # drone
            if self.simulation[1]: # TODO: Change to pass and apply input in 'main.py'
                x_next = xt[0] + self.dt * np.cos(xt[2]) * ut[0]
                y_next = xt[1] + self.dt * np.sin(xt[2]) * ut[0]
                theta_next = xt[2] + self.dt * ut[1]
            else:
                x_next = xt[0] + self.dt * np.cos(xt[2]) * ut[0]
                y_next = xt[1] + self.dt * np.sin(xt[2]) * ut[0]
                theta_next = xt[2] + self.dt * ut[1]

        elif self.agent == 0:
            if self.simulation[0]:  # quadruped
                vx = ut[0]
                vrz = ut[1]
                walk_mpc_idqp(vx=vx, vrz=vrz)

                data = get_tlm_data()
                x_next = data["q"][0]
                y_next = data["q"][1]
                theta_next = data["q"][5]
            else:
                x_next = xt[0] + self.dt * np.cos(xt[2]) * ut[0]
                y_next = xt[1] + self.dt * np.sin(xt[2]) * ut[0]
                theta_next = xt[2] + self.dt * ut[1]

        state_next = np.array([x_next, y_next, theta_next])

        self.x.append(state_next)

    def reset_IC(self):
        self.x = [self.x0]
        self.u = []
        self.w = []
