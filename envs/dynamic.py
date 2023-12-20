### Continuous version for Hamiltonian dynamics training of environments in OpenAI Gym classical controls
### The rendering part is the same as OpenAI Gym
import casadi
import numpy as np
import math
from gym.utils import seeding
from os import path
from physys import PhySys
from quadrotor import toQuaternion, Quadrotor


### Generic continuous environment for reduced Hamiltonian dynamics framework
class ContinuousEnv():
    def __init__(self, q_dim=1, u_dim=1, control_coef=0.5):
        self.q_dim = q_dim
        self.u_dim = u_dim
        self.control_coef = control_coef
        self.eps = 1e-8
        self.id = np.eye(q_dim)
        self.seed()

        # Viewer for rendering image
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Dynamics f
    def f(self, q, u):
        return np.zeros((q.shape[0], self.q_dim))

    # Partial derivative of dynamics f wrt control u. Assuming linear control
    def f_u(self, q):
        return np.zeros((q.shape[0], self.q_dim, self.u_dim))

    # Lagrangian or running cost L
    def L(self, q, u):
        return self.control_coef * np.sum(u ** 2, axis=1) + self.g(q)

    # Terminal cost g
    def g(self, q):
        return np.zeros(q.shape[0])

    # Nabla of g
    def nabla_g(self, q):
        ret = np.zeros((q.shape[0], self.q_dim))
        for i in range(self.q_dim):
            ret[:, i] = (self.g(q + self.eps * self.id[i]) - self.g(q - self.eps * self.id[i])) / (2 * self.eps)
        return ret

    # Sampling state q
    def sample_q(self, num_examples, mode='train'):
        return np.zeros((num_examples, self.q_dim))

    # Image rendering
    def render(self, q, mode="rgb_array"):
        return

    # Close rendering
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


#### Mountain car for PMP ####
class QuadroCopter(ContinuousEnv):
    def __init__(self, q_dim=13, u_dim=4, control_coef=0.5, goal_velocity=0):
        super().__init__(q_dim, u_dim, control_coef)
        uav = Quadrotor()
        Jx, Jy, Jz, mass, win_len = 1, 1, 1, 1, 0.4
        uav.initDyn(Jx=Jx, Jy=Jy, Jz=Jz, mass=mass, l=win_len, c=0.01)
        wr, wv, wq, ww = 1, 1, 5, 1
        uav.initCost(wr=wr, wv=wv, wq=wq, ww=ww, wthrust=0.1)

        dt = 0.1
        dyn = uav.X + dt * uav.f
        # set initial, control and dynamic state
        self.physics = PhySys(uav.X, uav.U, dyn)
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = (
            0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        )
        self.goal_velocity = goal_velocity
        self.power = 0.0015

    # q([np.array]) = (q0, q1) = (position, velocity)
    def f(self, q, u):
        qn = self.physics.next_step(q, u)
        return qn - q / 0.1

    def f_u(self, q):
        dfu = casadi.jacobian(self.physics.dyn, self.physics.control)
        dfu_f = casadi.Function('dfu', [self.physics.state, self.physics.control], [dfu])
        arr = np.zeros(shape=(13, 4))
        for i in range(13 * 4):
            arr[i // 4][i % 4] = dfu_f(q, np.array([0, 0, 0, 0]))[i]
        return arr

    def L(self, q, u):
        print(q)
        print(u)
        dx = q[0][0] - 5
        dy = q[0][1] - 5
        dz = q[0][2] - 5
        return np.array([dx ** 2 + dy ** 2 + dz ** 2 + self.control_coef * np.dot(u[0], u[0])])

    def g(self, q):
        dx = q[0][0] - 5
        dy = q[0][1] - 5
        dz = q[0][1] - 5
        d = dx ** 2 + dy ** 2 + dz ** 2
        return d ** 2

    def sample_q(self, num_examples, mode='train'):
        ini_r_I = [-4, -6, 9.]
        ini_v_I = [0.0, 0.0, 0.0]
        ini_q = toQuaternion(0, [1, -1, 1])
        ini_w = [0.0, 0.0, 0.0]
        ini_state = ini_r_I + ini_v_I + ini_q + ini_w
        print("ini_state", ini_state)
        return ini_state

