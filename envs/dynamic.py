### Continuous version for Hamiltonian dynamics training of environments in OpenAI Gym classical controls
### The rendering part is the same as OpenAI Gym
import casadi
import numpy as np
import math
from gym.utils import seeding
from os import path
from physys import PhySys
from quadrotor import toQuaternion, Quadrotor
import torch

BB_POS = 5
BB_VEL = 10
BB_CONTROL = 9
BB_ANG = np.pi/2

# QUADROTOR MASS AND GRAVITY VALUE
M, G = 1.03, 9.82

# AIR DENSITY
RHO = 1.2041

#DRAG COEFFICIENT
C_D = 1.1

# ELETRIC MOTOR THRUST AND MOMENT
K_F = 1.435e-5
K_M = 2.4086e-7
I_R = 5e-5
T2WR = 2

## INDIRECT CONTROL CONSTANTS ##
IC_THRUST = 6
IC_MOMENTUM = 0.8


# INERTIA MATRIX
J = np.array([[16.83e-3, 0, 0],
              [0, 16.83e-3, 0],
              [0, 0, 28.34e-3]])

# ELETRIC MOTOR DISTANCE TO CG
D = 0.26

#PROJECTED AREA IN X_b, Y_b, Z_b
BEAM_THICKNESS = 0.05
A_X = BEAM_THICKNESS*2*D
A_Y = BEAM_THICKNESS*2*D
A_Z = BEAM_THICKNESS*2*D*2
A = np.array([[A_X,A_Y,A_Z]]).T


## REWARD PARAMETERS ##

# CONTROL REWARD PENALITIES #
P_C = 0.2
P_C_D = 0.3

## TARGET STEADY STATE ERROR ##
TR = [0.01, 0.1]
TR_P = [100, 10]
def quat_rot_mat(q):
    q = q.flatten()
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]
    R = np.array([[a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c],
                  [2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b],
                  [2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]])
    return R
def deriv_quat(w, q):
    w = w.flatten()
    q.reshape((4,1))
    wx = w[0]
    wy = w[1]
    wz = w[2]
    omega = np.array([[0, -wx, -wy, -wz],
                      [wx, 0, wz, -wy],
                      [wy, -wz, 0, wx],
                      [wz, wy, -wx, 0]])
    dq = 1/2*np.dot(omega,q).flatten()
    return dq

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
        self.id = np.eye(q_dim)
        self.eps = 1e-8
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
        self.inv_j = np.linalg.inv(J)

    # q([np.array]) = (q0, q1) = (position, velocity)
    def f(self, q, u):
        result = []
        for i in range(q.shape[0]):
            # qn = self.physics.next_step(q[i], u[i])
            # cur = (qn-q[i]) * 10
            result.append(self.fsingle(q[i], u[i]))
        result = np.array(result)
        return result

    def f2F(self, f_action):
        f = (f_action.squeeze() + 1) * T2WR * M * G / 8
        w = np.array([[np.sqrt(f[0] / K_F)],
                      [np.sqrt(f[1] / K_F)],
                      [np.sqrt(f[2] / K_F)],
                      [np.sqrt(f[3] / K_F)]])

        F_new = np.sum(f)
        M_new = np.array([[(f[2] - f[0]) * D],
                          [(f[1] - f[3]) * D],
                          [(-f[0] + f[1] - f[2] + f[3]) * K_M / K_F]])
        return w, F_new, M_new
    def fsingle(self, qx, action):
        #print('u:', qx, action)
        self.w, f_in, m_action = self.f2F(action)
        #print('m:', m_action)
        x = qx.squeeze()
        # print(x.shape)
        # BODY INERTIAL VELOCITY
        vel_x = x[1]
        vel_y = x[3]
        vel_z = x[5]

        # QUATERNIONS
        q0 = x[6]
        q1 = x[7]
        q2 = x[8]
        q3 = x[9]

        # BODY ANGULAR VELOCITY
        w_xx = x[10]
        w_yy = x[11]
        w_zz = x[12]

        # QUATERNION NORMALIZATION (JUST IN CASE)
        q = np.array([[q0, q1, q2, q3]]).T
        q = q / np.linalg.norm(q)

        # DRAG FORCES ESTIMATION (BASED ON BODY VELOCITIES)
        self.mat_rot = quat_rot_mat(q)
        v_inertial = np.array([[vel_x, vel_y, vel_z]]).T
        v_body = np.dot(self.mat_rot.T, v_inertial)
        f_drag = -0.5 * RHO * C_D * np.multiply(A, np.multiply(abs(v_body), v_body))

        # DRAG MOMENTS ESTIMATION (BASED ON BODY ANGULAR VELOCITIES)

        # Discretization over 10 steps (linear velocity varies over the body)
        d_xx = np.linspace(0, D, 10)
        d_yy = np.linspace(0, D, 10)
        d_zz = np.linspace(0, D, 10)
        m_x = 0
        m_y = 0
        m_z = 0
        for xx, yy, zz in zip(d_xx, d_yy, d_zz):
            m_x += -RHO * C_D * BEAM_THICKNESS * D / 10 * (abs(xx * w_xx) * (xx * w_xx))
            m_y += -RHO * C_D * BEAM_THICKNESS * D / 10 * (abs(yy * w_yy) * (yy * w_yy))
            m_z += -2 * RHO * C_D * BEAM_THICKNESS * D / 10 * (abs(zz * w_zz) * (zz * w_zz))

        m_drag = np.array([[m_x],
                           [m_y],
                           [m_z]])

        # GYROSCOPIC EFFECT ESTIMATION (BASED ON ELETRIC MOTOR ANGULAR VELOCITY)
        omega_r = (-self.w[0] + self.w[1] - self.w[2] + self.w[3])[0]

        m_gyro = np.array([[-w_xx * I_R * omega_r],
                           [+w_yy * I_R * omega_r],
                           [0]])

        # BODY FORCES
        self.f_in = np.array([[0, 0, f_in]]).T
        self.f_body = self.f_in + f_drag

        # BODY FORCES ROTATION TO INERTIAL
        self.f_inertial = np.dot(self.mat_rot, self.f_body)

        # INERTIAL ACCELERATIONS
        accel_x = self.f_inertial[0, 0] / M
        accel_y = self.f_inertial[1, 0] / M
        accel_z = self.f_inertial[2, 0] / M - G
        self.accel = np.array([[accel_x, accel_y, accel_z]]).T

        # BODY MOMENTUM
        W = np.array([[w_xx],
                      [w_yy],
                      [w_zz]])

        m_in = m_action + m_drag - np.cross(W.flatten(), np.dot(J, W).flatten()).reshape((3, 1))

        # INERTIAL ANGULAR ACCELERATION
        accel_ang = np.dot(self.inv_j, m_in).flatten()
        accel_w_xx = accel_ang[0]
        accel_w_yy = accel_ang[1]
        accel_w_zz = accel_ang[2]

        # QUATERNION ANGULAR VELOCITY (INERTIAL)

        self.V_q = deriv_quat(W, q).flatten()
        dq0 = self.V_q[0]
        dq1 = self.V_q[1]
        dq2 = self.V_q[2]
        dq3 = self.V_q[3]

        # RESULTS ORDER:
        # 0 x, 1 vx, 2 y, 3 vy, 4 z, 5 vz, 6 q0, 7 q1, 8 q2, 9 q3, 10 w_xx, 11 w_yy, 12 w_zz
        out = np.array([vel_x, accel_x,
                        vel_y, accel_y,
                        vel_z, accel_z,
                        dq0, dq1, dq2, dq3,
                        accel_w_xx, accel_w_yy, accel_w_zz])
        #print(out)
        return out

    def f_usingle(self, q):
        #print('--------------------------------------------')
        #print(q)
        eps = 0.01
        arr = []
        for i in range(4):
            u = np.zeros(4)
            u[i] = eps
            R1 = self.fsingle(q, np.zeros(4) + u)
            R2 = self.fsingle(q, np.zeros(4) - u)
            arr.append(((R1 - R2) / (2 * eps))[np.newaxis, :])
        result = np.vstack(arr)#.T
        #print('----------------------------------------------')
        return result
    def f_u(self, q):
        eps = 1e-6
        zeros = np.array([0, 0, 0, 0])
        epses = [[eps, 0, 0, 0], [0, eps, 0, 0], [0, 0, eps, 0], [0, 0, 0, eps]]
        result = []
        for i in q:
            i = i[None, :]
            res = []
            res.append(self.f_usingle(i))
            res = np.array(res)
            res = res.flatten()
            res = np.reshape(res, (4, 13))
            result.append(res)
        result = np.array(result)
        return result

    def L(self, q, u):
        res = []
        for i in range(q.shape[0]):
            res.append(np.sum(u[i] ** 2))
        res = np.array(res)
        return res * self.control_coef + self.g(q)

    def g(self, q):
        res = []
        for i in range(q.shape[0]):
            res.append((q[i][0] - 5) ** 2 + (q[i][1] - 5) ** 2 + (q[i][2] - 5) ** 2)
        res = np.array(res)
        return res

    # def nabla_g(self, q):
    #     ret = np.zeros((q.shape[0], self.q_dim))
    #     for i in range(self.q_dim):
    #         ret[:, i] = (self.g(q + self.eps * self.id[i]) - self.g(q - self.eps * self.id[i])) / (2 * self.eps)
    #     return ret
    def nabla_g(self, q):
        res = [0] * 13
        res[0] = 2 * (q[0] - 5)
        res[1] = 2 * (q[1] - 5)
        res[2] = 2 * (q[2] - 5)
        return torch.tensor(res, dtype=torch.float32)

    def sample_start(self):
        ini_r_I = [-4, -6, 9.]
        ini_v_I = [0.0, 0.0, 0.0]
        ini_q = toQuaternion(0, [1, -1, 1])
        ini_w = [0.0, 0.0, 0.0]
        ini_state = ini_r_I + ini_v_I + ini_q + ini_w
        #print("ini_state", ini_state)
        return np.array(ini_state)

    def sample_q(self, num_examples, mode='train'):
        res = np.random.uniform(low=-10, high=10, size=(num_examples, 13))
        return res

