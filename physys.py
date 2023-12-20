import casadi
import numpy as np
from casadi import *


class PhySys:
    def __init__(self, state, control, ode):
        self.n_state = 13
        self.state = state
        self.control = control
        self.dyn = ode
        self.dyn_fn = casadi.Function('dynFun', [self.state, self.control], [self.dyn])

    # поиск траектории по массиву control_val
    # ini_state - начальное состояние
    # horizon - кол во кадров
    # control_val - двумерный массив размера horizon, хранит 4 числа - управление моторами квадрокоптера
    def integrate_sys(self, ini_state, horizon, control):
        state_traj = np.zeros((horizon + 1, self.n_state))
        state_traj[0, :] = ini_state
        for t in range(horizon):
            curr_x = state_traj[t, :]
            curr_u = np.array(control(curr_x))

            state_traj[t + 1, :] = self.dyn_fn(curr_x, curr_u).full().flatten()
        return state_traj

    # то же самое что и integrate_sys, только для 1 шага
    # control_val - массив размера 4 - управление моторами квадрокоптера на текущем шаге
    def next_step(self, state, control_val):
        new_state = self.dyn_fn(state, np.array(control_val)).full().flatten()

        return new_state
