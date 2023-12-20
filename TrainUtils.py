from collections import namedtuple, deque
import random
import time

import numpy as np

import torch
from torchdiffeq import odeint, odeint_adjoint
from torchsde import sdeint

from CommonNets import Mlp

MAX_VAL = 10.0

def get_architectures():
    q_dim = 3 # Кол-во измерений
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32], activation="tanh")
    hnet = Mlp(input_dim=2 * q_dim, output_dim=1, layer_dims=[8, 16, 32], activation="tanh")
    hnet_target = Mlp(input_dim=2 * q_dim, output_dim=1, layer_dims=[8, 16, 32], activation="tanh")
    return q_dim, adj_net, hnet, hnet_target

def get_train_params():
    T_hnet = 10
    T_adj = 1
    n_timesteps = 0.5

    control_coef = 0.5
    lr_hnet, lr_adj = 1e-3, 1e-4
    return T_hnet, T_adj, n_timesteps, control_coef, lr_hnet, lr_adj

def save_models(adj_net, hnet):
    torch.save(adj_net.state_dict(), 'models/adjoint.pth')
    torch.save(hnet.state_dict(), 'models/hamiltonian_dynamics.pth')

def load_models(adj_net, hnet):
    adj_net.load_state_dict(torch.load('models/adjoint.pth'))
    hnet.load_state_dict(torch.load('models/hamiltonian_dynamics.pth'))

Data = namedtuple('Data', ('q', 'p', 'u', 'f', 'r'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        if len(self.memory) == self.capacity:
            random.shuffle(self.memory)
        self.memory.append(Data(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def sample_step(q, p, env, HDnet, times, memory, control_coef, device):
    qp = torch.cat((q, p), axis=1).to(device)
    with torch.no_grad():
        qps = sdeint(HDnet, qp, times)
    # Go over each time-datapoint in the trajectory to update replay memory
    for i in range(qps.shape[0]):
        qpi_np = qps[i].cpu().detach().numpy()
        qi_np, pi_np = np.split(qpi_np, 2, axis=1)
        # Clipping if things are stochastic
        qi_np, pi_np = np.clip(qi_np, -MAX_VAL, MAX_VAL), np.clip(pi_np, -MAX_VAL, MAX_VAL)
        # Calculate u based on PMP condition H_u = 0
        u = (1.0 / (2 * control_coef)) * np.einsum('ijk,ij->ik', env.f_u(qi_np), -pi_np)
        # Store info into a tuple for replay memory
        dynamic = env.f(qi_np, u);
        reward = env.L(qi_np, u)
        for j in range(qi_np.shape[0]):
            memory.push(torch.tensor(qi_np[j:(j + 1), :], dtype=torch.float, device=device),
                        torch.tensor(pi_np[j:(j + 1), :], dtype=torch.float, device=device),
                        torch.tensor(u[j:(j + 1), :], dtype=torch.float, device=device),
                        torch.tensor(dynamic[j:(j + 1), :], dtype=torch.float, device=device),
                        torch.tensor(reward[j:(j + 1)], dtype=torch.float, device=device))
