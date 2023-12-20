from collections import namedtuple, deque
import random
import time

import numpy as np

import torch
from torchdiffeq import odeint, odeint_adjoint
from torchsde import sdeint

from CommonNets import Mlp

def get_architectures():
    q_dim = 3 # Кол-во измерений
    adj_net = Mlp(input_dim=q_dim, output_dim=q_dim, layer_dims=[8, 16, 32], activation="tanh")
    hnet = Mlp(input_dim=2 * q_dim, output_dim=1, layer_dims=[8, 16, 32], activation="tanh")
    hnet_target = Mlp(input_dim=2 * q_dim, output_dim=1, layer_dims=[8, 16, 32], activation="tanh")
    return q_dim, adj_net, hnet, hnet_target

