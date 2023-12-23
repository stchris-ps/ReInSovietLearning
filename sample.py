from quadrotor import *
from physys import *
from TrainUtils import get_architectures, load_models, get_environment
import torch
import matplotlib


_, adj_net, hnet, hnet_target = get_architectures()
env = get_environment("quadrotor")
load_models(adj_net, hnet)


uav = Quadrotor()
Jx, Jy, Jz, mass, win_len = 1, 1, 1, 1, 0.4
uav.initDyn(Jx=Jx, Jy=Jy, Jz=Jz, mass=mass, l=win_len, c=0.01)
wr, wv, wq, ww = 1, 1, 5, 1
uav.initCost(wr=wr, wv=wv, wq=wq, ww=ww, wthrust=0.1)

dt = 0.1
dyn = uav.X + dt * uav.f
# set initial, control and dynamic state
uavoc = PhySys(uav.X, uav.U, dyn)


def u(q):
    p = adj_net(torch.tensor(q, dtype=torch.float32))
    print(-env.f_usingle(q) @ p.detach().numpy().reshape(1, p.shape[0]).transpose())
    return -env.f_usingle(q) @ p.detach().numpy().reshape(1, p.shape[0]).transpose()

ini_r_I = [-4, -6, 9.]
ini_v_I = [0.0, 0.0, 0.0]
ini_q = toQuaternion(0, [1, -1, 1])
ini_w = [0.0, 0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w

uav.play_animation(1.5, uavoc.integrate_sys(ini_state, 70, u))