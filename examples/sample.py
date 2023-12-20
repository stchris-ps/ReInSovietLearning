from quadrotor import *
from physys import *
from NeuralNetworks.classical_controls import QuadroCopter

uav = Quadrotor()
Jx, Jy, Jz, mass, win_len = 1, 1, 1, 1, 0.4
uav.initDyn(Jx=Jx, Jy=Jy, Jz=Jz, mass=mass, l=win_len, c=0.01)
wr, wv, wq, ww = 1, 1, 5, 1
uav.initCost(wr=wr, wv=wv, wq=wq, ww=ww, wthrust=0.1)

dt = 0.1
dyn = uav.X + dt * uav.f
# set initial, control and dynamic state
uavoc = PhySys(uav.X, uav.U, dyn)

copter = QuadroCopter(uavoc)

ini_r_I = [-4, -6, 9.]
ini_v_I = [0.0, 0.0, 0.0]
ini_q = toQuaternion(0, [1, -1, 1])
ini_w = [0.0, 0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w

print(copter.f_u(ini_state))
