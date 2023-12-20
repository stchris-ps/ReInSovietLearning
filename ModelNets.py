import torch
import torch.nn as nn

# (Forward) Hamiltonian dynamics network
class HDNet(nn.Module):
    def __init__(self, Hnet):
        super(HDNet, self).__init__()
        self.Hnet = Hnet

    # Copy paramter for kernel Hnet from another net
    def copy_params(self, net):
        self.Hnet.load_state_dict(net.state_dict())

    def forward(self, t, x):
        with torch.enable_grad():
            one = torch.tensor(1, dtype=torch.float, requires_grad=True)
            x = one * x
            q, p = torch.chunk(x, 2, dim=1)
            q_p = torch.cat((q, p), dim=1)
            H = self.Hnet(q_p)
            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            dq, dp = torch.chunk(dH, 2, dim=1)
            # q_dim = x.shape[1]//2; dq, dp, dt = torch.split(dH, [q_dim, q_dim, 1], dim=1)
            # Use forward dynamics: f = (h_p, -h_q)
            return torch.cat((dp, -dq), dim=1)


class HDStochasticNet(nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self, Hnet, sigma, device):
        super(HDStochasticNet, self).__init__()
        self.Hnet = Hnet
        self.sigma = sigma
        self.device = device

    # Drift
    def f(self, t, x):
        with torch.enable_grad():
            one = torch.tensor(1, dtype=torch.float, requires_grad=True)
            x = one * x
            q, p = torch.chunk(x, 2, dim=1)
            q_p = torch.cat((q, p), dim=1)
            H = self.Hnet(q_p)
            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            dq, dp = torch.chunk(dH, 2, dim=1)
            # q_dim = x.shape[1]//2; dq, dp, dt = torch.split(dH, [q_dim, q_dim, 1], dim=1)
            # Use forward dynamics: f = (h_p, -h_q)
            return torch.cat((dp, -dq), dim=1)

    # Constant diffusion
    def g(self, t, x):
        return self.sigma * torch.stack([torch.eye(x.shape[1]) for _ in range(x.shape[0])], axis=0).to(self.device)

    # Copy paramter for kernel Hnet from another net
    def copy_params(self, net):
        self.Hnet.load_state_dict(net.state_dict())