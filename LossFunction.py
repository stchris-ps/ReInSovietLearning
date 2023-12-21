
q0 = env.sample_q()
p0 = adj_net(torch.tensor(q0, dtype = float32))
