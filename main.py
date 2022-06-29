import autograd.numpy as np
from source import decentralized


x0_try = np.array([1.5,2.5,0,0,])
x_Ref_try = np.array([2.5,2.5,0,0])
N = 10
max_iter = 10
regu_init = 100
alpha_init = 1
x_dim = [4]
u_dim = [2]
n_agents = 1
n_states = 4
n_inputs = 2

x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace, alpha_trace = decentralized.run_ilqr(x0_try, N, max_iter, regu_init, alpha_init, x_dim, u_dim, n_agents, x_Ref_try, x_Ref_try)