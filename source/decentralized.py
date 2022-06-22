import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl    
import seaborn as sns
plt.style.use('seaborn') 
import itertools
import torch



def compute_pairwise_distance(X, x_dims):
    """Compute the distance between each pair of agents"""
    assert len(set(x_dims)) == 1

    n_agents = len(x_dims)
    n_states = x_dims[0]
    
    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    X_agent = X.reshape(-1, n_agents, n_states).swapaxes(0, 2)
    dX = X_agent[:2, pair_inds[:, 0]] - X_agent[:2, pair_inds[:, 1]]

    if isinstance(X, np.ndarray):
        return np.linalg.norm(dX, axis=0)
    elif torch.is_tensor(X):
        return torch.linalg.norm(dX, dim=0)



def online_split_states(n_agents,n_states,N,problems,X):
    # problems is a list of indices grouped together for each set of agents
    #i.e., problems = [
#     (1,),
#     (2, 0)
#                    ]

    x_dims = [n_states]*n_agents
#     X = np.tile(np.arange(n_agents * n_states), (N, 1))
    full_states = [np.zeros((N,0))] * len(problems)
    for i, problem in enumerate(problems):
        for id_ in problem:
            full_states[i] = np.concatenate([
                full_states[i], X[:,id_*n_states:(id_+1)*n_states]
            ], axis=1)
        
    return full_states


def online_split_inputs(n_agents,n_inputs,N,problems,U):

        u_dims = [n_inputs]*n_agents
#         U = np.tile(np.arange(n_agents * n_inputs), (N, 1))
        full_inputs = [np.zeros((N,0))] * len(problems)
        for i, problem in enumerate(problems):
            for id_ in problem:
                full_inputs[i] = np.concatenate([
                    full_inputs[i], U[:,id_*n_inputs:(id_+1)*n_inputs]
                ], axis=1)
        
        return full_inputs
    
    

    
    
    


def hello():
    
    print("goobye")
    
def add(a, b):
    return a + b