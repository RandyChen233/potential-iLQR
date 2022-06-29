import autograd.numpy as np
import torch
import itertools
from autograd import elementwise_grad as egrad
from autograd import hessian,jacobian

def unicycle_continuous_dynamics(x, u):
    # x = [x position, y position, heading, forward velocity] 
    # u = [omega, forward acceleration]

    x_pos = x[3]*np.cos(x[2])
    y_pos = x[3]*np.sin(x[2])
    heading_rate = u[0]
    v_dot = u[1]
    
    x_d = np.array([
        x_pos,
        y_pos,
        heading_rate,
        v_dot    
    ])
    
    return x_d


def discrete_dynamics(x, u):
    
    dt = 0.05
    #Euler integrator below and return the next state
    x = x + dt*unicycle_continuous_dynamics(x,u)
    x_next = x
    return x_next


def discrete_dynamics_multiple(x,u,x_dim,u_dim):
    n_inputs = 2
    n_states = 4
    x_new = np.hstack(
         [discrete_dynamics(x[i*n_states:(i+1)*n_states],u[i*n_inputs:(i+1)*n_inputs]) for i in range(len(x_dim))]
    )

    return x_new

def rollout(x0, u_trj, x_dim, u_dim): #rolling out the state trajectory based on the euqations of motion (discretized)

    x_trj = np.zeros((u_trj.shape[0]+1, x0.shape[0]))

    x_trj[0,:] = x0
    
    for i in range(0,x_trj.shape[0]-1): 
        x_trj[i+1,:] = discrete_dynamics_multiple(x_trj[i,:],u_trj[i,:],x_dim,u_dim)
    
    return x_trj

def cost_trj(x,u,x_ref): #x-> state vector, u-> input vectora
    #m = sym if x.dtype == object else np
    
    Q = np.eye(x.shape[0])*100
    
    R = np.eye(u.shape[0])
    
    cost = (x-x_ref).T @ Q @(x-x_ref) +(u).T @ R @ (u)

    return cost #trajectory cost for an agent with index i


def cost_trj_Final(x_T,x_ref_T):
   # m = sym if x.dtype == object else np
    Q = np.eye(x_T.shape[0])*1600
    
    terminal_cost = (x_T-x_ref_T).T @ Q @ (x_T-x_ref_T)

    return terminal_cost #final trajectory cost for an agent with index i





def compute_pairwise_distance(X, x_dims):
    """Compute the distance between each pair of agents"""
    assert len(set(x_dims)) == 1

    n_agents = len(x_dims)
    n_states = x_dims[0]
    
    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    X_agent = X.reshape(-1, n_agents, n_states).swapaxes(0, 2)
    dX = X_agent[:2, pair_inds[:, 0]] - X_agent[:2, pair_inds[:, 1]]

    if torch.is_tensor(X):
        return torch.linalg.norm(dX, dim=0)
    
    
    return np.linalg.norm(dX, axis=0)


def cost_avoidance(x,x_dim):
    
    if len(x_dim) == 1:
        
        return 0
    
    threshold = 0.5

    distances = compute_pairwise_distance(x,x_dim)

    cost_avoid = np.sum((distances[distances<threshold]-threshold)**2)*1000

    return cost_avoid


def cost_stage(x, u, x_dim, x_ref):
    
    c_avoid = cost_avoidance(x,x_dim)
    c_trj = cost_trj(x,u,x_ref)
    
    return c_avoid + c_trj

def cost_sum(x_trj, u_trj, x_dim, x_ref, x_ref_T):

    total = 0.0
    # TODO: Sum up all costs
    for i in range(0,x_trj.shape[0]-1): #(0,1,2,3,....)
        total = total + cost_stage(x_trj[i,:], u_trj[i,:], x_dim, x_ref) 

    total = total + cost_trj_Final(x_trj[-1,:], x_ref_T)

    return total

from scipy.optimize import approx_fprime
jac_eps = np.sqrt(np.finfo(float).eps)
hess_eps = np.sqrt(jac_eps)
def l_x(x,u,x_dim,x_ref):
  
    return egrad(cost_stage,0)(x,u,x_dim,x_ref)
    # return approx_fprime(x, lambda x: cost_stage(x, u, x_dim, x_ref), jac_eps)
    
def l_u(x,u,x_dim,x_ref):
    
    return egrad(cost_stage,1)(x,u,x_dim,x_ref)
    # return approx_fprime(u, lambda u: cost_stage(x, u, x_dim, x_ref), jac_eps)
    
def l_xx(x,u,x_dim,x_ref):
    
    return hessian(cost_stage,0)(x,u,x_dim,x_ref)
    # return np.vstack(
         # [approx_fprime(x, lambda x: l_x(x,u,x_dim,x_ref)[i], hess_eps) for i in range(len(x))]
    # )
    
def l_uu(x,u,x_dim,x_ref):
    
    return hessian(cost_stage,1)(x,u,x_dim,x_ref)
    # return  np.vstack(
    #      [approx_fprime(u, lambda u: l_u(x,u,x_dim,x_ref)[i], hess_eps) for i in range(len(u))]
    #  )
    

def l_ux(x,u,x_dim,x_ref): #this is not correct?
    
    # return egrad(l_u,0)(x,u,x_dim,x_ref)#, somehow auto-differentiation throws warning on l_ux all the time
    
    return  np.vstack(
        [approx_fprime(x, lambda x: l_u(x,u,x_dim,x_ref)[i], hess_eps) for i in range(len(u))]
    )




def f(x,u,x_dim,u_dim):
    
    return discrete_dynamics_multiple(x,u,x_dim,u_dim)

def f_x(x,u,x_dim,u_dim):
    
    return jacobian(f,0)(x,u,x_dim,u_dim)
    # return np.vstack([approx_fprime(x, lambda x: f(x,u,x_dim,u_dim)[i], jac_eps) for i in range(len(x))])

def f_u(x,u,x_dim,u_dim): #this is not correct
    
    
    return jacobian(f,1)(x,u,x_dim,u_dim)
    # return np.vstack([approx_fprime(u, lambda u: f(x,u,x_dim,u_dim)[i], jac_eps) for i in range(len(x))])
    
    
    
def l_final_x(x_T,x_ref_T):
    
    
    return np.eye(x_T.shape[0])*1600@(x_T-x_ref_T)


def l_final_xx(x_T):
    

    return np.eye(x_T.shape[0])*1600


def Q_terms(V_x, V_xx , x, u, x_dim, u_dim, x_ref):

    Q_x = l_x(x,u,x_dim,x_ref) + f_x(x,u,x_dim,u_dim).T.dot(V_x);
    
    Q_u = l_u(x,u,x_dim,x_ref) + f_u(x,u,x_dim,u_dim).T.dot(V_x);
    
    Q_xx = l_xx(x,u,x_dim,x_ref) + f_x(x,u,x_dim,u_dim).T.dot(V_xx.dot(f_x(x,u,x_dim,u_dim)))
    
    Q_ux = l_ux(x,u,x_dim,x_ref) + f_u(x,u,x_dim,u_dim).T.dot(V_xx.dot(f_x(x,u,x_dim,u_dim)))
    Q_uu = l_uu(x,u,x_dim,x_ref) + f_u(x,u,x_dim,u_dim).T.dot(V_xx.dot(f_u(x,u,x_dim,u_dim)))

    return Q_x, Q_u, Q_xx, Q_ux, Q_uu


def gains(Q_uu, Q_u, Q_ux):
    
    k = np.linalg.solve(Q_uu,-Q_u)
    K = np.linalg.solve(Q_uu,-Q_ux)
    
    
    return k, K


def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):

    V_x = Q_x + K.T.dot(Q_uu.dot(k)) + K.T.dot(Q_u) + Q_ux.T.dot(k);
    V_xx = Q_xx  + K.T.dot(Q_uu.dot(K)) + K.T.dot(Q_ux) + Q_ux.T.dot(K) ;
    
    
    
    return V_x, V_xx


def expected_cost_reduction(Q_u, Q_uu, k, alpha):
    
    return - alpha*Q_u.T.dot(k) - 0.5  * alpha**2 *   k.T.dot(Q_uu.dot(k))

def forward_pass(x_trj, u_trj, k_trj, K_trj, expected_cost_redu, total_cost, alpha, x_dim, u_dim):
    #alpha is the gradient descent rate

  
    x_trj_new = np.zeros(x_trj.shape)
    x_trj_new[0,:] = x_trj[0,:]
    u_trj_new = np.zeros(u_trj.shape)
    
#     for n in range(u_trj.shape[0]):
#         u_trj_new[n,:] = # Apply feedback law
#         x_trj_new[n+1,:] = # Apply dynamics

    for n in range(u_trj.shape[0]):
        
        u_trj_new[n,:] =u_trj[n,:]+ alpha * k_trj[n,:] + K_trj[n,:].dot((x_trj_new[n,:]-x_trj[n,:])); # Apply feedback law
        
        x_trj_new[n+1,:] =discrete_dynamics_multiple(x_trj_new[n,:],u_trj_new[n,:], x_dim, u_dim);
    
   
    
    return x_trj_new, u_trj_new


def backward_pass(x_trj, u_trj, regu, alpha, x_dim, u_dim,x_ref_T,x_ref):
    k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
    K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
    expected_cost_redu = 0

    V_x = l_final_x(x_trj[-1,:],x_ref_T)

    V_xx = l_final_xx(x_trj[-1,:])
    
    
    for n in range(u_trj.shape[0]-1, -1, -1):
        
        Q_x,Q_u,Q_xx,Q_ux,Q_uu = Q_terms(V_x, V_xx,x_trj[n+1,:],u_trj[n,:],x_dim,u_dim,x_ref)
        
        # We add regularization to ensure that Q_uu is invertible and nicely conditioned
        Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
       
        k, K = gains(Q_uu_regu, Q_u, Q_ux)
        
        k_trj[n,:] = k
        K_trj[n,:,:] = K
        
        V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
        
        expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k, alpha)
    return k_trj, K_trj, expected_cost_redu


def run_ilqr(x0, N, max_iter, regu_init, alpha_init, x_dim, u_dim, n_agents, x_ref, x_ref_T):
    # First forward rollout
    n_inputs = 2
    n_states = 4
    u_trj = np.random.randn(N-1, n_agents*n_inputs)*0.0001
    x_trj = rollout(x0, u_trj,x_dim,u_dim)
    total_cost = cost_sum(x_trj, u_trj, x_dim, x_ref, x_ref_T)
    regu = regu_init
    max_regu = 10000
    min_regu = 0.01
    
    alpha = alpha_init
    max_alpha = 1.0
    min_alpha = 0.0
    
    # Setup traces
    cost_trace = [total_cost]
    expected_cost_redu_trace = []
    redu_ratio_trace = [1]
    redu_trace = []
    regu_trace = [regu]
    
    alpha_trace = [alpha]
    # Run main loop
    for it in range(max_iter):
        # Backward and forward pass
        
        k_trj, K_trj, expected_cost_redu = backward_pass(x_trj, u_trj, regu, alpha, x_dim, u_dim, x_ref_T,x_ref)
        print(alpha)
        x_trj_new, u_trj_new = forward_pass(x_trj, u_trj, k_trj, K_trj, expected_cost_redu, total_cost, alpha, x_dim, u_dim)
        # Evaluate new trajectory
        total_cost = cost_sum(x_trj_new, u_trj_new, x_dim,x_ref,x_ref_T)
        
        cost_redu = cost_trace[-1] - total_cost
        redu_ratio = cost_redu / abs(expected_cost_redu)
        # Accept or reject iteration
        if redu_ratio >= 1e-4 and redu_ratio <= 10  :
            # Improvement! Accept new trajectories and lower regularization
            redu_ratio_trace.append(redu_ratio)
            cost_trace.append(total_cost)
            x_trj = x_trj_new
            u_trj = u_trj_new
            regu *= 0.7
            # alpha doesn't change if accepted
        else:
            # Reject new trajectories and increase regularization
            regu *= 2.0
            alpha = alpha* 0.5 # a scaling factor of 0.5 for alpha is a typical value
            cost_trace.append(cost_trace[-1])
            redu_ratio_trace.append(0)
        regu = min(max(regu, min_regu), max_regu)
        regu_trace.append(regu)
        redu_trace.append(cost_redu)
        
        alpha = min(max(alpha,min_alpha),max_alpha)
        alpha_trace.append(alpha)
        
        # Early termination if expected improvement is small
        if expected_cost_redu <= 1e-6:
            break
            
    return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace, alpha_trace




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

    
def define_inter_graph_threshold(X, n_agents, radius, x_dims):
    """Compute the interaction graph based on a simple thresholded distance
    for each pair of agents sampled over the trajectory
    """
    planning_radii = 4 * radius
    rel_dists = compute_pairwise_distance(X, x_dims).T

    N = X.shape[0]
    n_samples = 10 
    sample_step = max(N // n_samples, 1)
    sample_slice = slice(0, N + 1, sample_step)
    
    # Put each pair of agents within each others' graphs if they are within
    # some threshold distance from each other.
    graph = {i: [i] for i in range(n_agents)}
    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    for i, pair in enumerate(pair_inds):
        if np.any(rel_dists[sample_slice, i] < planning_radii):
            graph[pair[0]].append(pair[1])
            graph[pair[1]].append(pair[0])
    
    graph = {agent_id: sorted(prob_ids) for agent_id, prob_ids in graph.items()}
    
    return graph
