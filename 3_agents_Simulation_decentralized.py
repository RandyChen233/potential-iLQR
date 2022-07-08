#!/usr/bin/env python
# coding: utf-8


# python libraries
import autograd.numpy as np
import matplotlib.pyplot as plt


from autograd import elementwise_grad as egrad
from autograd import jacobian,hessian

import itertools

from matplotlib.animation import FuncAnimation

from matplotlib import rc




n_x = 4
n_u = 2
def unicycle_continuous_dynamics(x, u):
    # x = [x position, y position, heading, forward velocity] 
    # u = [omega, forward acceleration]
    #m = sym if x.dtype == object else np # Check type for autodiff
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


# In[9]:


def discrete_dynamics(x, u):
    dt = 0.05
    #Euler integrator below and return the next state
    x = x + dt*unicycle_continuous_dynamics(x,u)
    x_next = x
    return x_next


# In[10]:


def discrete_dynamics_multiple(x,u,x_dim,u_dim):
        
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


def cost_trj(x,u,x_ref): #x-> state vector, u-> input vector
    
    
    Q = np.zeros((len(x),len(x)))
    
    for i,j in zip(range(n_agents),range(n_agents)): 
      #n_agents,n_states are global variables to be declared before the main iLQR loop
      
        Q[i*n_states:(i+1)*n_states,j*n_states:(j+1)*n_states] = np.diag([200,200,0,0,])  
      
      #we don't want to penalize the 3rd and 4th state in each agent (heading&velocity)
    
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

    # if torch.is_tensor(X):
    #     return torch.linalg.norm(dX, dim=0)
    
    return np.linalg.norm(dX, axis=0)



def cost_avoidance(x,x_dim):
    
    if len(x_dim) == 1:
        
        return 0
    
    threshold = 0.5 #threshold distance below which cost avoidance is activated

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


from autograd import elementwise_grad as egrad


# In[20]:



def l_x(x,u,x_dim,x_ref):
  
    return egrad(cost_stage,0)(x,u,x_dim,x_ref)
    # return approx_fprime(x, lambda x: cost_stage(x, u, x_dim, x_ref), jac_eps)
    
def l_u(x,u,x_dim,x_ref):
    
    return egrad(cost_stage,1)(x,u,x_dim,x_ref)
    # return approx_fprime(u, lambda u: cost_stage(x, u, x_dim, x_ref), jac_eps)
    
def l_xx(x,u,x_dim,x_ref):
    
    return hessian(cost_stage,0)(x,u,x_dim,x_ref)
    # return np.vstack(
    #      [approx_fprime(x, lambda x: l_x(x,u,x_dim,x_ref)[i], hess_eps) for i in range(len(x))]
    # )
    
def l_uu(x,u,x_dim,x_ref):
    
    return hessian(cost_stage,1)(x,u,x_dim,x_ref)
    # return  np.vstack(
    #      [approx_fprime(u, lambda u: l_u(x,u,x_dim,x_ref)[i], hess_eps) for i in range(len(u))]
    #  )
    

def l_ux(x,u,x_dim,x_ref): #this is not correct?
    
    return jacobian(l_u,0)(x,u,x_dim,x_ref)#, somehow auto-differentiation throws warning on l_ux all the time
    
    # return  np.vstack(
    #     [approx_fprime(x, lambda x: l_u(x,u,x_dim,x_ref)[i], hess_eps) for i in range(len(u))]
    # )

#note: warning shows up later due to the fact that l_ux( ) is always a zero array, but that's okay
 


def f(x,u,x_dim,u_dim):
    
    return discrete_dynamics_multiple(x,u,x_dim,u_dim)

def f_x(x,u,x_dim,u_dim):
    
    return jacobian(f,0)(x,u,x_dim,u_dim)
    # return np.vstack([approx_fprime(x, lambda x: f(x,u,x_dim,u_dim)[i], jac_eps) for i in range(len(x))])

def f_u(x,u,x_dim,u_dim): #this is not correct
    
    
    return jacobian(f,1)(x,u,x_dim,u_dim)
    # return np.vstack([approx_fprime(u, lambda u: f(x,u,x_dim,u_dim)[i], jac_eps) for i in range(len(x))])




def l_final_x(x_T,x_ref_T):
    
    
    return np.eye(x_T.shape[0])*1500@(x_T-x_ref_T)


def l_final_xx(x_T):
    

    return np.eye(x_T.shape[0])*1500




def Q_terms(V_x, V_xx , x, u, x_dim, u_dim, x_ref):

    Q_x = l_x(x,u,x_dim,x_ref) + f_x(x,u,x_dim,u_dim).T@(V_x);
    
    Q_u = l_u(x,u,x_dim,x_ref) + f_u(x,u,x_dim,u_dim).T@(V_x);
    
    Q_xx = l_xx(x,u,x_dim,x_ref) + f_x(x,u,x_dim,u_dim).T@(V_xx.dot(f_x(x,u,x_dim,u_dim)))
    
    Q_ux = l_ux(x,u,x_dim,x_ref) + f_u(x,u,x_dim,u_dim).T@(V_xx.dot(f_x(x,u,x_dim,u_dim)))
    Q_uu = l_uu(x,u,x_dim,x_ref) + f_u(x,u,x_dim,u_dim).T@(V_xx.dot(f_u(x,u,x_dim,u_dim)))

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
        
        Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(V_x, V_xx,x_trj[n+1,:],u_trj[n,:],x_dim,u_dim,x_ref)
        
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
    u_trj = np.random.randn(N-1, n_agents*n_inputs)*0.0001
    x_trj = rollout(x0, u_trj, x_dim, u_dim)
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
        
        k_trj, K_trj, expected_cost_redu = backward_pass(x_trj, u_trj, regu, alpha, x_dim, u_dim, x_ref_T, x_ref)
        
        x_trj_new, u_trj_new = forward_pass(x_trj, u_trj, k_trj, K_trj, expected_cost_redu, total_cost, alpha, x_dim, u_dim)
        # Evaluate new trajectory
        total_cost = cost_sum(x_trj_new, u_trj_new, x_dim, x_ref, x_ref_T)
        
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
            print("iteration accepted!")
            # alpha doesn't change if accepted
        else:
            # Reject new trajectories and increase regularization
            regu *= 2.0
            alpha = alpha* 0.5 # a scaling factor of 0.5 for alpha is a typical value
            cost_trace.append(cost_trace[-1])
            redu_ratio_trace.append(0)
            print("iteration not accepted, increase regularization and lower line search descent rate!")
        regu = min(max(regu, min_regu), max_regu)
        regu_trace.append(regu)
        redu_trace.append(cost_redu)
        
        alpha = min(max(alpha,min_alpha),max_alpha)
        print(alpha)
        alpha_trace.append(alpha)
        
        # Early termination if expected improvement is small
        if expected_cost_redu <= 1e-6:
            break
            
    return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace, alpha_trace

x_ref = np.array([2.5, 1.5, 0 , 0 , 0.5, 1.5, np.pi , 0 , 1.5, 2.2, np.pi/2, 0 ])
x_ref_T = x_ref

# Setup problem and call iLQR


x_dim = [4,4,4]
u_dim = [2,2,2]

n_agents = 3

n_inputs = 2
N = 15 #prediction horizon at each step
total_horizon = 50 #total horizon for receding-horizon controller
n_states = 4
x0 = np.array([0.5, 1.5, 0, 0,      #1st row for agent1, 2nd row for agent2, 3rd row for agent3
               2.5, 1.5, np.pi ,0 ,  
               1.5, 1.3, np.pi/2 , 0.1 ]) 

# x_ref = np.array([2.5, 1.5, 0 , 0.1 , 0.5, 1.5, np.pi , -0.1 , 1.5, 2.0, np.pi/2, 0.1 ])
x_ref = np.array([2.5, 1.5, 0 , 0 , 0.5, 1.5, np.pi , 0 , 1.5, 2.0, np.pi/2, 0 ])
x_ref_T = x_ref


max_iter  = 15
regu_init = 100
alpha_init = 1

x_trj_opt = np.zeros((total_horizon+1,n_agents*n_states))
x_trj_opt[0,:] = x0
u_trj_opt = []


def main():


    
    for m in range(total_horizon):

        x_trj, u_trj, _, _, _, _, _ = run_ilqr(x_trj_opt[m,:], N, max_iter, regu_init, alpha_init, x_dim, u_dim, n_agents, x_ref, x_ref_T)
        #each time run_ilqr is called, x_trj and u_trj are returned based on an optiomization over the prediction horizon N
        x_trj_opt[m+1,:] = x_trj[1,:]
        u_trj_opt.append(u_trj[0])


if __name__ == "__main__":
    main()


