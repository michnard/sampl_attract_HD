import numpy as np

def DU_xy(s_t,r_new,r_old, beta, alpha, D, radius):
    observation =  -(beta.T@(s_t.T - 1)).T/alpha  * np.flip(D.T @ r_old) * [1,-1]
    ring = - 4 * (D.T @ r_new) * ( r_new @ D @ D.T @ r_new.T - radius**2)
    return  observation + ring

def run_simulation(N, K, nt, lam, D, Nr, radius, ang_vel, beta_scale, dt, seed=0):
    rng = np.random.default_rng(seed)

    r_real = np.zeros([K,nt])
    V = np.zeros([N,nt])
    sigma = np.zeros([N,nt])
    r = np.zeros([N,nt])

    T = np.diag(D @ D.T) /2
    O_f = - D @ D.T
    O_s = lam * D @ D.T

    ang=0
    V[:,0] = 0.95*T
    r_real[:,0] = r_real[:,1]=r_real[:,2]= np.array([radius*np.cos(ang),radius*np.sin(ang)])
    r[:,0] = r[:,1] = r[:,2]= np.array(np.linalg.pinv(D.T)@r_real[:,0])

    beta = rng.normal(size=(int(Nr/2),1))*beta_scale
    beta = np.concatenate([beta,-beta]) 
    kernel = np.exp(beta @ ang_vel.reshape(1,-1))
    s = rng.poisson(kernel) 
    alpha = (1 + beta.T@beta*dt)[0,0]

    for t in range(2,nt-1):
        ang += ang_vel[t-1]   
        r_real[:,t] = np.array([radius*np.cos(ang),radius*np.sin(ang)])
        
        V[:, t + 1] = V[:, t] + dt * (
            - lam * V[:, t]
            + D @ DU_xy(s[:,t-1],r[:,t-1],r[:,t-2], beta, alpha, D, radius)
            + O_s @ r[:, t]
            + O_f @ sigma[:, t]
        ) + D @ (rng.normal(size=K) * np.sqrt(2 * dt))

        above = np.where(V[:, t + 1] > T)[0]

        if len(above):
            sigma[above[rng.integers(len(above))], t + 1] = 1 / dt

        r[:, t + 1] = r[:, t] + dt * (sigma[:, t + 1] - lam * r[:, t])

    x_y_est = D.T @ r
    xs = x_y_est[0,:]
    ys = x_y_est[1,:]

    angle_est = np.arctan2(xs,ys)
    angle_real = np.arctan2(r_real[0],r_real[1])

    return V,r,s,sigma,angle_est, angle_real, ang_vel, x_y_est