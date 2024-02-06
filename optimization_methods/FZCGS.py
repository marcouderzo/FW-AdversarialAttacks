import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(2023)

def condg_gao(g_0, u_0, gamma, eta): # does not learn.
    u_t = u_0.reshape(-1)
    g_t = g_0.reshape(-1)
    t = 0 # timesteps 
    alpha_t=1

    while(True):
        print(t)

        # in the other condg, g_t changed at end of step 
        v = -np.sign(g_t+(1/gamma)*(u_t-u_0.reshape(-1)))*4 # -np.sign()*s is the solution of ||delta||_inf constraint

        # Finds the point x in OMEGA maximizing linear approximation of objective function improvement.        
        # dot product as a linear approximation of the objective function improvement if we move
        # from the current iterate u_t to the new point x in the direction of gradient g_t, adjusted by 1/g(v_t-u),
        # that takes into account the distance between the current iterate u_t and starting point u.
        dot = np.dot(g_t+(1/gamma)*(u_t-u_0.reshape(-1)),v+u_t)

        # stopping conditions
        if dot <= eta or alpha_t < 0.00001 or t==100000:
            return u_t.reshape(u_0.shape) 
        

        a_dot = np.dot((1/gamma)*(u_t - u_0.reshape(-1)) - g_t , v-u_t )
        a_normsq = (np.linalg.norm(v - u_t) ** 2)
        alpha_t = min(1, a_dot / ((1/gamma) * a_normsq))

        # Linear combination of the current position u_t and the new point v_t, to form the next u_t+1 where alpha_t stepsize is 
        # similar to an acceleration technique, since it is dynamically adjusted.

        u_t = (1 - alpha_t)*u_t + alpha_t*v
            
        t+=1


def condg(g_0, u_0, gamma, eta):
    u = u_0.reshape(-1)
    g = g_0.reshape(-1)
    t = 0
    alpha_t=1

    while True:
        
        
        # Compute the direction v. This line represents the solution to the linear optimization problem
        # argmin_{v in Omega} <g, v>, which in this case is approximated using the sign of the gradient
        # multiplied by 4. This implies a constraint ||delta||_inf <= 4, where the solution is along the
        # direction of the steepest descent within the feasible set.


        v = -np.sign(g)*4 # -np.sign(g)*s is the solution of ||delta||_inf constraint

        # Compute the dot product between the gradient g and the difference between u and v, which is
        # used in the convergence check and step size calculation.

        dot=np.dot(g, u - v) 

        if dot <= eta or alpha_t<0.00001 or t==100000:
            return u.reshape(u_0.shape)
        
        # Compute the step size alpha_t. It's the minimum between 1 and the ratio of the dot product
        # calculated above to the squared norm of the difference between u and v, scaled by gamma.
        # This step size ensures we do not overstep the minimum along the direction towards v.
        alpha_t = min(np.dot(g, u - v) / (gamma * np.linalg.norm(u - v) ** 2), 1)

        # Update the current point u by taking a step alpha_t towards v from u.
        u = u + alpha_t * (v - u)

        # Update the gradient estimate g for the next iteration.
        # The initial gradient g_0 is adjusted by the scaled difference between the current estimate u
        # and the initial point u_0.
        g = g_0.reshape(-1) + gamma * (u - u_0.reshape(-1))
        t+=1



def FZCGS(x_0, N, q, K, L, obj_func, MGR):

    t_start = time.time()
    best_Loss = 1e10

    # Initialize the best image perturbation to the starting point x_0.
    best_delImgAT = x_0  # at the end, x_k will be -> best_delImgAT

    n = np.square(q)
   
    shape = x_0.shape # For MNIST images, shape is 28x28
    
    d = shape[0]*shape[1] # number of iterations is sizes of x_0 multiplied.



    # Parameters as indicated in Gao et al.

    mu= 1 / np.sqrt(d*K) # =0.11
    gamma = 1/3*L # =0.01
    eta = 1/K # =0.1

    x_k = x_0.copy()

    q_val = d 
    e = np.eye(d)

    
    loss_values = []
    loss_l2_values = []
    loss_attack_values = []

    
    for k in range(N): # Iterations of the algorithm. nStage parameter.
        if np.mod(k, q) == 0: 
            # Sample S1 without replacement.
            S1_batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), n, replace = False) 

            v_k = np.zeros(x_k.shape)
            for j in range(d): # loops 28x28 times 
                
                # As specified in Gao et al., this is using the Coordinate-wise gradient estimator.
                # For each dimension, it calculates the difference in the function value (obj_func.evaluate)
                # when perturbed in the positive and negative directions of the basis vector e[j, :].

                # d: dimensionality of the optimization space
                # e: basis vector (in R^2)
                # mu: smoothing parameter

                v_k += (1 / (2 * mu)) * (
                                            obj_func.evaluate(x_k + mu * e[j, :].reshape(shape), S1_batch_idx) -
                                            obj_func.evaluate(x_k - mu * e[j, :].reshape(shape), S1_batch_idx)
                                        ) * e[j, :].reshape(shape)

        else:
            # Sample S1 with replacement.
            S2_batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), q, replace=True) 

            v_k = np.zeros(x_k.shape)
            for i in range(q):
                for j in range(d): # loops 28x28 times

                    # As specified in Gao et al., this is using the Coordinate-wise gradient estimator.
                    # Similar to the previous gradient estimation but includes the previous gradient estimate (v_prev).
                    # Subtraction of grad estimates at consecutive point is used as variance reduction technique,
                    # stabilizing the gradient estimation by considering the change in gradient than absolute value.
                    # This approach also incorporates a momentum-like term from the previous iteration (+v_prev)

                    # d: dimensionality of the optimization space
                    # e: basis vector (in R^2)
                    # mu: smoothing parameter

                    v_k += (
                                (obj_func.evaluate(x_k + mu * e[j, :].reshape(shape), S2_batch_idx[i:i+1]) -
                                obj_func.evaluate(x_k - mu * e[j, :].reshape(shape), S2_batch_idx[i:i+1])) *
                                e[j, :].reshape(shape)
                            ) - (
                                (obj_func.evaluate(x_prev + mu * e[j, :].reshape(shape), S2_batch_idx[i:i+1]) -
                                obj_func.evaluate(x_prev - mu * e[j, :].reshape(shape), S2_batch_idx[i:i+1])) *
                                e[j, :].reshape(shape)
                            )

            # Normalize and add the previous gradient estimate to the current estimate.
            v_k = (1/q_val)*(1/(2*mu))*(v_k/q)+v_prev
            
        # Store the previous values for the next iteration.
        v_prev = v_k.copy() 
        x_prev = x_k.copy() 

        # Perform conditional gradient step
        x_k = condg(v_k, x_k, gamma, eta)

        # Evaluate the objective function at the new x_k
        obj_func.evaluate(x_k, np.array([]), False)

        

        print('Iteration Index: ', k)
        obj_func.print_current_loss()
        
        loss_values.append(obj_func.Loss_Overall)
        loss_l2_values.append(obj_func.Loss_L2)
        loss_attack_values.append(obj_func.Loss_Attack)

        if(obj_func.Loss_Overall < best_Loss):
            best_Loss = obj_func.Loss_Overall
            best_delImgAT = x_k
            #print('Updating best delta image record')

        MGR.logHandler.write('Iteration Index: ' + str(k))
        MGR.logHandler.write(' Query_Count: ' + str(obj_func.query_count))
        MGR.logHandler.write(' Loss_Overall: ' + str(obj_func.Loss_Overall))
        MGR.logHandler.write(' Loss_Distortion: ' + str(obj_func.Loss_L2))
        MGR.logHandler.write(' Loss_Attack: ' + str(obj_func.Loss_Attack))
        MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
        MGR.logHandler.write(' CurrentTime: ' + str(time.time()-t_start))
        MGR.logHandler.write('\n')
    print("Final Perturbation x_k: ", x_k[:,:,0])

    return best_delImgAT
