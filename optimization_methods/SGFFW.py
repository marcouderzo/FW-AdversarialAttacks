import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(2023)

def SGFFW(x_0, N, m, obj_func, grad_approx_scheme, MGR):
    t_start = time.time()

    best_Loss = 1e10
    best_delImgAT = x_0  #x_t perturbation will be saved in best_delImgAT
    d = x_0.shape[0]*x_0.shape[1] # number of iterations is sizes of x_0 multiplied.
    x_t = x_0.copy()
    d_t = np.zeros(d)

    loss_values = []
    loss_l2_values = []
    loss_attack_values = []


    # Choose N random batch indexes between [0 , nFunc] with replacement
    # Determine the batches of data used in each iteration for objective function evaluation.
    batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), N, replace = True) 

    
    for t in range(0, N):

        gamma_t = 2/(t+8) # Step-size parameter, decreasing over time to ensure convergence.
        
        g_t = np.zeros(x_t.shape)
        
        # difference is in the number of directions

        if grad_approx_scheme == 'RDSA':

            # RDSA: samples a single random direction z_t from a standard normal distribution. Only one direction is used
            # at each iteration. The gradient is estimated using a finite difference in this random direction.

            s = 8 # s value for ||delta||_inf <= s constraint 

            # Calculated rho_t and c_t parameters for RDSA, as per Sahu et al. pseudocode
            rho_t = 4 / ((np.power(d, 1/3)) * np.power(t+8, 1/3))
            c_t = 2 / (np.power(d, 3/2)*np.power(t+8, 1/3))

            # Sampling 1 random direction from Normal distribution
            z_t = np.random.normal(size=(x_0.shape[0], x_0.shape[1], 1)) 

            # g_t(x_t, y, z_t) computation
            # The gradient g_t is estimated using a finite difference approach: The objective function is 
            # evaluated at the current point and a point slightly perturbed in the direction of z_t. 
            # Gradient estimate is the difference in those two evaluation, scaled by c_t.
            g_t = (obj_func.evaluate(x_t + c_t * z_t[:,:, :], batch_idx[t:t+1]) - obj_func.evaluate(x_t, batch_idx[t:t+1])) / c_t * z_t[:, :, :]
        
        if grad_approx_scheme == 'I-RDSA':

            # I-RDSA: similar to RDSA, but averages the gradient estimates over m independently sampled directions.
            # Each z_i,t is a random vector sampled from a normal distribution, and the final gradient estimate
            # is the average of the finite differences along these m directions.

            s = 4 # s value for ||delta||_inf <= s constraint 

            # Calculated rho_t and c_t parameters for I-RDSA, as per Sahu et al. pseudocode
            rho_t = 4 / (np.power(1+(d/m), 1/3) * np.power(t+8, 2/3))
            c_t = (2 / np.sqrt(m)) / (np.power(d, 3/2) * np.power(t+8, 1/3))

            # Sampling m random directions from Normal distribution
            z_it = np.random.normal(size=(x_0.shape[0], x_0.shape[1], m)) 


            # Like RDSA, but I-RDSA averages gradient estimates over the m directions. 
            # The gradient g_t is estimated by averaging the gradients computed in the direction of each z_i in z_it.
            # z_it: m-dimensional random vector containing the m directions.

            for i in range(m):
                g_t += (obj_func.evaluate(x_t + c_t * z_it[:,:, i:i+1], batch_idx[t:t+1]) - obj_func.evaluate(x_t, batch_idx[t:t+1])) / c_t * z_it[:,:, i:i+1]
            g_t = g_t / m
        
        if grad_approx_scheme == 'KWSA':
            
            # KWSA: estimates gradient using finite differences along each coordinate direction.
            # The number of directions is equal to the dimension d of the problem space.
            # Computes the function f for a small perturbation along each basis vector e_i and the current point.

            s = 4 # s value for ||delta||_inf <= s constraint 

            # Calculated rho_t and c_t parameters for KWSA, as per Sahu et al. pseudocode
            rho_t = 4 / (np.power(t+8, 2/3))
            c_t = 2 / (np.power(d, 1/2) * np.power(t+8, 1/3))

            # Each row of e is a unit vector along one of the axes of the d-dimensional space, used for gradient estimation
            e = np.eye(d)

            # Iterates through the d dimensions of the problem space.
            # For each dimension, it computes a finite difference approximation of the gradient along that direction.
            # obj_func.evaluate is called twice for each dimension: once at the point x_t + c_t * e[i, :] 
            # (a small step along the i-th direction) and once at x_t.
            # The gradient estimate in the i-th direction is computed by difference of these two evaluations.
            # This estimated component is added to g_t, accumulating the gradient estimate over all dimensions.
            for i in range(d):
                g_t = g_t + ((obj_func.evaluate(x_t + c_t * e[i, :].reshape(x_0.shape), batch_idx[t:t+1]) - obj_func.evaluate(x_t, batch_idx[t:t+1])) / c_t * e[i, :].reshape(x_0.shape))
            

        # Updating the direction d_t using the current and previous gradient
        d_t = (1 - rho_t)*d_t + rho_t * g_t.reshape(-1)


        #### As specified in the pseudocode (SGFFW, Sahu et al.) ####

        # This passage is not used since we are directly finding v_t with -np.sign(d_t)*s

        #v_t = C[:, np.argmin(np.dot(d_t, C))] # Convex Set C
        #v_t = v_t.reshape(x_0.shape)

        ##############################################################

        # if you multiply by dt this is the value that minimizes it
        v_t = -np.sign(d_t) * s   # ||delta||_inf <= s constraint
        v_t = v_t.reshape(x_0.shape)


        x_t = (1-gamma_t)*x_t + gamma_t*v_t


        obj_func.evaluate(x_t,np.array([]),False)
        print('Iteration Index: ', t)
        obj_func.print_current_loss()


        loss_values.append(obj_func.Loss_Overall)
        loss_l2_values.append(obj_func.Loss_L2)
        loss_attack_values.append(obj_func.Loss_Attack)



        if(obj_func.Loss_Overall < best_Loss):
            best_Loss = obj_func.Loss_Overall
            best_delImgAT = x_t

        MGR.logHandler.write('Iteration Index: ' + str(t))
        MGR.logHandler.write(' Query_Count: ' + str(obj_func.query_count))
        MGR.logHandler.write(' Loss_Overall: ' + str(obj_func.Loss_Overall))
        MGR.logHandler.write(' Loss_Distortion: ' + str(obj_func.Loss_L2))
        MGR.logHandler.write(' Loss_Attack: ' + str(obj_func.Loss_Attack))
        MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
        MGR.logHandler.write(' CurrentTime: ' + str(time.time()-t_start))
        MGR.logHandler.write('\n')


    print("Final Perturbation x_t: ", x_t[:,:,0])

    ##### Loss Plot #####################################################################

    # All plots moved to separate .ipynb notebook loading data from log.txt

    #plt.plot(range(len(loss_values)), loss_values, label='Loss_Overall')
    #plt.plot(range(len(loss_l2_values)), loss_l2_values, label='Loss_L2')
    #plt.plot(range(len(loss_attack_values)), loss_attack_values, label='Loss_Attack')
    #plt.xlabel('Iteration Index')
    #plt.ylabel('Loss Values')
    #plt.title('Losses Over Iterations')
    #plt.legend()
    #plt.savefig('Results/SGFFW/SGFFW_Losses.png')
    #plt.show()
    #
    ######################################################################################

    return best_delImgAT

        

        