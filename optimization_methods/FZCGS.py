import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2023)

def condg(g_0, u_0, gamma, eta, OMEGA):
    u_t = u_0.reshape(-1)
    g_t = g_0.reshape(-1)
    t = 0 # timesteps 
    alpha_t=1

    while(True):
        u_expanded = np.repeat(np.expand_dims(u_t,axis=1),OMEGA.shape[1],axis=1)
        v_i = np.argmax(np.dot(g_t+(1/gamma)*(u_t-u_0.reshape(-1)),-OMEGA+u_expanded))
        v = OMEGA[:,v_i]
        dot = np.dot(g_t+(1/gamma)*(u_t-u_0.reshape(-1)),-OMEGA[:,v_i]+u_t)

        if dot <= eta or t==10000000:
            return u_t.reshape(u_0.shape) 
        
        a_dot = np.dot((1/gamma)*(u_t - u_0.reshape(-1)) - g_t , v-u_t ) # to check if (u_t - u_0.reshape(-1)) is correct.
        a_normsq = (np.linalg.norm(v - u_t) ** 2)
        alpha_t = min(1, a_dot / ((1/gamma) * a_normsq))

        u_t = (1 - alpha_t)*u_t + alpha_t*v
            
        t+=1



def FZCGS(x_0, N, q, K, L, obj_func, MGR):

    best_Loss = 1e10
    best_delImgAT = x_0  # at the end, x_k will be -> best_delImgAT

    n = np.square(q)
    # q=3 
    #print("n=",n)
   
    shape = x_0.shape # For MNIST images, shape is 28x28
    
    d = shape[0]*shape[1] # number of iterations is sizes of x_0 multiplied.
    #print("d = " + str(d))
    #print("x_0 shapes: " + str(shape[0]) + " " + str(x_0.shape[1]))


    # Parameters as indicated in Gao et al.

    mu= 1 / np.sqrt(d*K) # =0.11
    #mu = 0.01
    gamma = 1/3*L # 0.01?
    eta = 1/K # =0.1
    #eta = 0.01

    x_k = x_0.copy()
    


    ### FEASIBLE SET ### 

    # This feasible set contains all possible directions in which sum of elements 
    # is equal to 0 (all elems except ones equal to 2000/d and their opposite)
    # We need to implement ||delta||_inf <= s, as in the Gao et al. paper

    num = 2000
    OMEGA = np.eye(d)*num
    OMEGA_rm = np.full(OMEGA.shape, num/d)
    OMEGA = OMEGA - OMEGA_rm
    OMEGA= np.concatenate((OMEGA,-OMEGA), axis=1)

    ####################

    q_val = d 
    e = np.eye(d)
    #e = np.zeros((d, q_val))

    
    loss_values = []
    loss_l2_values = []
    loss_attack_values = []

    # The code is using the original obj_func from the git repo. We need to implement the adversarial obj_func from the Gao et al. paper

    for k in range(N): # iterations, change to N

        if np.mod(k, q) == 0: 
            S1_batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), n, replace = False) 
            v_k = np.zeros(x_k.shape)
            for j in range(d): # loops 28x28 times 
                v_k += (1 / (2 * mu)) * (
                                            obj_func.evaluate(x_k + mu * e[j, :].reshape(shape), S1_batch_idx) -
                                            obj_func.evaluate(x_k - mu * e[j, :].reshape(shape), S1_batch_idx)
                                        ) * e[j, :].reshape(shape)
                #if np.mod(k, 100) == 0:
                #    print(v_k)
        else:
            S2_batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), q, replace=True) 
            v_k = np.zeros(x_k.shape)
            for i in range(q):
                for j in range(d): # loops 28x28 times
                    v_k += (
                                (obj_func.evaluate(x_k + mu * e[j, :].reshape(shape), S2_batch_idx[i:i+1]) -
                                obj_func.evaluate(x_k - mu * e[j, :].reshape(shape), S2_batch_idx[i:i+1])) *
                                e[j, :].reshape(shape)
                            ) - (
                                (obj_func.evaluate(x_prev + mu * e[j, :].reshape(shape), S2_batch_idx[i:i+1]) -
                                obj_func.evaluate(x_prev - mu * e[j, :].reshape(shape), S2_batch_idx[i:i+1])) *
                                e[j, :].reshape(shape)
                            )
                #if np.mod(k, 100) == 0:
                #    print(v_k)

            v_k = (1/q_val)*(1/(2*mu))*(v_k/q)+v_prev
            
        
        v_prev = v_k.copy() 
        x_prev = x_k.copy() 

        x_k = condg(v_k, x_k, gamma, eta, OMEGA)
        #print("New Perturbation x_k:", x_k[:,:,0])


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
        MGR.logHandler.write('\n')
    print("Final Perturbation x_k: ", x_k[:,:,0])

    #### Loss Plot #####################################################################
    plt.plot(range(len(loss_values)), loss_values, label='Loss_Overall')
    plt.plot(range(len(loss_l2_values)), loss_l2_values, label='Loss_L2')
    plt.plot(range(len(loss_attack_values)), loss_attack_values, label='Loss_Attack')
    plt.xlabel('Iteration Index')
    plt.ylabel('Loss Values')
    plt.title('Losses Over Iterations')
    plt.legend()
    plt.savefig('Results/FZCGS/FZCGS_Losses.png')
    plt.show()
    
    #####################################################################################

    return best_delImgAT