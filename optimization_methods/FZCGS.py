import numpy as np
import random




def V(g, u, u_t, gamma, x, OMEGA): 
        # TODO: this needs clarification
        # x is an element of the feasible convex set OMEGA

        v_max_idx = np.argmax(np.dot(g + (1/gamma)*(u_t - u.reshape(-1)), u_t - x)) #finds the index that maximizes the function
        v_max = np.dot(g + (1/gamma)*(u_t - u.reshape(-1)), u_t - x)
        return v_max


def condg(g_0, u_0, gamma, eta, Q):
    t = 1

    u_t = u_0.reshape(-1) # to array -> why reshaping to array?
    g_t = g_0.reshape(-1)

    #u_t= u_0
    while(True):
        v_t = V(g_0, u_0, u_t , gamma, Q)

        if v_t <= eta:  # v_t is indeed the Frank-Wolfe gap.
            break
        
        norm = np.square(np.linalg.norm(v_t - u_t))
        arg = np.inner((1/gamma) * (u_0.reshape(-1) - u_t) - g_0, v_t - u_t) / ((1/gamma) * norm)
        alpha_t = np.min([1, arg])
        u_t = (1-alpha_t) * u_t + alpha_t * v_t

        t = t+1

    return u_t 


def FZCGS(x_0, n, q, K, L, obj_func, MGR):

    v_k = 10e11
    best_delImgAT = x_0  # at the end, x_k will be -> best_delImgAT

    n = np.square(q) #why?
    print("here3")
    shape = x_0.shape
    
    d = shape[0]*shape[1] # dimensionality is sizes of x_0 multiplied.

    mu= 1 / np.sqrt(d*K) # =0.01
    gamma = 1/3*L
    eta = 1/K # =0.1
    print("here4")
    x_k = x_0.copy()
    e = np.eye(d)

    ### FEASIBLE SET ###
    num = 2000
    Q = np.eye(d)*num
    Qrm = np.full(Q.shape, num/d)
    Q = Q - Qrm
    Q= np.concatenate((Q,-Q), axis=1)
    ####################

    q_val = 50 #B


    for k in range(0, n): #it goes from 0 to n-1

        
        if np.mod(k, q) == 0: 
            S1_batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), n, replace = False) #what is 50? number of elements randomly chosen
            e = np.zeros(d)
            v_k = np.zeros(x_k.shape)
            for j in range(d):
                e[j] = 1  # Set the j-th element of the basis vector to 1 -> e basis vector needs to be corrected, there's something I'm messing up
                v_k= v_k + ((obj_func.evaluate(x_k + mu*e[j, :], S1_batch_idx) - obj_func.evaluate(x_k - mu*e[j, :], S1_batch_idx)) / (2*mu)) * e[j, :] # e to reshape to x0.shape        
        else:
            S2_batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), q, replace=True) #false?
            v_k = np.zeros(x_k.shape)
            for j in range(q):
                for k in range(q_val):
   
                   v_k = v_k +  (
                                    
                                    obj_func.evaluate(x_k + mu * e[k, :], S2_batch_idx[j : j + 1])
                                    - obj_func.evaluate(x_k - mu * e[k, :], S2_batch_idx[j : j + 1])
                                ) * e[k, :] / (2*mu) - (
                                    
                                    obj_func.evaluate(x_prev + mu * e[k, :], S2_batch_idx[j : j + 1])
                                    - obj_func.evaluate(x_prev - mu * e[k, :], S2_batch_idx[j : j + 1])
                                ) * e[k, :] / (2*mu)
                            

                v_k = v_k + v_prev
            v_k = (1/q_val)*(v_k)
        
        # at the first iteration of the for loop, k=0 so it doesn't jump to the else branch, and therefore v_prev and x_prev are safe to put here
        v_prev = v_k.copy() # is it right to copy prev here?
        x_prev = x_k.copy() # is it right to copy prev here?

        x_k = condg(v_k, x_k, gamma, eta, Q)
        print(x_k)
        
        # Insert here print current loss at iteration index

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

    return best_delImgAT