import numpy as np
import random

def condg(g_0, u_0, gamma, eta, OMEGA):
    u_t = u_0.reshape(-1)
    g_t = g_0.reshape(-1)
    t = 0
    alpha_t=1

    while(True):
        u_expanded = np.repeat(np.expand_dims(u_t,axis=1),OMEGA.shape[1],axis=1)
        v_idx = np.argmax(np.dot(g_t+(1/gamma)*(u_t-u_0.reshape(-1)),-OMEGA+u_expanded))
        v = OMEGA[:,v_idx]
        dot = np.dot(g_t+(1/gamma)*(u_t-u_0.reshape(-1)),-OMEGA[:,v_idx]+u_t)

        if dot <= eta or alpha_t < 0.00001 or t==10000000:  # v_t is indeed the Frank-Wolfe gap.
            return u_t.reshape(u_0.shape) 
        
        a_dot = np.dot((1/gamma)*(u_t - u_0.reshape(-1)) - g_t , v-u_t ) # to check if (u_t - u_0.reshape(-1)) is correct.
        a_normsq = (np.linalg.norm(v - u_t) ** 2)
        alpha_t = min(1, a_dot / ((1/gamma) * a_normsq))

        u_t = (1 - alpha_t)*u_t + alpha_t*v
            
        t+=1


def CG_working(g_0, u_0, eta, beta, Q):  # Another implementation of CG from another paper. Use this for benchmarking with condg.
    u = u_0.reshape(-1)
    g = g_0.reshape(-1)
    d = len(u)
    t = 0
    alpha_t=1

    while True:
        vidx = np.argmin(np.dot(g,Q))
        v=Q[:,vidx]

        #v = -np.sign(g)*4

        dot=np.dot(g, u - v)
        #print(str(dot) + " " + str(beta))
        if (dot <= beta) or (t==1000000) or (alpha_t<0.00001):
            print('T number in CG = ' + str(t))
            print("final alpha_t: " + str(alpha_t))
            print(str(dot) + " " + str(beta))
            #print(u.reshape((28,28)))
            return u.reshape(u_0.shape)
        alpha_t = min(np.dot(g, u - v) / (eta * np.linalg.norm(u - v) ** 2), 1)
        #print("alpha_t: " + str(alpha_t))
        u = u + alpha_t * (v - u)
        g = g_0.reshape(-1) + eta * (u - u_0.reshape(-1))
        t+=1



def FZCGS(x_0, N, q, K, L, obj_func, MGR):

    best_Loss = 1e10
    best_delImgAT = x_0  # at the end, x_k will be -> best_delImgAT

    n = np.square(q) #why?
    print("n=",n)
   
    shape = x_0.shape
    
    d = shape[0]*shape[1] # dimensionality is sizes of x_0 multiplied.
    print("d = " + str(d))
    print("x_0 shapes: " + str(shape[0]) + " " + str(x_0.shape[1]))

    mu= 1 / np.sqrt(d*K) # =0.01
    gamma = 1/3*L
    eta = 1/K # =0.1

    x_k = x_0.copy()
    


    ### FEASIBLE SET ### 

    # This feasible set contains all possible directions in which sum of elements 
    # is equal to 0 (all elems except ones equal to 2000/d and their opposite
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



    for k in range(50): # iterations, change to N

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
            
        
        ## at the first iteration of the for loop, k=0 so it doesn't jump to the else branch, and therefore v_prev and x_prev are safe to put here
        v_prev = v_k.copy() 
        x_prev = x_k.copy() 

        x_k = condg(v_k, x_k, gamma, eta, OMEGA)
        #x_k = CG_working(v, x_k, gamma,eta,Q)
        print("AFTER condg:", x_k[:,:,0])
        
        # Insert here print current loss at iteration index

        print('Iteration Index: ', k)
        obj_func.print_current_loss()

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

    return best_delImgAT