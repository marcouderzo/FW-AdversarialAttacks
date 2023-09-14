import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from PIL import Image





#class FZCGS: # implement class for tidier code
#    def __init__(self, x_0, n, q, K, L, objective_function, MGR):
#        self.x_0 = x_0
#        self.n = n
#        self.q = q
#        self.K = K
#        self.L = L
#        self.objective_function = objective_function
#        self.MGR = MGR
#        self.v_k = 10e11
#        self.x_k = x_0  # best_delImgAT
#
#        self.n = np.square(q) #why?
#        self.shape = x_0.shape
#
#        self.d = self.shape[0]*self.shape[1] # dimensionality is sizes of x_0 multiplied.
#
#        self.mu= 1 / np.sqrt(self.d*K) # =0.01
#        self.gamma = 1/3*L
#        self.eta = 1/K # =0.1
#
#        self.e = np.eye(self.d)
#



def estimate_gradient_n(x, MGR, n, d, mu, obj_func): #mu smoothing parameter, e_j (R^d) basis vector where only j-th element is 1, otherwise 0.
        S1_batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), n, replace = False) #what is 50? number of elements randomly chosen
        for j in range(d):
            e = np.zeros(x.shape)
            v_estim = np.zeros(x.shape)
            e[j] = 1  # Set the j-th element of the basis vector to 1 
            v_estim= v_estim + ((obj_func(x + mu*e[j, :], S1_batch_idx) - obj_func(x - mu*e[j, :], S1_batch_idx)) / (2*mu)) * e[j, :] # e to reshape to x0.shape        
        return v_estim



# TODO: this needs clarification, i am not very sure on how to implement it. 

def V(g, u, u_t, gamma, x, OMEGA): 
        
        # x is an element of the feasible convex set OMEGA

        v_max_idx = np.argmax(np.dot(g + (1/gamma)*(u_t - u.reshape(-1)), u_t - x)) #finds the index that maximizes the function
        v_max = np.dot(g + (1/gamma)*(u_t - u.reshape(-1)), u_t - x)
        return v_max


def condg(g_0, u_0, gamma, eta, OMEGA):
    t = 1

    u_t = u_0.reshape(-1) # to array -> why reshaping to array?
    g_t = g_0.reshape(-1)

    #u_t= u_0
    while(True):
        v_t = V(g_0, u_0, u_t , gamma, OMEGA)

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
    x_k = x_0  # at the end, x_k will be -> best_delImgAT

    n = np.square(q) #why?
    shape = x_0.shape

    d = shape[0]*shape[1] # dimensionality is sizes of x_0 multiplied.

    mu= 1 / np.sqrt(d*K) # =0.01
    gamma = 1/3*L
    eta = 1/K # =0.1

    x_k = x_0.copy()
    e = np.eye(d)

    q_val = 50

    OMEGA = np.random.uniform(-1, 1, (10000, d))

    for k in range(0, K): #K-1?
        v_prev = v_k 
        x_prev = x_k 
        # probably change where these above are set
        if np.mod(k, q) == 0: 
            v_k = estimate_gradient_n(x_k, MGR, n, d, mu)
        else:
            S2_batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), q, replace=True) #false?
            v_estim = np.zeros(x_k.shape)
            for j in range(q):
                v_estim = (1/q)* (obj_func(x_k, S2_batch_idx[j: j+1]) - obj_func(x_prev, S2_batch_idx[j: j+1]) + v_prev)
                
        
        x_k = condg(v_k, x_k, gamma, eta, OMEGA)
        print(x_k)

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