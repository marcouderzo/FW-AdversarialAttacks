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
        v = -np.sign(g_t+(1/gamma)*(u_t-u_0.reshape(-1)))*4
        dot = np.dot(g_t+(1/gamma)*(u_t-u_0.reshape(-1)),v+u_t)

        if dot <= eta or alpha_t < 0.00001 or t==100000:
            return u_t.reshape(u_0.shape) 
        
        a_dot = np.dot((1/gamma)*(u_t - u_0.reshape(-1)) - g_t , v-u_t )
        a_normsq = (np.linalg.norm(v - u_t) ** 2)
        alpha_t = min(1, a_dot / ((1/gamma) * a_normsq))

        u_t = (1 - alpha_t)*u_t + alpha_t*v
            
        t+=1


def condg(g_0, u_0, gamma, eta):
    u = u_0.reshape(-1)
    g = g_0.reshape(-1)
    t = 0
    alpha_t=1

    while True:
        print(t)
        v = -np.sign(g)*4

        dot=np.dot(g, u - v)

        if dot <= eta or alpha_t<0.00001 or t==100000:
            return u.reshape(u_0.shape)
        
        alpha_t = min(np.dot(g, u - v) / (gamma * np.linalg.norm(u - v) ** 2), 1)
        u = u + alpha_t * (v - u)
        g = g_0.reshape(-1) + gamma * (u - u_0.reshape(-1))
        t+=1



def FZCGS(x_0, N, q, K, L, obj_func, MGR):

    t_start = time.time()
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
    gamma = 1/3*L # 0.01
    eta = 1/K # =0.1
    #eta = 0.01

    x_k = x_0.copy()

    q_val = d 
    e = np.eye(d)
    #e = np.zeros((d, q_val))

    
    loss_values = []
    loss_l2_values = []
    loss_attack_values = []

    
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

        x_k = condg(v_k, x_k, gamma, eta)
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
        MGR.logHandler.write(' CurrentTime: ' + str(time.time()-t_start))
        MGR.logHandler.write('\n')
    print("Final Perturbation x_k: ", x_k[:,:,0])

    ##### Loss Plot #####################################################################

    # All plots moved to separate .ipynb notebook loading data from log.txt

    #plt.plot(range(len(loss_values)), loss_values, label='Loss_Overall')
    #plt.plot(range(len(loss_l2_values)), loss_l2_values, label='Loss_L2')
    #plt.plot(range(len(loss_attack_values)), loss_attack_values, label='Loss_Attack')
    #plt.xlabel('Iteration Index')
    #plt.ylabel('Loss Values')
    #plt.title('Losses Over Iterations')
    #plt.legend()
    #plt.savefig('Results/FZCGS/FZCGS_Losses.png')
    #plt.show()
    #
    ######################################################################################

    return best_delImgAT