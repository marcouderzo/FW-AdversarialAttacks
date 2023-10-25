import numpy as np
import random

def condg(g_0, u_0, gamma, eta, Q):
    u_t = u_0.reshape(-1)
    g_t = g_0.reshape(-1)
    t = 0
    alpha_t=1

    while(True):
        uexp = np.repeat(np.expand_dims(u_t,axis=1),Q.shape[1],axis=1)
        vidx = np.argmax(np.dot(g_t+(1/gamma)*(u_t-u_0.reshape(-1)),-Q+uexp))
        v = Q[:,vidx]
        dot = np.dot(g_t+(1/gamma)*(u_t-u_0.reshape(-1)),-Q[:,vidx]+u_t)

        if dot <= eta:  # v_t is indeed the Frank-Wolfe gap.
            return u_t.reshape(u_0.shape)
        
        #norm = np.square(np.linalg.norm(v_t - u_t))
        #arg = np.inner((1/gamma) * (u_0.reshape(-1) - u_t) - g_0, v - u_t) / ((1/gamma) * norm)
        #alpha_t = np.min([1, arg])
        #u_t = (1-alpha_t) * u_t + alpha_t * v
        #t = t+1
        alpha = min( gamma * np.dot((1/gamma)*(u_t - u_0.reshape(-1)) - g_t , v-u_t ) / (np.linalg.norm(v - u_t) ** 2), 1)
        u_t = (1 - alpha)*u_t + alpha*v
        t+=1


def CG_working(g_0, u_0, eta, beta, Q):
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
    num = 2000
    Q = np.eye(d)*num
    Qrm = np.full(Q.shape, num/d)
    Q = Q - Qrm
    Q= np.concatenate((Q,-Q), axis=1)
    ####################

    q_val = d 
    e = np.eye(d)
    #e = np.zeros((d, q_val))


    #e = np.zeros((d, q_val))
    #for j in range(q_val):
    #    basis_vector = np.zeros(q_val)
    #    basis_vector[j]=1
    #    e[j, :] = basis_vector

    for k in range(N):

        if np.mod(k, q) == 0: 
            S1_batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), n, replace = False) 
            v_k = np.zeros(x_k.shape)
            for idx in range(d):
                (1/(2*mu))*(obj_func.evaluate(x_k + mu * e[idx,:].reshape(shape),S1_batch_idx) - obj_func.evaluate(x_k - mu * e[idx,:].reshape(shape),S1_batch_idx)) * e[idx,:].reshape(shape)
                #print("if", v_k)
        else:
            S2_batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), q, replace=True) 
            v_k = np.zeros(x_k.shape)
            for qidx in range(q):
                for idx in range(d):
                    v += ((obj_func.evaluate(x_k + mu * e[idx,:].reshape(shape),S2_batch_idx[qidx:qidx+1]) - obj_func.evaluate(x_k - mu * e[idx,:].reshape(shape),S2_batch_idx[qidx:qidx+1])) * e[idx,:].reshape(shape)) - ((obj_func.evaluate(x_prev + mu * e[idx,:].reshape(shape),S2_batch_idx[qidx:qidx+1]) - obj_func.evaluate(x_prev - mu * e[idx,:].reshape(shape),S2_batch_idx[qidx:qidx+1])) * e[idx,:].reshape(shape))
                #print("else: ", v_k)

            v = (1/q_val)*(1/(2*mu))*(v/q)+v_prev
            
        
        ## at the first iteration of the for loop, k=0 so it doesn't jump to the else branch, and therefore v_prev and x_prev are safe to put here
        v_prev = v_k.copy() # is it right to copy prev here?
        x_prev = x_k.copy() # is it right to copy prev here?

        x_k = condg(v_k, x_k, gamma, eta, Q)
        #x_k = CG(v, x_k, gamma,eta,Q)
        print(x_k[:,:,0])
        
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
    print(x_k[:,:,0])

    return best_delImgAT