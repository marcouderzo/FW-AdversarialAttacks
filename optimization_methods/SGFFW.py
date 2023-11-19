import numpy as np

def SGFFW(x_0, N, m, obj_func, grad_approx_scheme, MGR):
    best_Loss = 1e10
    best_delImgAT = x_0  # at the end, x_t will be -> best_delImgAT
    d = x_0.shape[0]*x_0.shape[1] # number of iterations is sizes of x_0 multiplied.
    x_t = x_0.copy()
    d_t = np.zeros(d)

    #### INSERT "CONVEX SET C" HERE ####

    C = np.eye(d) * 400
    scaling_factor = 400 / d
    C = C - np.full(C.shape, scaling_factor)
    C = np.concatenate((C, -C), axis=1) # Concatenate the matrix with its negative counterpart along the horizontal axis

    ####################################

    batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), N, replace = True) # choose N random batch indexes between [0 , nFunc] with replacement

    for t in range(0, N):
        gamma_t = 2/(t+8)

        if grad_approx_scheme == 'RDSA':
            rho_t = 4 / ((np.power(d, 1/3)) * np.power(t+8, 1/3))
            c_t = 2 / (np.power(d, 3/2)*np.power(t+8), 1/3)
            z_t = np.random.normal(size=(x_0.shape[0], x_0.shape[1])) # maybe 3D (x_0.shape, 1)
            g_t = (obj_func.evaluate(x_t + c_t * z_t[:,:].reshape(x_0.shape), batch_idx[t:t+1]) - obj_func.evaluate(x_t, batch_idx[t:t+1])) / c_t * z_t[:, :].reshape(x_0.shape)
        
        if grad_approx_scheme == 'I-RDSA':
            rho_t = 4 / (np.power(1+(d/m), 1/3) * np.power(t+8, 2/3))
            c_t = (2 / np.sqrt(m)) / (np.power(d, 3/2) * np.power(t+8, 1/3))
            z_it = np.random.normal(size=(x_0.shape[0], x_0.shape[1], m))
            g_t = (1/m)* np.sum([(obj_func.evaluate(x_t + c_t * z_it[:,:, i:i+1], batch_idx[t:t+1]) - obj_func.evaluate(x_t, batch_idx[t:t+1])) / c_t * z_it[:,:, i:i+1]  for i in range(m)])
            
        
        if grad_approx_scheme == 'KWSA':
            rho_t = 4 / (np.power(t+8, 2/3))
            c_t = 2 / (np.power(d, 1/2) * np.power(t+8, 1/3))
            e = np.eye(d)
            g_t = np.sum([(obj_func.evaluate(x_t + c_t * e[i, :].reshape(x_0.shape), batch_idx[t:t+1]) - obj_func.evaluate(x_t, batch_idx[t:t+1])) / c_t * e[i, :].reshape(x_0.shape)   for i in range(d)])
            
        d_t = (1 - rho_t)*d_t + rho_t * g_t
        v_t = np.argmin(np.dot(d_t, C))

        x_t = (1-gamma_t)*x_t + gamma_t*v_t


        obj_func.evaluate(x_t,np.array([]),False)
        print('Iteration Index: ', t)
        obj_func.print_current_loss()


        if(obj_func.Loss_Overall < best_Loss):
            best_Loss = obj_func.Loss_Overall
            best_delImgAT = x_t

        MGR.logHandler.write('Iteration Index: ' + str(t))
        MGR.logHandler.write(' Query_Count: ' + str(obj_func.query_count))
        MGR.logHandler.write(' Loss_Overall: ' + str(obj_func.Loss_Overall))
        MGR.logHandler.write(' Loss_Distortion: ' + str(obj_func.Loss_L2))
        MGR.logHandler.write(' Loss_Attack: ' + str(obj_func.Loss_Attack))
        MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
        MGR.logHandler.write('\n')


    print("Final Perturbation x_t: ", x_t[:,:,0])

    return best_delImgAT

        

        