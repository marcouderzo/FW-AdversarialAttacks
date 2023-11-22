import numpy as np
import matplotlib.pyplot as plt

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

    loss_values = []
    loss_l2_values = []
    loss_attack_values = []

    batch_idx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), N, replace = True) # choose N random batch indexes between [0 , nFunc] with replacement

    d_t = np.zeros(d)
    for t in range(0, N):
        gamma_t = 2/(t+8)
        g_t = np.zeros(x_t.shape)

        if grad_approx_scheme == 'RDSA':
            numerator = 4
            rho_t = 4 / ((np.power(d, 1/3)) * np.power(t+8, 1/3))
            c_t = 2 / (np.power(d, 3/2)*np.power(t+8, 1/3))
            z_t = np.random.normal(size=(x_0.shape[0], x_0.shape[1], 1))
            g_t = (obj_func.evaluate(x_t + c_t * z_t[:,:, :], batch_idx[t:t+1]) - obj_func.evaluate(x_t, batch_idx[t:t+1])) / c_t * z_t[:, :, :]
        
        if grad_approx_scheme == 'I-RDSA':
            numerator = 4
            rho_t = 4 / (np.power(1+(d/m), 1/3) * np.power(t+8, 2/3))
            c_t = (2 / np.sqrt(m)) / (np.power(d, 3/2) * np.power(t+8, 1/3))
            z_it = np.random.normal(size=(x_0.shape[0], x_0.shape[1], m))
            for i in range(m):
                g_t += (obj_func.evaluate(x_t + c_t * z_it[:,:, i:i+1], batch_idx[t:t+1]) - obj_func.evaluate(x_t, batch_idx[t:t+1])) / c_t * z_it[:,:, i:i+1]
            g_t = g_t / m
        
        if grad_approx_scheme == 'KWSA':
            numerator = 8
            rho_t = 4 / (np.power(t+8, 2/3))
            c_t = 2 / (np.power(d, 1/2) * np.power(t+8, 1/3))
            e = np.eye(d)
            for i in range(d):
                g_t = g_t + ((obj_func.evaluate(x_t + c_t * e[i, :].reshape(x_0.shape), batch_idx[t:t+1]) - obj_func.evaluate(x_t, batch_idx[t:t+1])) / c_t * e[i, :].reshape(x_0.shape))
        
        print("g_t: {}".format(type(g_t)))
        #print(g_t)

        d_t = (1 - rho_t)*d_t + rho_t * g_t.reshape(-1)

        print("d_t: {}".format(type(d_t)))

        v_t = C[:, np.argmin(np.dot(d_t, C))]
        v_t = v_t.reshape(x_0.shape)

        print("x_t: {}".format(x_t.shape))
        #print(x_t.shape)
        #print(gamma_t)
        #print(np.argmin(np.dot(d_t, C)))
        print("v_t: {}".format(v_t.shape))
        #print(v_t)
        #v_t = -np.sign(d_t)
        #v_t = v_t.reshape(x_0.shape)
        print("updated v_t: {}".format(v_t.shape))

        x_t = (1-gamma_t)*x_t + gamma_t*v_t
        
        print("updated x_t: {}".format(x_t.shape))


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
        MGR.logHandler.write('\n')


    print("Final Perturbation x_t: ", x_t[:,:,0])

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

        

        