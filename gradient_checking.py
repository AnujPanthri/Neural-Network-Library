import numpy as np
import copy

def g_c(model,x,y):
    theta_vec,bias_vec=model.get_parameters()
    estimated_grads=np.zeros_like(theta_vec)
    estimated_bias_grads=np.zeros_like(bias_vec)
    loss=model.loss.forward
    esp=1e-7
    for i in range(len(theta_vec)):
        theta_pos=copy.deepcopy(theta_vec)
        theta_neg=copy.deepcopy(theta_vec)
        theta_pos[i]+=esp
        theta_neg[i]-=esp
        
        model.set_parameters(theta_pos,bias_vec)    
        y_pred=model.predict(x)
        loss_pos=loss(y,y_pred)

        model.set_parameters(theta_neg,bias_vec)    
        y_pred=model.predict(x)
        loss_neg=loss(y,y_pred)

        estimated_grads[i]=(loss_pos-loss_neg)/(2*esp)

    for i in range(len(bias_vec)):
        bias_pos=copy.deepcopy(bias_vec)
        bias_neg=copy.deepcopy(bias_vec)
        bias_pos[i]+=esp
        bias_neg[i]-=esp
        
        model.set_parameters(theta_vec,bias_pos)    
        y_pred=model.predict(x)
        loss_pos=loss(y,y_pred)

        model.set_parameters(theta_vec,bias_neg)    
        y_pred=model.predict(x)
        loss_neg=loss(y,y_pred)

        estimated_bias_grads[i]=(loss_pos-loss_neg)/(2*esp)


    
    
    model.set_parameters(theta_vec,bias_vec)
    return estimated_grads,estimated_bias_grads


def get_grads(model,x,y):
    
    model.forward(x)
    y_pred=model.predict(x)
    model.loss.forward(y,y_pred)
    model.backward(y)
    grads=model.get_gradients()
    return grads

def check(grads,estimated_grads):
    grads=np.r_[grads[0],grads[1]]  # theta , bias
    estimated_grads=np.r_[estimated_grads[0],estimated_grads[1]]
    numerator = np.linalg.norm(grads - estimated_grads)                       # Step 1'
    denominator = np.linalg.norm(grads) + np.linalg.norm(estimated_grads)     # Step 2'
    difference = numerator / denominator                                      # Step 3'
    ### END CODE HERE ###
    if difference < 1e-6:
        print ("The gradient is correct!")
    else:
        print ("The gradient is wrong!")
    print("difference:",difference)


def gradient_checking(model,X,y):
    '''gradient checking is to makes sure backpropagation is implemented correctly'''
    estimated_grads=g_c(model,X,y)
    grads=get_grads(model,X,y)
    check(grads,estimated_grads)