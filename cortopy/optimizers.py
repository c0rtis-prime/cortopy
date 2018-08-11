__author__ = "Chinmay Rao"

import numpy as np
from cortopy.model_utils import *
#from model_utils import *


def minibatch_GD(X_train_batch, parameters, cache, L, learning_rate, batch_size, dA, dZ=None):
    #print("back-prop")
    m = batch_size
    for l in range(L,0,-1):       
        g = cache[l][2]        
        if g is not "softmax":
            dZ = dA * activate(cache[l][0], act_fn = g, derivative = True)                 
        if l > 1:
            dW = (1/m) * np.dot(dZ, cache[l-1][1].T)
        else:
            dW = (1/m) * np.dot(dZ, X_train_batch.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(parameters['W'+str(l)].T, dZ)
        
        parameters['W'+str(l)] -= learning_rate*dW
        parameters['b'+str(l)] -= learning_rate*db        
    cache = [None]    
    return parameters, cache


def momentum_GD(Velocities, batch, momentum_beta, X_train_batch, parameters, cache, L, learning_rate, batch_size, dA, dZ=None):
    m = batch_size
    if batch == 1:
        for l in range(1,L+1):
            zeros_dW = np.zeros_like(parameters['W'+str(l)])
            zeros_db = np.zeros_like(parameters['b'+str(l)])
            Velocities.append(  [zeros_dW, zeros_db]  )    
    for l in range(L,0,-1):       
        g = cache[l][2]        
        if g is not "softmax":
            dZ = dA * activate(cache[l][0], act_fn = g, derivative = True)                 
        if l > 1:
            dW = (1/m) * np.dot(dZ, cache[l-1][1].T)
        else:
            dW = (1/m) * np.dot(dZ, X_train_batch.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(parameters['W'+str(l)].T, dZ)       
        
        Velocities[l][0] = momentum_beta * Velocities[l][0] + (1-momentum_beta)*dW
        Velocities[l][1] = momentum_beta * Velocities[l][1] + (1-momentum_beta)*db               
        
        parameters['W'+str(l)] -= learning_rate * Velocities[l][0]
        parameters['b'+str(l)] -= learning_rate * Velocities[l][1]        
    cache = [None]
    return parameters, cache, Velocities