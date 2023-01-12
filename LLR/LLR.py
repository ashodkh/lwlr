import numpy as np
from sklearn.neighbors import KDTree

def LLR(x1, x1_train, y1_train, nn, weight, extra_weights = None):
    '''
    Local linear regression with inverse distance weight and nn number of nearest neighbors.
    
    Input
    -----
    
    x1: Matrix of features (inputs) in the shape (number of data points, number of features). Outcome will be evaluated at these points.
    
    x1_train: Matrix of features used for training in the shape of (number of data points, number of features).
    
    y1_train: Matrix of outcomes used for training in the shape of (number of data points, number of outcomes).
    
    nn: Number of nearest neighbors to include for each point.
    
    weight: weight of nearest neighbors. 
            -'constant' sets an equal constant weight for all neighbors.
            -'inverse_distance' sets an inverse distance weight.
            -'inverse_distance_squared' sets an inverse distance squared weight.
    Output
    ------
    
    y_fit: Matrix of outcomes predicte in the same shape as y1_train.
    
    zeros: indices corresponding to points in x1 that have nearest neighbor at zero distance. If the data is good quality, this shouldn't happen.
    
    '''
    
    ones = np.ones([x1.shape[0],1])
    x1 = np.concatenate((ones,x1), axis=1)
    ones = np.ones([x1_train.shape[0],1])
    x1_train = np.concatenate((ones,x1_train), axis=1)
    
    nl = nn
    tree = KDTree(x1_train[:,:])
    dist, ind = tree.query(x1[:,:], k=nl)

    # removing points on top of each other
    
    zeros = np.where(dist==0)[0]
    print(zeros)
    dist = np.delete(dist, obj=zeros, axis=0)
    ind = np.delete(ind, obj=zeros, axis=0)
    x1 = np.delete(x1, obj=zeros, axis=0)
    n_valid = x1.shape[0]

    # Fitting the coefficients based on the analytical solution

    theta = np.zeros([n_valid,x1.shape[1],1])
    W = np.zeros([n_valid,nl,nl])
    X = np.zeros([n_valid,nl,x1.shape[1]])
    Y = np.zeros([n_valid,nl,1])
    if extra_weights is None:
        extra_weights = np.ones(nl)
    if weight == 'constant':
        for j in range(nl):
            W[:,j,j] = 1*extra_weights[j]
            X[:,j,:] = x1_train[ind[:,j],:]
            Y[:,j,0] = y1_train[ind[:,j]]
    elif weight == 'inverse_distance':
        for j in range(nl):
            W[:,j,j] = 1/dist[:,j]*extra_weights[j]
            X[:,j,:] = x1_train[ind[:,j],:]
            Y[:,j,0] = y1_train[ind[:,j]]
    elif weight == 'inverse_distance_squared':
        for j in range(nl):
            W[:,j,j] = 1/dist[:,j]**2*extra_weights[j]
            X[:,j,:] = x1_train[ind[:,j],:]
            Y[:,j,0] = y1_train[ind[:,j]]
    else:
        raise ValueError("Weight does not match one of the three options")
    a1 = np.zeros([n_valid,x1.shape[1],1])
    a2 = np.zeros([n_valid,x1.shape[1],x1.shape[1]])
    y_fit = np.zeros(n_valid)
    for ii in range(n_valid):
        a1[ii,:,:] = np.matmul(X[ii,:,:].transpose(),np.matmul(W[ii,:,:],Y[ii,:,:]))
        a2[ii,:,:] = np.matmul(X[ii,:,:].transpose(),np.matmul(W[ii,:,:],X[ii,:,:]))
        theta[ii,:,:] = np.matmul(np.linalg.inv(a2[ii,:,:]),a1[ii,:,:])
        y_fit[ii] = np.matmul(theta[ii,:,:].transpose(),x1[ii,:])
        
    return y_fit, zeros

def LLR_slow(x1, x1_train, y1_train, nn, weight, extra_weights = None):
    '''
    Local linear regression with inverse distance weight and nn number of nearest neighbors.
    
    Input
    -----
    
    x1: Matrix of features (inputs) in the shape (number of data points, number of features). Outcome will be evaluated at these points.
    
    x1_train: Matrix of features used for training in the shape of (number of data points, number of features).
    
    y1_train: Matrix of outcomes used for training in the shape of (number of data points, number of outcomes).
    
    nn: Number of nearest neighbors to include for each point.
    
    weight: weight of nearest neighbors. 
            -'constant' sets an equal constant weight for all neighbors.
            -'inverse_distance' sets an inverse distance weight.
            -'inverse_distance_squared' sets an inverse distance squared weight.
    Output
    ------
    
    y_fit: Matrix of outcomes predicte in the same shape as y1_train.
    
    zeros: indices corresponding to points in x1 that have nearest neighbor at zero distance. If the data is good quality, this shouldn't happen.
    
    '''
    ones = np.ones([x1.shape[0],1])
    x1 = np.concatenate((ones,x1), axis=1)
    ones = np.ones([x1_train.shape[0],1])
    x1_train = np.concatenate((ones,x1_train), axis=1)
    
    nl = nn
    tree = KDTree(x1_train[:,:])
    dist, ind = tree.query(x1[:,:], k=nl)

    # removing points on top of each other
    
    zeros = np.where(dist==0)[0]
    print(zeros)
    dist = np.delete(dist, obj=zeros, axis=0)
    ind = np.delete(ind, obj=zeros, axis=0)
    x1 = np.delete(x1, obj=zeros, axis=0)
    n_valid = x1.shape[0]

    # Fitting the coefficients based on the analytical solution

    theta = np.zeros([x1.shape[1],1])
    W = np.zeros([nl,nl])
    X = np.zeros([nl,x1.shape[1]])
    Y = np.zeros([nl,1])
    y_fit = np.zeros(n_valid)
    if extra_weights is None:
        extra_weights = np.ones(nl)
    for i in range(n_valid):
        if weight == 'constant':
            for j in range(nl):
                W[j,j] = 1*extra_weights[j]
                X[j,:] = x1_train[ind[i,j],:]
                Y[j,0] = y1_train[ind[i,j]]
        elif weight == 'inverse_distance':
            for j in range(nl):
                W[j,j] = 1/dist[i,j]*extra_weights[j]
                X[j,:] = x1_train[ind[i,j],:]
                Y[j,0] = y1_train[ind[i,j]]
        elif weight == 'inverse_distance_squared':
            for j in range(nl):
                W[j,j] = 1/dist[i,j]**2*extra_weights[j]
                X[j,:] = x1_train[ind[i,j],:]
                Y[j,0] = y1_train[ind[i,j]]
        else:
            raise ValueError("Weight does not match one of the three options")
        a1 = np.zeros([x1.shape[1],1])
        a2 = np.zeros([x1.shape[1],x1.shape[1]])

        a1[:,:] = np.matmul(X[:,:].transpose(),np.matmul(W[:,:],Y[:,:]))
        a2[:,:] = np.matmul(X[:,:].transpose(),np.matmul(W[:,:],X[:,:]))
        theta[:,:] = np.matmul(np.linalg.inv(a2[:,:]),a1[:,:])
        y_fit[i] = np.matmul(theta[:,:].transpose(),x1[i,:])
        
    return y_fit, zeros