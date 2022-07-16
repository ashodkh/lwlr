import numpy as np
from sklearn.neighbors import KDTree

def LLR(x1, x1_train, y1_train, nn, weight):
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
    
    nl=nn
    tree=KDTree(x1_train[:,:])
    dist, ind=tree.query(x1[:,:],k=nl)

    # removing points on top of each other
    
    zeros=np.where(dist==0)[0]
    print(zeros)
    dist=np.delete(dist,obj=zeros,axis=0)
    ind=np.delete(ind,obj=zeros,axis=0)
    x1=np.delete(x1,obj=zeros,axis=0)
    n_valid=x1.shape[0]

    # Fitting the coefficients based on the analytical solution
    
    theta=np.zeros([n_valid,x1.shape[1],1])
    W=np.zeros([n_valid,nl,nl])
    X=np.zeros([n_valid,nl,x1.shape[1]])
    Y=np.zeros([n_valid,nl,1])
    if weight=='constant':
        for j in range(nl):
            W[:,j,j]=1
            X[:,j,:]=x1_train[ind[:,j],:]
            Y[:,j,0]=y1_train[ind[:,j]]
    else if weight=='inverse_distance':
        for j in range(nl):
            W[:,j,j]=1/dist[:,j]
            X[:,j,:]=x1_train[ind[:,j],:]
            Y[:,j,0]=y1_train[ind[:,j]]
    else if weight=='inverse_distance_squared':
        for j in range(nl):
            W[:,j,j]=1/dist[:,j]**2
            X[:,j,:]=x1_train[ind[:,j],:]
            Y[:,j,0]=y1_train[ind[:,j]]
    else:
        raise ValueError("Weight does not match one of the three options")
    a1=np.zeros([n_valid,x1.shape[1],1])
    a2=np.zeros([n_valid,x1.shape[1],x1.shape[1]])
    y_fit=np.zeros(n_valid)
    for ii in range(n_valid):
        a1[ii,:,:]=np.matmul(X[ii,:,:].transpose(),np.matmul(W[ii,:,:],Y[ii,:,:]))
        a2[ii,:,:]=np.matmul(X[ii,:,:].transpose(),np.matmul(W[ii,:,:],X[ii,:,:]))
        theta[ii,:,:]=np.matmul(np.linalg.inv(a2[ii,:,:]),a1[ii,:,:])
        y_fit[ii]=np.matmul(theta[ii,:,:].transpose(),x_valid[ii,:])
        
    return y_fit, zeros
