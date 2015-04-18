
# coding: utf-8

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle
import math

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    
    
    covmat =  np.cov(X, rowvar = 0)
    k = np.unique(y).size
    yflat = y.flatten().tolist()
    occm = [[p,yflat.count(p)] for p in set(yflat)]
    
    xy = np.hstack([X,y])
    
    xy = xy[xy[:,2].argsort()]
    means =np.array([])
    
    initial = 0
    final = 0
    for occ in occm:
        final = initial + occ[1]
        slicem = xy[initial:final,0:2]
        mean=slicem.mean(0)
        means = np.hstack([means,mean,occ[0]])
        initial = final
    
    means = means.reshape(-1,3)
    
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    covmat = [] 
    k = np.unique(y).size
    
    yflat = y.flatten().tolist()
    occm = [[p,yflat.count(p)] for p in set(yflat)]
    
    xy = np.hstack([X,y])
    
    xy = xy[xy[:,2].argsort()]
    means =np.array([])
    
    initial = 0
    final = 0
    for occ in occm:
        final = initial + occ[1]
        slicem = xy[initial:final,0:2]
        
        covmat.append(np.cov(slicem, rowvar = 0))
        mean=slicem.mean(0)
        means = np.hstack([means,mean,occ[0]])
        initial = final
    
    means = means.reshape(-1,3)
    
    return means,covmat

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    
    Xytest = np.hstack([Xtest,ytest])
    
    Xytest = Xytest[Xytest[:,2].argsort()]
    ytest = ytest[ytest[:,0].argsort()]
    d=2;
    yprediction = np.array([])

    for xy in Xytest:

        x= xy[0:2]
        
        pxmeanlist=np.array([])
        
        for mean in means:
            
            nmean=mean.flatten()[0:2]

            nitem = 1/(((2*math.pi)**(d/2))*np.sqrt(np.linalg.det(covmat) **(1/2)))

            covinv = np.dot(np.linalg.inv(covmat),(x-nmean))
            xmutc= np.dot((x-nmean).T,covinv )
            expitem = - xmutc/2
            px = nitem * math.exp(expitem)
            
            pxmeanlist =np.hstack([pxmeanlist,px])
            
        

        pxmeanlist = np.reshape(pxmeanlist, (means.shape[0],1))
        yprediction =np.hstack([yprediction,np.argmax(pxmeanlist)+1])
        
    
    
    yprediction = np.reshape(yprediction, ytest.shape)

    acc = 100*np.mean((yprediction == ytest).astype(float))
    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    Xytest = np.hstack([Xtest,ytest])
    

    Xytest = Xytest[Xytest[:,2].argsort()]
    ytest = ytest[ytest[:,0].argsort()]
    d=2;
    yprediction = np.array([])
    for xy in Xytest:

        x= xy[0:2]
        
        pxmeanlist=np.array([])
        counter =0
        for mean in means:
            
            nmean=mean.flatten()[0:2]

            nitem = 1/(((2*math.pi)**(d/2))*np.sqrt(np.linalg.det(covmats[counter]) **(1/2)))


            covinv = np.dot(np.linalg.inv(covmats[counter]),(x-nmean))
            xmutc= np.dot((x-nmean).T,covinv )

            expitem = - xmutc/2
            px = nitem * math.exp(expitem)
            
            pxmeanlist =np.hstack([pxmeanlist,px])
            counter+=1
        
        pxmeanlist = np.reshape(pxmeanlist, (means.shape[0],1))
        yprediction =np.hstack([yprediction,np.argmax(pxmeanlist)+1])
        
    
    
    yprediction = np.reshape(yprediction, ytest.shape)
    acc = 100*np.mean((yprediction == ytest).astype(float))
    return acc

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD  

    XtX_inv = np.linalg.inv(np.dot(X.T, X))
    w = np.dot(np.dot(XtX_inv, X.T), y)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    N = X.shape[0]
    XTX = np.dot(X.T, X)
    I = np.identity(XTX.shape[0])
    XTy = np.dot(X.T, y)
    base = (lambd * N * I) + XTX
    w = np.dot(np.linalg.inv(base), XTy)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    yMinusXw = ytest - np.dot(Xtest, w)
    numerator = np.dot(yMinusXw.T, yMinusXw)
    rmse = np.sqrt(numerator) / Xtest.shape[0]
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD       
    w = np.reshape(w,(w.shape[0],-1))
    N = X.shape[0]
    M = ((np.dot(w.T, np.dot(X.T, X)) - np.dot(y.T, X))/N) + (w.T * lambd)
    
    error_grad = np.squeeze(np.asarray(M))
    yMinusXw = y - np.dot(X, w)
    error =  (np.dot(yMinusXw.T, yMinusXw)/(2*N)) + ((lambd/2) * np.dot(w.T, w))
    error = error.flatten()
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)        
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    
    Xd = np.array([])
    for xelem in x:
        parr = np.array([])
        for pelem in xrange(p+1):
            parr = np.hstack([parr,xelem**pelem])
        
        Xd = np.hstack([Xd,parr])
    
    Xd = np.reshape(Xd,(x.shape[0],p+1))

    return Xd


# In[163]:

# Main script

# Problem 1
# load the sample data                                                                 
#X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')            
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))            

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))


# In[164]:

# Problem 2

#X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')   
X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))   

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))


# In[165]:

# Problem 3
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)


# In[166]:

# Problem 4
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# In[167]:

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]

rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))

