import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from scipy.misc import imresize
import random
import math
import datetime
import pickle


debug=False
test_size_limit=20;

obj_list=[]



def resizeImage(arr,row,col):
    
    arr = imresize(arr, (row,col), 'bilinear')
    return arr

resizeImage_vector = np.vectorize(resizeImage)

def removeZeros(a):
    arr = np.array(a)
    arr.shape = (28,28)
    zero_rows = np.nonzero(arr.sum(axis=1) == 0) 
    zero_cols = np.nonzero(arr.sum(axis=0) == 0) 
    arr = np.delete(arr, zero_rows, axis=0) 
    arr = np.delete(arr, zero_cols, axis=1) 
    return arr, arr.shape

removeZeros_vector = np.vectorize(removeZeros)

def normalize(arr):
    k = np.divide(arr.astype(float),255)
    #arr[arr < 45] = 0
    #arr[arr >= 45] = 1
    return k
normalize_vector = np.vectorize(normalize)

def extractFeature(arr,row,col):
    features =[]
    dimensions=[1]
    rowindex = colindex =0
    for dim in dimensions:
        rowindex =0
        while rowindex <row:
            colindex =0
            while colindex <col:
                sub =arr[rowindex:rowindex+ dim,colindex:colindex+ dim]
                sum = np.sum(sub)
                #print rowindex, colindex, dim,sum
                features.append(sum)
                colindex = colindex + dim
            rowindex = rowindex + dim 
    #features.add(1)
    return features

extractFeature_vector = np.vectorize(extractFeature)

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return 1.0 / (1.0 + np.exp(-1.0 * z))
sigmoid_vector = np.vectorize(sigmoid)

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_derivative_vector = np.vectorize(sigmoid_derivative)

    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    #mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    mat = loadmat('C:\\Users\\Sathish\\Dropbox\\UB_Spring_2014\\ML 574\\basecode\\basecode\\mnist_al.mat')
    
    dimensions=[]
    train_data_list =[]
    trainnmMatrices=[]
    train_label_list = []
    validation_data_list = []
    validation_label_list =[]
    vaidatenmMatrices=[]
    test_data_list =[]
    test_label_list =[]
    testnmMatrices=[]

    
    for key, value in mat.iteritems():
        if "train" in key:
            digit = int(key.replace("train",""))
            a = range(test_size_limit)  if debug else range(value.shape[0])
            aperm = np.random.permutation(a)
            #validate = value[aperm[0:test_size_limit/2],:] if debug else value[aperm[0:1000],:]
            #train = value[aperm[test_size_limit/2:],:] if debug else value[aperm[1000:],:]
            
            validate = value[aperm[0:1000],:]
            train = value[aperm[1000:],:]
            #print "validate" ,validate
            for row in train:
                arr,shape = removeZeros(row)
                dimensions.append(shape)
                #nm = normalize(arr)
                trainnmMatrices.append({'d':digit,'m':arr})
            
            for row in validate:
                arr,shape = removeZeros(row)
                dimensions.append(shape)
                #nm = normalize(arr)
                vaidatenmMatrices.append({'d':digit,'m':arr})
                
        if "test" in key:
            
            digit = int(key.replace("test",""))
            a = range(test_size_limit)
            #value = value[a[0:test_size_limit],:] if debug else value
            
            for row in value:
                
                arr,shape = removeZeros(row)
                dimensions.append(shape)
                nm = normalize(arr)
                testnmMatrices.append({'d':digit,'m':nm})
    
    #print dimensions
    maxRowD, maxColD = np.amax(dimensions,axis=0)
    #print "max dim", maxRowD, maxColD
   
    random.shuffle(trainnmMatrices)
    random.shuffle(vaidatenmMatrices)
    random.shuffle(testnmMatrices)
    
    
    
    for x in trainnmMatrices:
        img = x.get('m')
        #print "img", img
        r =resizeImage(img,maxRowD,maxColD)
        #print "resizeImage" ,r
        r = normalize(r)
        f= extractFeature(r,maxRowD,maxColD)
        #print "extractFeature",f
        train_data_list.append(f)
        train_label_list.append(x.get('d'))
        
    for x in vaidatenmMatrices:
        validation_data_list.append(extractFeature(normalize(resizeImage(x.get('m'),maxRowD,maxColD)),maxRowD,maxColD))
        validation_label_list.append(x.get('d'))
        
    for x in testnmMatrices:
        test_data_list.append(extractFeature(normalize(resizeImage(x.get('m'),maxRowD,maxColD)),maxRowD,maxColD))
        test_label_list.append(x.get('d'))
    
    #print "train_data" ,train_data_list
    train_data = np.array(train_data_list)
    train_label = np.array(train_label_list)
    validation_data = np.array(validation_data_list)
    validation_label = np.array(validation_label_list)
    test_data = np.array(test_data_list)
    test_label = np.array(test_label_list)
    
    #print trainnmMatrices[0]
    #print test_data[0]
    print "train_data inputd" ,train_data.shape
    print "train_label inputd" ,train_label.shape
    
    print "validation_data inputd" ,validation_data.shape
    print "validation_label inputd" ,validation_label.shape
    
    print "test_data inputd" ,test_data.shape
    print "test_label inputd" ,test_label.shape
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    input_data_size = training_data.shape[0]
    #print "w1: ", w1.shape
    #print "w2: ",w2.shape
    #w1:  (50L, 105L)
    #w2:  (10L, 51L)
    #print "training_data.shape", training_data.shape #(50000L, 104L)
    #print "training_label.shape", training_label.shape #(1L, 50000L)
    
    obj_val = 0  
    
    I= np.identity(n_class) #(10L,10L)
    Y=np.zeros((input_data_size,n_class)) #(50000L,10L)
    
    # '1' is the maximum probability
    # if a digit is 6 then , all other output nodes should be 0 except 6th node
    # i.e, it will look like [0,0,0,0,0,1,0,0,0,0]
    
    for i in range(input_data_size):
        Y[i, :]= I[training_label[i], :];
    
    #adding input bias(1) at the zeroth column
    a1 = np.hstack([np.ones(training_data.shape[0]).reshape(training_data.shape[0],1),training_data]) #(50000L,105L)
    
    
    dw1=np.zeros(w1.shape) #(50L, 105L)
    dw2=np.zeros(w2.shape) #(10L, 51L)
    
   
    z2= np.dot(a1 , w1.T) #(50000L,50L) since  (50000L,105L) * (105L,50L)
    
    #adding hidden bias (1) at the zeroth column
    #output of the hidden nodes
    a2 =  np.hstack([np.ones(z2.shape[0]).reshape(-1,1),sigmoid_vector(z2)]) # (50000L,51L)
    
    z3= np.dot(a2,w2.T) #(50000L,10L) since (50000L,51L) * (51l,10L)
    
    #final output
    h =a3 = sigmoid_vector(z3); #(50000L,10L)
    
    reg =(lambdaval/(2*input_data_size))*(np.sum(np.square(w1[:,1:])) + np.sum(np.square(w2[:,1:]))); #scalar
    
    J= (-1/input_data_size)*np.sum(Y*np.log(h) + (1-Y)*np.log(1-h)) #scalar
    
    obj_val = J+reg #scalar
    
    
    #error function a3 is actual Y is expected
    s3 = a3 - Y; #(50000L,10L)
    
    #obj_val = np.sum(s3*s3)/(2*input_data_size)
    
    
    z21= np.hstack([np.ones(z2.shape[0]).reshape(-1,1),z2])  #(50000L,51L)
    
    #start backpropagating
    #(50000L,10L)*(10L, 51L) = (50000L,51L)
    s2 = (np.dot(s3,w2) * sigmoid_derivative_vector(z21))[:,1:] #(50000L,50L)
    
    
    d1=np.dot(s2.T,a1) # (50L, 105L) since (50L,50000L)*(50000L,105L)
    d2=np.dot(s3.T,a2) # (10L,51L) (10L,50000L) * (50000L,51L)
    
    gw1Zeros = np.hstack([np.zeros(w1.shape[0]).reshape(-1,1),w1[:,1:]]) # (50L, 105L)
    gw2Zeros = np.hstack([np.zeros(w2.shape[0]).reshape(-1,1),w2[:,1:]]) # (10L,51L)
    
    gw1 = (d1/input_data_size) + (lambdaval/input_data_size)*gw1Zeros #(50L, 105L)
    gw2 = (d2/input_data_size) + (lambdaval/input_data_size)*gw2Zeros #(10L,51L)
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    
    
    obj_grad = np.concatenate((gw1.flatten(), gw2.flatten()),0)
    #obj_grad = np.array([])
    
    #print "obj_grad ", obj_grad.shape
    obj_list.append(obj_val)
    #print "object_val",obj_val
    return (obj_val,obj_grad)


def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    
    #w1:  (50L, 105L)
    #w2:  (10L, 51L)
    #hiddenbias=inputbias=1
   
    input_size = data.shape[0]
    output_size = w2.shape[0]
    p = np.zeros((input_size, 1));
    
    a1 = np.hstack([np.ones(data.shape[0]).reshape(data.shape[0],1),data]) #(50000L,105L)
    z2= np.dot(a1 , w1.T )#(50000L,50L) since  (50000L,105L) * (105L,50L)
    
    #adding hidden bias (1) at the zeroth column
    #output of the hidden nodes
    a2 =  np.hstack([np.ones(z2.shape[0]).reshape(z2.shape[0],1),sigmoid_vector(z2)]) # (50000L,51L)
    z3= np.dot(a2,w2.T) #(50000L,10L) since (50000L,51L) * (51l,10L)
    #final output
    a3 = sigmoid_vector(z3); #(50000L,10L)
    #a3.argmax(axis=1).reshape(-1,1)
    
    #Your code here
    #print "labels shape" ,labels.shape
    #print data[0],data[3]
    #print a3[0],a3[1]
    l = a3.argmax(axis=1)
    #print "prediction shape" ,l.shape
    #print l
    return l
    

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 12;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.1;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')