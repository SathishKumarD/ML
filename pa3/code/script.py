import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from sklearn.svm import SVC
import pickle

counter = 0
def preprocess():
    """ 
     Input:
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
     - feature selection
    """
    
    mat = loadmat('mnist_all.mat'); #loads the MAT object as a Dictionary
    
    n_feature = mat.get("train1").shape[1];
    n_sample = 0;
    for i in range(10):
        n_sample = n_sample + mat.get("train"+str(i)).shape[0];
    n_validation = 1000;
    n_train = n_sample - 10*n_validation;
    
    # Construct validation data
    validation_data = np.zeros((10*n_validation,n_feature));
    for i in range(10):
        validation_data[i*n_validation:(i+1)*n_validation,:] = mat.get("train"+str(i))[0:n_validation,:];
        
    # Construct validation label
    validation_label = np.ones((10*n_validation,1));
    for i in range(10):
        validation_label[i*n_validation:(i+1)*n_validation,:] = i*np.ones((n_validation,1));
    
    # Construct training data and label
    train_data = np.zeros((n_train,n_feature));
    train_label = np.zeros((n_train,1));
    temp = 0;
    for i in range(10):
        size_i = mat.get("train"+str(i)).shape[0];
        train_data[temp:temp+size_i-n_validation,:] = mat.get("train"+str(i))[n_validation:size_i,:];
        train_label[temp:temp+size_i-n_validation,:] = i*np.ones((size_i-n_validation,1));
        temp = temp+size_i-n_validation;
        
    # Construct test data and label
    n_test = 0;
    for i in range(10):
        n_test = n_test + mat.get("test"+str(i)).shape[0];
    test_data = np.zeros((n_test,n_feature));
    test_label = np.zeros((n_test,1));
    temp = 0;
    for i in range(10):
        size_i = mat.get("test"+str(i)).shape[0];
        test_data[temp:temp+size_i,:] = mat.get("test"+str(i));
        test_label[temp:temp+size_i,:] = i*np.ones((size_i,1));
        temp = temp + size_i;
    
    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis = 0);
    index = np.array([]);
    for i in range(n_feature):
        if(sigma[i] > 0.001):
            index = np.append(index, [i]);
    train_data = train_data[:,index.astype(int)];
    validation_data = validation_data[:,index.astype(int)];
    test_data = test_data[:,index.astype(int)];

    # Scale data to 0 and 1
    train_data = train_data/255.0;
    validation_data = validation_data/255.0;
    test_data = test_data/255.0;
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return 1.0 / (1.0 + np.exp(-1.0 * z))
sigmoid_vector = np.vectorize(sigmoid)





    
def blrObjFunction(params, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    
    global counter
    counter+=1
    train_data, labeli = args;
    n_class = 10
    n_data = train_data.shape[0];
    n_feature = train_data.shape[1];
    
    #print('counte',counter)
    #print('data shape', train_data.shape)
    
    #print params.shape
    #w1 =params[0:n_feature+1].reshape((n_feature+1, 1))
    w1 = np.matrix(params)
    #print('weight shape', w1.shape)
    #print 'non zero w', np.count_nonzero(w1)
    a1 = np.hstack([np.ones(train_data.shape[0]).reshape(train_data.shape[0],1),train_data]) 
    #print( 'new data shape', a1.shape)
    #print a1[0]
    z3= np.dot(a1,w1.T) 
    #print 'non zero z3',np.count_nonzero(z3)
    #print('dot product shape',z3.shape)
    #final output
    h = sigmoid_vector(z3); 
    
    
    t1 = np.dot(labeli.T,np.log(h))
    t2 = np.dot((1-labeli).T,np.log(1-h))
    J= - np.sum(t1 + t2) #scalar
    
    error = J;
    #print(error)
    error_grad = np.dot(a1.T, h-labeli)

    #error_grad = np.dot(np.sum(h-Y),train_data);
    
    ##################
    # YOUR CODE HERE #
    ##################
    
    #print('error_grad', error_grad.shape)
    return (error, np.array(error_grad).flatten())

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    
    
    ##################
    # YOUR CODE HERE #
    ##################
    
    #print('W.shape', W.shape)
    #print('data.shape', data.shape)
    data = np.hstack([np.ones(data.shape[0]).reshape(data.shape[0],1),data]) 
    #print('data.shape', data.shape)
    z= np.dot(data,W)
    #print('z.shape', z.shape)
    a = sigmoid_vector(z); 
    
    label = a.argmax(axis=1).reshape(-1,1)
    #print(label)
    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();
counter = 0
# number of classes
n_class = 10;

# number of training samples
n_train = train_data.shape[0];

# number of features
n_feature = train_data.shape[1];

T = np.zeros((n_train, n_class));
for i in range(n_class):
    T[:,i] = (train_label == i).astype(int).ravel();
    
# Logistic Regression with Gradient Descent
W = np.zeros((n_feature+1, n_class));
initialWeights = np.zeros((n_feature+1,1));
opts = {'maxiter' : 50};
for i in range(n_class):
    labeli = T[:,i].reshape(n_train,1);
    args = (train_data, labeli);
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
    W[:,i] = nn_params.x.reshape((n_feature+1,));

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data);
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data);
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data);
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')



#pickle.dump( W, open( "params.pickle", "wb" ) )


"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################


def writeopToFile(title,train_data,train_label,validation_data,validation_label,test_data,test_label,clf):
    
    print(title)
    predicted = clf.predict(np.array(train_data))
    train_acc = str(100*np.mean((train_label == predicted).astype(float))) 
    print('\n Training set Accuracy:' + train_acc+  '%')

    predicted = clf.predict(np.array(validation_data))
    validation_acc = str(100*np.mean((validation_label == predicted).astype(float))) 
    print('\n Validation set Accuracy:' + validation_acc+ '%')

    predicted = clf.predict(np.array(test_data))
    test_acc = str(100*np.mean((test_label == predicted).astype(float))) 
    print('\n Test set Accuracy:' + test_acc+ '%')

    """with open("op.csv", "a") as myfile:
            strr = title + ','+train_acc+ ','+validation_acc+ ','+test_acc+ '\n'
            #print strr
            myfile.write(strr)"""
        



train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
train_label = train_label.ravel()
validation_label = validation_label.ravel()
test_label = test_label.ravel()


clf = SVC(kernel='linear')
clf.fit(train_data, train_label) 

writeopToFile('kernel = linear',train_data,train_label,validation_data,validation_label,test_data,test_label,clf)


clf = SVC(gamma=1.0)
clf.fit(train_data, train_label) 

writeopToFile('gamma = 1.0',train_data,train_label,validation_data,validation_label,test_data,test_label,clf)


clf = SVC()
clf.fit(train_data, train_label) 

writeopToFile('default',train_data,train_label,validation_data,validation_label,test_data,test_label,clf)

Clist=[1,10,20,30,40,50,60,70,90,100]

for cval in CList:
    clf = SVC(C=cval)
    clf.fit(train_data, train_label)
    title = 'C ='+ str(cval)
    writeopToFile(title,train_data,train_label,validation_data,validation_label,test_data,test_label,clf)




