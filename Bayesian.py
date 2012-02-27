#!/usr/bin/env python

#Author: TOSSOU Aristide

import numpy as np;
from numpy import linalg as LA;
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class BayesRecognition:
    
    num_components=12;
    n_samples=0;
    n_dim=0;
    U=[];
    S=[];
    X_components_white=[];
    den=1;
    y_recup=[];
    y_ori=[];
    y_nn=[];
    
    def  __init__(self,n_components):
        self.num_components=n_components;
        
        
    def fit(self,X,y):
        
        # Calculating the intrapersonnel differences
        
        # Renaming the class value so that they are in the
        # range 0...n_classes
        
        # y_rename contains the new values of classes in the
        # same order as they appear in y: We map each value
        # of y to a new value and write these new value in
        # the same order as in y
        # y_ori allows us to recover the original value of the
        # new class value.It's index are the new class value
        # and for each index the value is the original value
        self.y_ori,y_rename=np.unique(y,return_inverse=True);
        
        # Defining the number of classes
        n_classes=self.y_ori.shape[0];
        
        # Defining number of samples
        self.n_samples=y.shape[0];
        self.y_nn=y;
        
        
        
        # Defining array containing index for each class
        X_class=[];
            # Initialisation
        for i in range(n_classes):
            X_class.append([]);
            
            # Filling X_class with the indices of each class
            # So the indices of the first class are in X_class[0]
            # the indices of the 2nd are in X_class[1]...
        for i in range(self.n_samples):
            X_class[y_rename[i]].append(i);
            
            
        # Converting the X_class list to a numpy array
        
        for i in range(n_classes):
            X_class[i]=np.asarray(X_class[i]);
            
        X_class=np.array(X_class);
        
        # Converting X list to a numpy array
        
        X_np=np.array(X);
        
        # Defining the dimension of X
        self.n_dim=X_np[0].size;
        
        
        # Starting the Bayesian Recognition calculation
        
        
        # Computing the intrapersonnel differences in X_diff
        # for two given images index j,k we compute  X_ele[j]-X_ele[k] as
        # well as X_ele[k]-X_ele[j] so that the mean of X_diff is zero
        
        # X_diff is a n_samples*(n_samples-1) by n_dim matrix
        
        print "Calculating the intrapersonnel difference matrix"
        
        X_diff=[];
        leng=0.;
        self.y_recup=[];
        
        for i in range(n_classes):
            X_ele=X[X_class[i]];
            l=X_ele.shape[0];
            st=0;
            for j in range(l):
                for k in range(l):
                    st+=1;
                    if(st<=10):
                        if(j!=k):
                            X_diff.append(X_ele[j]-X_ele[k]);
                            self.y_recup.append(i);
                            leng+=1.;
        
        
        X_diff=np.transpose(np.matrix(X_diff));
        
        # Computing the eigenface decomposition to reduce the dimensionality
        # of X_diff
        
        # Calculating the covariance matrix of row observations in X_diff
        # sigma is a n_dim by n_dim symetric matrix
        
        sigma=(X_diff*np.transpose(X_diff)/leng);
        
        
        
        # Computing the svd decomposition of sigma: sigma is a symetric square matrix
        # S contains the eigenvalues of sigma in decreasing order (a 1d matrix)
        # U contains the corresponding eigenvectors in its columns
        
        # U is a n_dim by n_dim matrix;  S is a 1 by n_dim array(a vector)
        
        print "Computing the pca decomposition"
        
        self.U,self.S,V=LA.svd(sigma,full_matrices=True);
        # Converting the matrix to a matrix of real numbers
        self.U=np.real(self.U);
        self.S=np.real(self.S);
        
        # Computing the pca whitening retaining the first num_components of X_diff
        
        # Epsilon is used so that the sqrt is not zero
        epsilon=0.01;
        
        print "Computing the whitening"
        
        
        self.X_components_white=np.matrix(np.diagflat(1./np.sqrt(epsilon+np.abs(self.S[0:self.num_components]))))*np.transpose(self.U[:,0:self.num_components])*np.transpose(np.matrix(X));
        self.X_components_white=np.transpose(self.X_components_white);
        #print self.X_components_white;
        
        print "Computing the normalizing denominator"
        #  Computing the normalizing denominator
       
        
        #self.den=(np.power(2.*np.pi,self.num_components/2.))*(np.sqrt(LA.det(sigma)+epsilon));
        #if(self.den==0.):
        # Good ?? I ignore the denominator since I think It won't affect the classification
        # It is the same for all likelihoods
        self.den=1;
        
    def predict(self,X_test):
        
        n_test=X_test.shape[0];
        n_samples=self.X_components_white.shape[0];
        #print self.X_components_white;
        likelihoods=[];
        result=[];
        X_test=np.matrix(X_test);
        X_test=np.transpose(X_test);
        epsilon=0.01;
        
        # Whitening each input test images
        
        print "Whitening each input test images"
        
        X_test_white=np.matrix(np.diagflat(1./np.sqrt(epsilon+np.abs(self.S[0:self.num_components]))))*np.transpose(self.U[:,0:self.num_components])*X_test;
        X_test_white=np.transpose(X_test_white);
       
        
        
        
        # calculating the likelihoods for each test image against each training images
        for i in range(n_test):
            for j in range(self.n_samples):
                
                likelihoods.append((np.exp((np.power(LA.norm(X_test_white[i]-self.X_components_white[j,:]),2)*(-0.5))))/self.den);
                
            ind=np.argmax(likelihoods,axis=0);
            result.append([self.y_nn[ind],likelihoods[ind]]);
            likelihoods=[];
        return result;
        
        
    def evaluate(y_test,y_pred,target_names,n_classes):
    
        # Quantitative evaluation of the model quality on the test set
        num=0.;
        
        # Calculating the percentage of correct classification
        for i in range(y_test.size):
            if(y_test[i]==y_pred[i]):
                num+=1.;
        print "Percentage of correct classification"
        print (num/y_test.size)*100;
        print "Target value"
        print y_test;
        print "Predict values"
        print y_pred;
        
        print "Classification report"
        print classification_report(y_test, y_pred, target_names=target_names);
        print "Confusion matrix"
        print confusion_matrix(y_test, y_pred, labels=range(n_classes));
            
        
        
        
        
        

