#!/usr/bin/env python

# Author: TOSSOU Aristide

import os;
from PIL import Image;
import numpy as np;

# Helper to load the Extended Yale Database
# data_home is the directory containing the list of directories corresponding to faces
# num_faces number of fdifferent people to take
# num_train number of faces per personn for training
# num_test number of faces per personn for testing
# h,w dimension for reducing the images

def loadD(data_home='.',num_faces=10,num_train=10,num_test=3,h=64,w=64,extension=('.jpeg','.jpg','.png','.pgm')):
    X_train=[];
    X_test=[];
    y_train=[];
    y_test=[];
    rep=[];
    target_names=[];
    k=0;
    for dirname, dirnames, filenames in os.walk(data_home):
        for subdirname in dirnames:
            rep.append(os.path.join(dirname,subdirname));
    l=len(rep);
    for i in range(l):
        if(i>num_faces):
            break;
        target_names.append(os.path.basename(rep[i]));
        k=0;
        for dirname, dirnames, filenames in os.walk(rep[i]):
            # Iterating over files in a given directory
           
                
            for filename in filenames:
                k+=1;
                if(k<=num_train+num_test):
                    if(k<=num_train):
                        
                        str1=os.path.join(dirname,filename);
                        
                        if((str1.endswith(extension))==False):
                            k-=1;
                            continue;
                        
                        
                        im=Image.open(str1);
                        im=im.resize((h,w));
                        X_train.append(np.asarray(im));
                        y_train.append(i);
                    else:
                        str1=os.path.join(dirname,filename);
                        if((str1.endswith(extension))==False):
                            k-=1;
                            continue;
                        im=Image.open(str1);
                        im=im.resize((h,w));
                        X_test.append(np.asarray(im));
                        y_test.append(i);
    # Transformation of the images
    l=len(X_train);
    X_train=np.array(X_train);
    X_train=np.reshape(X_train,(l,h*w));
    X_train=np.float32(X_train);
    
    l=len(X_test);
    X_test=np.array(X_test);
    
    X_test=np.reshape(X_test,(l,h*w));
    X_test=np.float32(X_test);
    
    y_test=np.array(y_test);
    y_train=np.array(y_train);
    target_names=np.array(target_names);
    
    return X_train,y_train,X_test,y_test,target_names;