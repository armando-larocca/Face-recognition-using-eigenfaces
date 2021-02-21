### 2DPCA FUNCTION ###

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from numpy import linalg as LA
import os
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import glob


class E_2DPCA:
    def __init__(self,perc):
        self.perc = perc
        self.mean = []
        self.y = []
        self.eigvecs = []
        self.eigvals = []
        self.features =[]
        self.color = 1
        
        
    
    def matrix_generation(self,data):
        red_  = data[:,:,:,0]
        green_ = data[:,:,:,1]
        blue_ = data[:,:,:,2]
        
        red_ = np.reshape(red_, [red_.shape[0],1,red_.shape[1]*red_.shape[2]])
        green_ = np.reshape(green_, [green_.shape[0],1,green_.shape[1]*green_.shape[2]])
        blue_ = np.reshape(blue_, [blue_.shape[0],1,blue_.shape[1]*blue_.shape[2]])
        
        matrix = np.append(red_,green_,axis=1)
        matrix = np.append(matrix,blue_,axis=1)

        return matrix
    
        
    def fit(self,data,y,color = 1):
        self.y = y
        self.color = color
        
        if self.color == 1:
            matrix = self.matrix_generation(data)
        else : matrix=data
        
        ## 2DPCA ##
        self.mean = np.mean(matrix,0)
        matrix_mean = (matrix - self.mean)
        
        #Covariance matrix
        scatter_matrix = np.zeros((matrix.shape[2], matrix.shape[2]))
        for i in range(0, matrix.shape[0]):
            scatter_matrix += (np.dot(matrix_mean[i,:,:].T, matrix_mean[i,:,:]))
        scatter_matrix /= matrix.shape[0]
            
        lambda_, e_vec = np.linalg.eig(scatter_matrix)
        
        #Component choice
        p = self.give_p(lambda_)
        print("Number of PCs':", p)
        self.eigvecs = e_vec[:, 0:p]
        self.eigvals = lambda_
        
        #Transformation
        self.features = np.dot(matrix_mean, self.eigvecs)
        
    
    def predict(self,test):
        if self.color == 1:
            matrix = self.matrix_generation(test)
        else : matrix=test
        
        matrix_mean = (matrix - self.mean)
        x_test = np.dot(matrix_mean, self.eigvecs)
        predictions = self.NN(x_test)
        
        return predictions
            
            
    def NN(self, features):
        predictions = []

        for f in features:
            dist = LA.norm(self.features - f, axis=1)
            dist = np.sum(dist,axis=1)
            predictions.append( self.y[np.argmin(dist)] )
    
        predictions = np.array(predictions)
        return predictions
        
        
    def give_p(self, d):
        sum = np.sum(d)
        sum_perc = self.perc * sum
        temp = 0
        p = 0
        while temp < sum_perc:
            temp += d[p]
            p += 1
        return p
