### NORM_PCA FUNCTION ###

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy import linalg as LA
import os
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import glob


class Norm_pca:
    def __init__(self,perc = 0.95):
        self.perc = perc
        self.mean = 0
        self.std = 0
        self.train_weights = []
        self.y = []
        self.eigvecs = []
        self.eigvals = []
        self.pca = None


    def fit(self, data, y):
        data = np.reshape(data,[data.shape[0],data.shape[1]*data.shape[2]])
        
        self.mean = np.mean(data,axis=0)
        self.std = np.std(data,axis=0)

        normalized = (np.abs(data-self.mean)/self.std)

        col_mean = np.mean(normalized, axis=1)
        col_mean = np.expand_dims(col_mean, axis=-1)
        train_centered_images = normalized - col_mean

        pca = PCA(n_components=self.perc, svd_solver='full')
        pca.fit(train_centered_images)
        self.pca = pca
        print("Number of PCs':",pca.components_.shape[0])
        self.eigvecs = pca.components_
        self.train_weights = pca.transform(train_centered_images)

        self.y = y


    def predict(self, data):
        prova = []
        for x in data:
            prova.append( np.reshape(x,[x.shape[0]*x.shape[1]])  )
        data = np.array(prova)
        normalized = (np.abs(data-self.mean)/self.std)

        col_mean = np.mean(normalized, axis=1)
        col_mean = np.expand_dims(col_mean, axis=-1)
        test_centered_images = normalized - col_mean

        self.test_weights = self.pca.transform(test_centered_images)

        predictions = []

        for y in self.test_weights:
            dist = LA.norm(self.train_weights - y, axis=1)
            predictions.append( self.y[np.argmin(dist)] )
    
        predictions = np.array(predictions)
        
        return predictions

