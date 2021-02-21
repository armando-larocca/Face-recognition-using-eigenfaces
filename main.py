### EIGENFACES HOMEWORK ###

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn import datasets
import os
import matplotlib.image as mpimg
import numpy as np
from sklearn.metrics import accuracy_score
from PIL import Image
import glob
from norm_pca import Norm_pca
from e_2dpca import E_2DPCA

path = "./Pain_subdataset"
cartelle = os.listdir(path)
cartelle.remove(".DS_Store")

images = []
labels = []
gray_ = []

for nome in cartelle:
    immagini = os.listdir(path+"/"+nome)
    
    if len(immagini)==26:
        immagini.remove(".DS_Store")
    
    for i in immagini:
        image = Image.open(path+"/"+nome+"/"+str(i))
        new_image = image.resize((75, 75))
        tens = np.array(new_image)
        images.append(tens)
        gray_.append( tens.dot([0.07, 0.72, 0.21]) )
        labels.append(nome)
        
images = np.array(images)
labels = np.array(labels)
gray_ = np.array(gray_)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=97 )
print("Train dimension", X_train.shape)
print("Test dimension", X_test.shape)

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(gray_, labels, test_size=0.2, random_state=97)
print("\nTrain dimension", X_train_g.shape)
print("Test dimension", X_test_g.shape)

plt.imshow(X_train[0])
plt.title("Subject n."+str(y_train[0]))
#plt.savefig("1c.eps", format='eps')
plt.show()

plt.imshow(X_train[1])
plt.title("Subject n."+str(y_train[1]))
#plt.savefig("2c.eps", format='eps')
plt.show()
            
plt.imshow(X_train[4])
plt.title("Subject n."+str(y_train[4]))
#plt.savefig("3c.eps", format='eps')
plt.show()

###

plt.imshow(X_train_g[0],cmap=plt.cm.gray)
plt.title("Subject n."+str(y_train[0]))
#plt.savefig("1g.eps", format='eps')
plt.show()

plt.imshow(X_train_g[1],cmap=plt.cm.gray)
plt.title("Subject n."+str(y_train[1]))
#plt.savefig("2g.eps", format='eps')
plt.show()
            
plt.imshow(X_train_g[4],cmap=plt.cm.gray)
plt.title("Subject n."+str(y_train[4]))
#plt.savefig("3g.eps", format='eps')
plt.show()

ticks = ["0.75","0.80","0.85","0.90","0.95"]

# Standard faces recognition with eigenfaces
acc = []
PCs = []
print("\nNormalized PCA with colorgray images")

for p in [0.75,0.80,0.85,0.90,0.95]:
    npca = Norm_pca(perc=p)
    npca.fit(X_train_g,y_train_g)
    predictions = npca.predict(X_test_g)

    cumulative = 0

    for i in range(len(predictions)) :
        if predictions[i] == y_test_g[i]:
            cumulative += 1
            
    PCs.append(str(npca.eigvecs.shape[0]))
    print("Accuracy:",cumulative/(len(predictions))*100)
    acc.append(cumulative/(len(predictions)))
    
plt.figure()
plt.plot(ticks, acc)
plt.title('Npca eigenfaces: accuracy over explained variance')
plt.xlabel('Percentage of explained variance')
plt.ylabel('Accuracy')
plt.grid()
#plt.savefig('Eigenfaces-NPCA.eps', format='eps')

plt.figure()
plt.bar(PCs,[0.75,0.80,0.85,0.90,0.95])
plt.xticks(PCs)
plt.title('Npca eigenfaces: Explained variance per number of PCs')
plt.ylabel('Percentage of explained variance')
plt.xlabel('Number of Principal components')
plt.grid()
#plt.savefig('Eigenfaces-NPCA-bar.eps', format='eps')
#plt.show()


# 2D-PCA for color faces recognition
acc = []
PCs = []
print("\n2DPCA with color images")

for p in [0.75,0.80,0.85,0.9,0.95]:
    e2pca_c = E_2DPCA(perc= p)
    e2pca_c.fit(X_train,y_train)
    predictions1 = e2pca_c.predict(X_test)
    
    cumulative = 0

    for i in range(len(predictions1)) :
        if predictions1[i] == y_test[i]:
            cumulative += 1
            
    PCs.append(str(e2pca_c.eigvecs.shape[1]))
    print("Accuracy:",cumulative/(len(predictions1))*100)
    acc.append(cumulative/(len(predictions1)))
    
plt.figure()
plt.plot(ticks, acc)
plt.title('2DPCA color eigenfaces: accuracy over explained variance')
plt.xlabel('Percentage of explained variance')
plt.ylabel('Accuracy')
plt.grid()
#plt.savefig('Eigenfaces-color_2DPCA.eps', format='eps')

plt.figure()
plt.bar(PCs,[0.75,0.80,0.85,0.90,0.95])
plt.xticks(PCs)
plt.title('2DPCA color eigenfaces: Explained variance per number of PCs')
plt.ylabel('Percentage of explained variance')
plt.xlabel('Number of Principal components')
plt.grid()
#plt.savefig('Eigenfaces-color_2DPCA-bar.eps', format='eps')
#plt.show()

# 2D-PCA for gray faces recognition
acc = []
PCs = []
print("\n2DPCA with grayscale images")

for p in [0.75,0.80,0.85,0.90,0.95]:
    e2dpca = E_2DPCA(perc=p)
    e2dpca.fit(X_train_g,y_train_g, 0)
    predictions2 = e2dpca.predict(X_test_g)

    cumulative = 0

    for i in range(len(predictions2)) :
        if predictions2[i] == y_test_g[i]:
            cumulative += 1
        
    PCs.append(str(e2dpca.eigvecs.shape[1]))
    print("Accuracy:",cumulative/(len(predictions2))*100)
    acc.append(cumulative/(len(predictions2)))
    
plt.figure()
plt.plot(ticks, acc)
plt.title('2DPCA grey eigenfaces: accuracy over explained variance')
plt.xlabel('Percentage of explained variance')
plt.ylabel('Accuracy')
plt.grid()
#plt.savefig('Eigenfaces-gray_2DPCA.eps', format='eps')

plt.figure()
plt.bar(PCs,[0.75,0.80,0.85,0.90,0.95])
plt.xticks(PCs)
plt.title('2DPCA grey eigenfaces: Explained variance per number of PCs')
plt.ylabel('Percentage of explained variance')
plt.xlabel('Number of Principal components')
plt.grid()
#plt.savefig('Eigenfaces-gray_2DPCA-bar.eps', format='eps')
plt.show()
