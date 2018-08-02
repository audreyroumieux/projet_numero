# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:22:36 2018
@author: audrey roumieux
Projet: 
Description: Deep Learning tous seul sur data set mlist
"""
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.layers import Dense

# import du jeu de donnée
(x_mnist_train, y_mnist_train), (x_mnist_test, y_mnist_test) = mnist.load_data()

# Affichage des données
for i, img in enumerate(x_mnist_train[:3*5]):
    plt.subplot(3, 5, i+1)
    plt.imshow(img)
plt.show()
print(y_mnist_train[:3*5])

#
#print(y_mnist_train.shape)
#y_mnist_train = y_mnist_train.reshape((-1,1))
#print(y_mnist_train.shape)

# One hot encoding sur les y
y_mnist_train = to_categorical(y_mnist_train, 10) # il existe 10 chiffres different
y_mnist_test = to_categorical(y_mnist_test, 10)

#%%
######## Model sequentiel Fully conected
mon_model = keras.models.Sequential()
#Premier couche
mon_model.add(Dense(512, activation='relu', input_shape=(28 * 28,))) 
#strite = nb de saut de pixel d'un kernel (ici =1)
 
# une autre couche
mon_model.add(Dense(512, activation='relu'))
# derniere couche
mon_model.add(Dense(10, activation='softmax')) 
#softmax sert a normaliser les valeurs de sortie (entre 0 et 1)

print('SUMMARY model: ', mon_model.summary())

mon_model.compile(optimizer = 'rmsprop', #
              loss='binary_crossentropy', # fonction de cout
              metrics=['accuracy']) # unité de mesure de la performance


# modif la dimention si model dense <=> 
x_mnist_train = x_mnist_train.reshape((60000, 28 * 28))
x_mnist_train = x_mnist_train.astype('float32') / 255

x_mnist_test = x_mnist_test.reshape((10000, 28 * 28))
x_mnist_test = x_mnist_test.astype('float32') / 255

#entrainement du model
mon_model.fit(x_mnist_train, y_mnist_train, epochs=5)

#%%
######## Model sequentiel Convolutionel
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras import backend as K

#traitement dimention des données
x_mnist_train = x_mnist_train.astype('float32') # On convertie les données en nombre entre 0 et 1
x_mnist_train = x_mnist_train / 255
x_mnist_test = x_mnist_test.astype('float32') 
x_mnist_test = x_mnist_test / 255

print(K.image_data_format())
# change la forme de la matrice pour donnéer une pronfondeur de 1
x_mnist_train = x_mnist_train.reshape(x_mnist_train.shape[0], 28, 28, 1)
x_mnist_test = x_mnist_test.reshape(x_mnist_test.shape[0], 28, 28, 1)

print('x_mnist_train shape:', x_mnist_train.shape)
print(x_mnist_train.shape[0], 'train samples')


mon_model = keras.models.Sequential()
# Premier couche
mon_model.add(Conv2D(64, # il y a 64 images en sorti
                     kernel_size=(3, 3), # la taille du filtre, on selectionne des matrices de 3x3
                     activation='relu', input_shape=(28, 28, 1)))

#mon_model.add(Conv2D(32, 2, activation='relu'))
#mon_model.add(Conv2D(32, (3, 3), activation='relu'))

mon_model.add(MaxPooling2D()) # Reduit la taille de l'image

mon_model.add(Flatten()) # Reformatte les données en un vercteur
#mon_model.add(Dropout(0.5)) # Methode de regularisation, empeche l'over fiting en changent les valeur
mon_model.add(Dense(512, activation='relu'))
mon_model.add(Dense(10, activation='softmax'))
print(mon_model.summary())

mon_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#%%
mon_model.fit(x_mnist_train, y_mnist_train, 
              epochs = 6, # nombre de fois ou l'on traitre l'ensemble des données
              #batch_size=126,
              validation_data=(x_mnist_test, y_mnist_test)) # on valide le model sur les donnees de test

#%%

results = mon_model.evaluate(x_mnist_test, y_mnist_test) 
print("Evaluate ", results)

testPredit = mon_model.predict(x_mnist_test) # renvoi un vecteur de nombre
# comparaison entre val prédit et vrai valeur
diff = (y_mnist_test != testPredit )

#Affichagedes images pour lesquels le modèle se trompe

