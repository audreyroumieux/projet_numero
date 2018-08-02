# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:23:36 2018
@author: audrey roumieux
Projet:
Description:
"""
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras import backend as K
from keras.datasets import cifar10


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

(x_cifar10_train, y_cifar10_train), (x_cifar10_test, y_cifar10_test) = cifar10.load_data()

#transformation des données couleurs en donnée noir et blanc 
gray = rgb2gray(x_cifar10_train)
#plt.imshow(gray, cmap = plt.get_cmap('gray'))


# Affichage des données
for i, img in enumerate(gray[:3*5]):
    plt.subplot(3, 5, i+1)
    plt.imshow(gray, cmap = plt.get_cmap('gray'))
    
plt.show()

# One hot encoding sur les y
y_cifar10_train = to_categorical(y_cifar10_train, 10) # il existe 10 chiffres different
y_cifar10_test = to_categorical(y_cifar10_test, 10)

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
x_cifar10_train = x_cifar10_train.reshape((60000, 28 * 28))
x_cifar10_train = x_cifar10_train.astype('float32') / 255

x_cifar10_test = x_cifar10_test.reshape((10000, 28 * 28))
x_cifar10_test = x_cifar10_test.astype('float32') / 255


#%%
mon_model = keras.models.Sequential()
# Premier couche
mon_model.add(Conv2D(64, # il y a 64 images en sorti
                     kernel_size=(3, 3), # la taille du filtre, on selectionne des matrices de 3x3
                     activation='relu', input_shape=(28, 28, 1)))

mon_model.add(Conv2D(32, 2, activation='relu'))
mon_model.add(Conv2D(32, (3, 3), activation='relu'))
mon_model.add(MaxPooling2D(pool_size=(2, 2))) # Reduit la taille de l'image

mon_model.add(Flatten()) #Reformatte les données en un vercteur
mon_model.add(Dropout(0.5)) #Methode de regularisation, empeche l'over fiting en changent les valeur
#mon_model.add(Dense(12, activation='relu'))
mon_model.add(Dense(10, activation='softmax'))
print(mon_model.summary())

mon_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#%%
mon_model.fit(x_cifar10_train, y_cifar10_train)
results = mon_model.evaluate(x_cifar10_test, y_cifar10_test)
print(results)

print(mon_model.predict(x_cifar10_test))