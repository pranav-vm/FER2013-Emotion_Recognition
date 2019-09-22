import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D , MaxPooling2D, AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt

num_features=48
num_labels=7
batch_size=  #256 could also be used
epochs=30
width, height=48,48

x=np.load('./Data_x.npy')
y=np.load('./Labels.npy')

x=x-np.mean(x, axis=0)
x /= np.std(x, axis=0)
"""
for xx in range(10):
    plt.figure(xx)
    plt.imshow(x[xx].reshape((48, 48)), interpolation='none', cmap='gray')
plt.show()
"""
#splitting into training, validation and testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=4)

#saving the test samples to be used later
np.save('modXtest', X_test)
np.save('modytest', y_test)

model= Sequential()
model.add(Conv2D(num_features, kernel_size=(5,5), activation='relu', input_shape=(width, height, 1), kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_labels, activation='softmax'))
#model.summary() Helps us check the summary of the model.

#Compliling the model with adam optimixer and categorical crossentropy loss
model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7), metrics=['accuracy'])


#training the model
#For efficient training use a GPU on Colab or something
""" Checking GPU availability in colab
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
"""
model.fit(np.array(X_train), np.array(y_train), batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(np.array(X_valid), np.array(y_valid)), shuffle=True)

#saving the  model to be used later
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
print("Saved model to disk")
