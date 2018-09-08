from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import PReLU
from keras.layers import MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

import gc
import numpy as np
import pandas as pd

# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np

# input image dimensions
img_rows, img_cols = 100,100#200,200

# number of channels
img_channels = 1

immatrix = np.load('images_level1.npy')
print (immatrix.shape)
labal = np.load('labels_level1.npy')
print (labal.shape)

#labal=np.array(labal)
data,Label = shuffle(immatrix,labal, random_state=4)
train_data = [data,Label]

del immatrix,labal
gc.collect()

#batch_size to train
batch_size = 60
# number of output classes
nb_classes = 3
# number of epochs to train
nb_epoch = 100

(X, y) = (train_data[0],train_data[1])
del train_data
gc.collect()
# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=4)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11, random_state=4)
del X, y
gc.collect()

X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
X_val = X_val.reshape(X_val.shape[0], 3, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
import time
 
# time.sleep(5)
print X_train.shape[0]
# n = X_train.shape[0]
# d1 = X_train[:, :n/2].astype('float')
# d2 = X_train[:, n/2:].astype('float')
# X_train = np.hstack(d1, d2)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

X_train /= (256)
X_val /= (256)
X_test /= (256)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = y_train#np_utils.to_categorical(y_train, nb_classes)
Y_val = y_val#np_utils.to_categorical(y_val, nb_classes)
Y_test = y_test#np_utils.to_categorical(y_test, nb_classes)

del y_train,y_val,y_test


model = Sequential() 		
model.add(Convolution2D(128, (3, 3),input_shape=(3,img_rows,img_cols)))
model.add(PReLU(alpha_initializer = 'zeros',weights = None))
#model.add(BatchNormalization())
model.add(Convolution2D(128, (3, 3), data_format='channels_first'))
#model.add(Dropout(0.5))	
model.add(PReLU(alpha_initializer = 'zeros',weights = None))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides=(2,2)))
	

model.add(Convolution2D(64, (3, 3), data_format='channels_first'))
#model.add(Dropout(0.5))	
model.add(PReLU(alpha_initializer='zeros',weights = None))
#model.add(BatchNormalization())

# model.add(Convolution2D(64, (3, 3), data_format='channels_first'))
# #model.add(Dropout(0.5))	
# model.add(PReLU(alpha_initializer='zeros',weights = None))
#model.add(BatchNormalization()) 
model.add(MaxPooling2D((2,2), strides=(2,2)))


model.add(Flatten())
model.add(Dense(256))
model.add(PReLU(alpha_initializer = 'zeros',weights = None))
#model.add(Dropout(0.5))		
num_classes=3
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_val, Y_val))
           

# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.subplot(211)
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#plt.figure(2,figsize=(7,5))
plt.subplot(212)
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.tight_layout()
plt.show()



score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])


# Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix
y_pred = model.predict_classes(X_test)
p=model.predict_proba(X_test) # to predict probability
target_names = ['class 0(benign)', 'class 1(malignant)']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))
