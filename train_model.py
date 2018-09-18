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


 
# input image dimensions
img_rows, img_cols = 100,100#200,200

# number of channels
img_channels = 1

level=[2,3,4,5]

for i in level:

	immatrix = np.load('images_level'+str(i)+'.npy')
	print (immatrix.shape)
	labal = np.load('labels_level'+str(i)+'.npy')
	print (labal.shape)

	#labal=np.array(labal)
	data,Label = shuffle(immatrix,labal, random_state=4)
	train_data = [data,Label]

	del immatrix,labal
	gc.collect()

	(X, y) = (train_data[0],train_data[1])
	del train_data
	gc.collect()
	# STEP 1: split X and y into training and testing sets

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=4)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11, random_state=4)
	del X, y
	gc.collect()

	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
	X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 3)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
	import time
	 
	# time.sleep(5)
	# print X_train.shape[0]
	X_train = X_train.astype('float32')
	X_val = X_val.astype('float32')
	X_test = X_test.astype('float32')

	X_train /= (256)
	X_val /= (256)
	X_test /= (256)

	# print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	Y_train = y_train#np_utils.to_categorical(y_train, nb_classes)
	Y_val = y_val#np_utils.to_categorical(y_val, nb_classes)
	Y_test = y_test#np_utils.to_categorical(y_test, nb_classes)

	del y_train,y_val,y_test


	################################################################
	#class_weight to use in case of skewed dataset
	from sklearn.utils.class_weight import compute_class_weight
	y_integers = np.argmax(Y_train, axis=1)
	class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
	d_class_weights = dict(enumerate(class_weights))

	nb_classes=len(d_class_weights.keys())



	from keras import applications
	from keras.models import Sequential, Model 
	from keras import optimizers

	model = applications.VGG19(weights = "imagenet", include_top=False,input_shape = (img_rows, img_cols,3))

	for layer in model.layers[:12]:
		layer.trainable = False

	#Adding custom Layers 
	x = model.output
	x = Flatten()(x)
	x = Dense(1024, activation="relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(512, activation="relu")(x)
	predictions = Dense(nb_classes, activation="softmax")(x)

	# creating the final model 
	model= Model(input = model.input, output = predictions)

	# compile the model 
	model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

	################################################################

	hist = model.fit(
		X_train, 
		Y_train, 
		batch_size=128, 
		epochs=100, 
		verbose=1, 
		validation_data=(X_val, Y_val),
		class_weight = d_class_weights, 
		#callbacks=callbacks_list
	)

	################################################################

	model.save("model_tomato_vgg19_l"+str(i)+".h5")
	del hist,model,X_train, Y_train,X_val, Y_val
	gc.collect()
	################################################################
	
	