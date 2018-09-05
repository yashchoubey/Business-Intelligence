# # Importing the Keras libraries and packages
# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense


# classifier = Sequential()
# classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# classifier.add(Flatten())
# classifier.add(Dense(units = 128, activation = 'relu'))
# classifier.add(Dense(units = 1, activation = 'sigmoid'))
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
# test_datagen = ImageDataGenerator(rescale = 1./255)
# training_set = train_datagen.flow_from_directory('training_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
# test_set = test_datagen.flow_from_directory('test_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')

# classifier.fit_generator(training_set,steps_per_epoch = 8000,epochs = 25,validation_data = test_set,validation_steps = 2000)

image_list=[]
label_list=[]
import os,cv2,numpy as np
par_dir='/home/yash/github/t_cLASSIFICTION/Layer/layer1/'
train_labels = os.listdir(par_dir)

label_dict={'others':2, 'ruptured_skin':0, 'deformed_double':1}
# print(train_labels)
for label in train_labels:
	print par_dir+label
	images=os.listdir(par_dir+label)
	print len(images)
	for image in images:
		file= par_dir+label+'/'+image
		image = cv2.imread(file)
		# print image.shape
		image = cv2.resize(image, (200,200))
		image_list.append(image)
		label_list.append(label_dict[label])
		# print par_dir+label+image
		# break
# print len(image_list)
a=np.asarray(image_list)
a=a.reshape(27587,120000)
del image_list
from tempfile import TemporaryFile
outfile = TemporaryFile()
np.save('images_level1', a)

# print set(label_list)
import keras
label_list=keras.utils.to_categorical(label_list, num_classes=3)
# print len(label_list),type(label_list)
np.save('labels_level1', label_list)
