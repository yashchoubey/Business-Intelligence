from keras.models import load_model
import cv2
import keras
import numpy as np

# label_dict={2:'others', 0:'ruptured_skin', 1:'deformed_double'}
# label_dict={2:'others',0: 'spot_hole_tip',1: 'soft_watery_patches'}
# label_dict={0:'spots_holes',1: 'Tips'}
# label_dict={2:'green', 0:'Normal', 1:'sunburn'}
# label_dict={0:'half_green', 1:'Unripe'}
layer=5

label_dict={
	1:{ 0:'ruptured_skin', 1:'deformed_double',      2:'others'},
	2:{ 0: 'spot_hole_tip',1: 'soft_watery_patches', 2:'others',},
	3:{ 0:'spots_holes',   1: 'Tips'},
	4:{ 0:'Normal',        1:'sunburn',              2:'green',},
	5:{ 0:'half_green',    1:'Unripe'}
}

model = load_model('model_tomato_vgg19_l'+str(layer)+'.h5')
import os
images=os.listdir('/home/yash/github/Tomato_Classification/test/test/Unripe/')
print len(images)
for image in images:
	print image
	image = cv2.imread('/home/yash/github/Tomato_Classification/test/test/Unripe/'+image)
	# print image
	# break
	image = cv2.resize(image, (100,100))
	# image=np.reshape()

	image=image[None,:,:,:]
	# print image.shape

	print label_dict[layer][np.argmax(model.predict(image))]

# from sklearn.utils import shuffle

# import gc
# import numpy as np
# from sklearn.model_selection import train_test_split

# for layer in model.layers: print(layer.get_config(), layer.get_weights())

# for i in range(1,6):

# 	print "#################### Level:"+str(i)+"####################"
# 	model = load_model('model_tomato_vgg19_l'+str(i)+'.h5')

# 	immatrix = np.load('/home/yash/github/t_cLASSIFICTION/Layer/images_level'+str(i)+'.npy')
# 	labal = np.load('/home/yash/github/t_cLASSIFICTION/Layer/labels_level'+str(i)+'.npy')

# 	# print immatrix.shape
# 	# print labal.shape

# 	#labal=np.array(labal)
# 	data,Label = shuffle(immatrix,labal, random_state=4)
# 	train_data = [data,Label]

# 	del immatrix,labal
# 	gc.collect()

# 	(X, y) = (train_data[0],train_data[1])
# 	del train_data
# 	gc.collect()

# 	X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.99999, random_state=4)
# 	del X, y
# 	gc.collect()

# 	X_test = X_test.reshape(X_test.shape[0], 100, 100, 3)
	 

# 	X_test = X_test.astype('float32')
# 	X_test /= (256)

# 	# print X_test.shape
# 	# print Y_test.shape

# 	score = model.evaluate(X_test, Y_test,batch_size=128, verbose=0)
	
# 	print 'Test accuracy:', score[1] 
	
# 	from sklearn.metrics import classification_report,confusion_matrix
# 	y_pred= model.predict(X_test)
# 	y_pred=np.argmax(y_pred,axis=1)

# 	print confusion_matrix(np.argmax(Y_test,axis=1), y_pred) 
	
# 	del y_pred,score,X_test, Y_test
# 	gc.collect()
