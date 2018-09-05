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
