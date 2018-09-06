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
		image = cv2.resize(image, (100,100))
		image_list.append(image)
		label_list.append(label_dict[label])
		# print par_dir+label+image
		# break

# print len(image_list)
a=np.asarray(image_list)

#ruptured_skin
# 3795
# deformed_double
# 27
# others
# 23765
# 27587
# 3795+27+3800=7622
# 100*100*3
# a=a.reshape(27587,120000)

a=a.reshape(27587,30000)
del image_list
from tempfile import TemporaryFile
outfile = TemporaryFile()
np.save('images_level1', a)

# print set(label_list)
import keras
label_list=keras.utils.to_categorical(label_list, num_classes=3)
# print len(label_list),type(label_list)
np.save('labels_level1', label_list)
