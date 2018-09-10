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
	for image in images:#[:30]:
		file= par_dir+label+'/'+image
		image = cv2.imread(file)
		# print image.shape
		image = cv2.resize(image, (100,100))
		image_list.append(image)
		label_list.append(label_dict[label])
		# print par_dir+label+image
		# break

image_list = np.asarray(image_list).transpose(0,3,1,2)
label_list = np.asarray(label_list)[None,None,None].transpose(3,0,1,2)

print image_list.shape
print label_list.shape

from sklearn.model_selection import train_test_split
image_list, X_test, label_list, y_test = train_test_split(image_list, label_list, test_size=0, random_state=4)


import h5py

hdf_file = h5py.File('train.h5', 'w')
hdf_file.create_dataset('data', dtype=np.float, data=image_list[:-5518,:,:,:], chunks=True)
hdf_file.create_dataset('label', dtype=np.float, data=label_list[:-5518,:,:,:])
hdf_file = h5py.File('test.h5', 'w')
hdf_file.create_dataset('data', dtype=np.float, data=image_list[-5518:,:,:,:], chunks=True)
hdf_file.create_dataset('label', dtype=np.float, data=label_list[-5518:,:,:,:])












'''
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
'''