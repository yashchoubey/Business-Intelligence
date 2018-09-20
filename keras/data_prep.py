image_list=[]
label_list=[]
import os,cv2,numpy as np
par_dir='/home/yash/GAN/English (copy)/Fnt/'
train_labels = os.listdir(par_dir)

for label in train_labels:
	# print par_dir+label
	images=os.listdir(par_dir+label)
	# print len(images)
	for image in images:#[:30]:
		file= par_dir+label+'/'+image
		image = cv2.imread(file,0)

		image = cv2.resize(image, (40,40))
		label_list.append(image/255.0)

		#blur
		image=cv2.blur(image, (5,5))
		image[image>128]=80
		image=cv2.blur(image, (5,5))
		image_list.append(image/255.0)
		
print len(image_list)	
print len(label_list)		

print image_list[0].shape
print label_list[0].shape
#62992

image_list = np.asarray(image_list)[:,:,:,None].transpose(0,3,1,2)
label_list = np.asarray(label_list)[:,:,:,None].transpose(0,3,1,2)
# label_list = np.asarray(label_list)[None,None,None].transpose(3,0,1,2)

print image_list.shape
print label_list.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state=4)


import h5py

hdf_file = h5py.File('train.h5', 'w')
hdf_file.create_dataset('data', dtype=np.float, data=X_train[:,:,:,:], chunks=True)
hdf_file.create_dataset('label', dtype=np.float, data=y_train[:,:,:,:])

hdf_file1 = h5py.File('test.h5', 'w')
hdf_file1.create_dataset('data', dtype=np.float, data=X_test[:,:,:,:], chunks=True)
hdf_file1.create_dataset('label', dtype=np.float, data=y_test[:,:,:,:])
