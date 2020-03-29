import matplotlib.pyplot as plt
import numpy as np
import os
from six.moves import cPickle as pickle
from pre_proccess import procc_image
import zipfile
import json

import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input,Flatten
from keras.utils import to_categorical
from keras.backend.tensorflow_backend import set_session

from autoencoder import encoder
from fer_autoenc import fc
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)

json_file = open('./settings.json')
json_str = json_file.read()
settings = json.loads(json_str)

original_image_size =  896 # Original Pixel width and height
image_size =  settings['image_size']
local_path = './trajectories'

def load_emotion(folder, min_num_images, augment):
	'''Load the data for a single emotion label. '''
	image_files = sorted(os.listdir(os.path.join(local_path,folder)))
	dataset = np.ndarray(shape=(len(image_files*settings['aug_multy']), image_size[0], image_size[1]), dtype = np.float32)

	print(folder)
	num_images = 0
	for image in image_files:
		image_file = os.path.join(local_path, folder, image)
		try:
			images = procc_image(image_file,original_image_size,image_size, augment, aug_mul=settings['aug_multy'])
			for im in images:
				dataset[num_images,:,:] = im
				num_images = num_images + 1
		except (IOError, ValueError) as e:
			print('Could not read:', image_file,':',e,'- its ok, skipping.')
	dataset = dataset[0:num_images,:,:]
	if num_images<min_num_images:
		raise Exception('Fewer images than expected: %d<%d'%(num_images,min_num_images))

	print('Full dataset tensor:', dataset.shape)
	print('Mean:',np.mean(dataset))
	print('Standard deviation:', np.std(dataset))
	return dataset

def maybe_pickle(data_folders, min_num_images_per_class, augment, force = False):
	'''Load an emotion and save it into a pickle''' 
	dataset_names=[]
	for folder in data_folders:
		set_filename = folder + '.pickle'
		set_filename = os.path.join(local_path, set_filename)
		dataset_names.append(set_filename)
		if os.path.exists(set_filename) and not force:
			print('%s already present - Skipping pickling.' % set_filename)
		else:
			print('Pickling %s' % set_filename)
			dataset = load_emotion(folder, min_num_images_per_class, augment)
			try:
				with open(set_filename, 'wb') as f:
					pickle.dump(dataset,f,pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print('Unable to save data to' , set_filename, ':', e)
	return dataset_names

def make_arrays(nb_rows, img_size):
	if nb_rows:
		dataset = np.ndarray((nb_rows, img_size[0], img_size[1]), dtype=np.float32)
		labels = np.ndarray(nb_rows, dtype=np.int32)
	else:
		dataset,labels=None,None
	return dataset,labels

def merge_datasets(pickle_files, sizes, train_size, fold=1, valid_size=0):
	sizes.append(0)
	num_classes = len(pickle_files)
	vs = [int(x * valid_size) for x in sizes]
	valid_dataset, valid_labels = make_arrays(sum(vs), image_size)
	train_dataset, train_labels = make_arrays(sum(sizes)-sum(vs), image_size)

	start_v, start_t = 0, 0
	for label, pickle_file in enumerate(pickle_files):
		try:
			with open(pickle_file, 'rb') as f:
				emotion_set = pickle.load(f)
				vsize_per_class = int(valid_size*sizes[label])
				if valid_dataset is not None:
					valid_emotion = emotion_set[(fold-1)*vsize_per_class:fold*vsize_per_class:1, :, :]
					valid_dataset[start_v:start_v+len(valid_emotion), :, :] = valid_emotion
					valid_labels[start_v:start_v+len(valid_emotion)] = label
					start_v += len(valid_emotion)
				tsize_per_class = int(train_size*sizes[label])
				train_emotion = emotion_set[:(fold-1)*vsize_per_class,:,:]
				train_emotion = np.concatenate((train_emotion, emotion_set[fold*vsize_per_class:, :, :]))

				train_dataset[start_t:start_t+len(train_emotion), :, :] = train_emotion
				train_labels[start_t:start_t+len(train_emotion)] = label
				start_t += len(train_emotion)
		except Exception as e:
			print('Unable to process data from', pickle_file, ':', e)
			raise
	return valid_dataset, valid_labels, train_dataset, train_labels

def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

def fcfl(enco):
	return Flatten()(enco)

if __name__ == '__main__':
	test_folders = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
	test_datasets = maybe_pickle(test_folders, 0, False)
	sizes=list()

	test_sizes = list()
	for x in test_datasets:
		ax = pickle.load(open(x, 'rb'))
		test_sizes.append(len(ax))

	_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_sizes, 1)

	print('Testing:', test_dataset.shape, test_labels.shape)
	pickle_file = os.path.join(local_path, 'FER_traj.pickle')

	translate_labels = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

	print('Test set', test_dataset.shape, test_labels.shape)

	test_dataset = test_dataset.reshape(
		(-1, settings['image_size'][0], settings['image_size'][1], settings['num_channels'])).astype(np.float32)
	test_labels_oh = to_categorical(test_labels)
	print('Test set', test_dataset.shape, test_labels_oh.shape)

	input_img = Input(shape = (settings['image_size'][0], settings['image_size'][1], settings['num_channels']))

	encode = encoder(input_img)
	flat_level = (Model(input_img,fcfl(encode)))
	full_model = Model(input_img,fc(encode))
	full_model.load_weights('classification_complete.h5')

	for l1,l2 in zip(flat_level.layers[:],full_model.layers[:14]):
		l1.set_weights(l2.get_weights())

	flat_level.compile(
		loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

	predicted_classes = flat_level.predict(test_dataset)

	from sklearn.decomposition import PCA
	from sklearn.preprocessing import StandardScaler

	scaler = StandardScaler()
	scaler.fit(predicted_classes)
	X_sc_train = scaler.transform(predicted_classes)

	pca = PCA(n_components=3)
	X_pca = pca.fit_transform(X_sc_train)

	print(pca.explained_variance_ratio_)
	print(X_pca[0])
	from mpl_toolkits.mplot3d import Axes3D

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = [x[0] for x in X_pca[:36]]
	y = [x[1] for x in X_pca[:36]]
	z = [x[2] for x in X_pca[:36]]
	c = 'r'
	ax.plot(x,y,z,color=c,label='Anger')

	x = [x[0] for x in X_pca[36:63]]
	y = [x[1] for x in X_pca[36:63]]
	z = [x[2] for x in X_pca[36:63]]
	c = 'y'
	ax.plot(x,y,z,color=c,label='Disgust')

	x = [x[0] for x in X_pca[63:90]]
	y = [x[1] for x in X_pca[63:90]]
	z = [x[2] for x in X_pca[63:90]]
	c = 'g'
	ax.plot(x,y,z,color=c,label='Fear')

	x = [x[0] for x in X_pca[90:122]]
	y = [x[1] for x in X_pca[90:122]]
	z = [x[2] for x in X_pca[90:122]]
	c = 'b'
	ax.plot(x,y,z,color=c,label='Happiness')

	x = [x[0] for x in X_pca[122:137]]
	y = [x[1] for x in X_pca[122:137]]
	z = [x[2] for x in X_pca[122:137]]
	c = 'k'
	ax.plot(x,y,z,color=c,label='Neutral')

	x = [x[0] for x in X_pca[137:179]]
	y = [x[1] for x in X_pca[137:179]]
	z = [x[2] for x in X_pca[137:179]]
	c = 'c'
	ax.plot(x,y,z,color=c,label='Sadness')

	x = [x[0] for x in X_pca[179:]]
	y = [x[1] for x in X_pca[179:]]
	z = [x[2] for x in X_pca[179:]]
	c = 'm'
	ax.plot(x,y,z,color=c,label='Surprise')

	plt.legend()
	plt.show()