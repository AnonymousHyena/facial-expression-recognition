import numpy as np
from six.moves import cPickle as pickle
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Reshape,Conv2D,MaxPooling2D,UpSampling2D,AveragePooling2D,Conv2DTranspose,SpatialDropout2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.backend.tensorflow_backend import set_session

import tensorflow as tf
import json

def encoder(input_img):
	#encoder
	#input = 128 x 128 x 1 (wide and thin)
	conv1 = Conv2D(8, (5, 5), activation='relu', padding='same')(input_img)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(16, (5, 5), activation='relu', padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = MaxPooling2D(pool_size=(2,2))(conv3)

	coded = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
	coded = BatchNormalization()(coded)
	coded = AveragePooling2D(pool_size=(2,2))(coded)

	#8 x 8 x 64 (small and thick)
	return coded

def decoder(coded):    
	#decoder
	conv7 = Conv2DTranspose(32, (3, 3), strides=(2,2),activation='relu', padding='same')(coded)
	conv7 = BatchNormalization()(conv7)

	conv8 = Conv2DTranspose(16, (3, 3), strides=(2,2),activation='relu', padding='same')(conv7)
	conv8 = BatchNormalization()(conv8)

	conv9 = Conv2DTranspose(8, (5, 5), strides=(2,2),activation='relu', padding='same')(conv8)
	conv9 = BatchNormalization()(conv9)

	decoded = Conv2DTranspose(1, (5, 5), strides=(2,2), activation='sigmoid', padding='same')(conv9)	
	return decoded

if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
	# config.log_device_placement = True  # to log device placement (on which device the operation ran)
	sess = tf.Session(config=config)

	json_file = open('./settings.json')
	json_str = json_file.read()
	settings = json.loads(json_str)

	pickle_file = './db/FER.pickle'

	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		train_dataset = save['train_dataset']
		valid_dataset = save['valid_dataset']
		test_dataset = save['test_dataset']
		del save 
		print('Training set', train_dataset.shape)
		print('Validation set', valid_dataset.shape)

	test_dataset = test_dataset.reshape(
		(-1, settings['image_size'][0], settings['image_size'][1], settings['num_channels'])).astype(np.float32)
	train_dataset = train_dataset.reshape(
		(-1, settings['image_size'][0], settings['image_size'][1], settings['num_channels'])).astype(np.float32)
	valid_dataset = valid_dataset.reshape(
		(-1, settings['image_size'][0], settings['image_size'][1], settings['num_channels'])).astype(np.float32)
	print('Training set', train_dataset.shape)
	print('Validation set', valid_dataset.shape)

	input_img = Input(shape = (settings['image_size'][0], settings['image_size'][1], settings['num_channels']))

	autoencoder = Model(input_img, decoder(encoder(input_img)))
	autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop(lr=1e-3,decay=1e-5))
	autoencoder.summary()

	autoencoder_train = autoencoder.fit(
		train_dataset, train_dataset, batch_size=256,
		epochs=100,
		verbose=1,validation_data=(valid_dataset, valid_dataset))

	autoencoder.save_weights('autoencoder.h5')

	loss = autoencoder_train.history['loss']
	val_loss = autoencoder_train.history['val_loss']
	epochs = range(len(loss))

	fig1 = plt.figure(dpi=200, figsize=(8,4.5))
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	fig1.savefig('./img/Auto training and validation loss.jpg')
	# plt.show()

	results=autoencoder.predict(test_dataset[:5])

	#Comparing original images with reconstructions
	f,a=plt.subplots(2,5,figsize=(10,4))
	for i in range(5):
		a[0][i].imshow(np.reshape(test_dataset[i],(settings['image_size'][0],settings['image_size'][1])),cmap='gray', interpolation='none')
		a[1][i].imshow(np.reshape(results[i],(settings['image_size'][0],settings['image_size'][1])),cmap='gray', interpolation='none')
	f.savefig('./img/Auto comparisons.jpg')
	# plt.show()