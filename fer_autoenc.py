import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import Input,Dense,Flatten,Dropout,Conv2D,MaxPooling2D, AveragePooling2D, SpatialDropout2D,AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model
from keras import regularizers
from six.moves import cPickle as pickle
from autoencoder import encoder,decoder
import tensorflow as tf
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

def fc(enco):
	conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(enco)
	conv1 = BatchNormalization()(conv1)

	# drop = SpatialDropout2D(rate = 0.4)(conv1)

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
	conv2 = BatchNormalization()(conv2)

	# conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
	# conv3 = BatchNormalization()(conv3)

	# conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv3)
	# conv4 = BatchNormalization()(conv4)

	# conv5 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv4)
	# conv5 = BatchNormalization()(conv5)

	# # # drop0 = SpatialDropout2D(rate = 0.5)(conv5)

	# conv6 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv5)
	# conv6 = BatchNormalization()(conv6)

	# # # drop = SpatialDropout2D(rate = 0.5)(conv6)

	# conv7 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv6)
	# conv7 = BatchNormalization()(conv7)

	# pool = AveragePooling2D(pool_size=(2, 2))(conv7)

	flat = Flatten()(conv2)
	drop0 = Dropout(rate=0.7)(flat)

	den1 = Dense(16, activation='relu')(drop0)
	drop1 = Dropout(rate=0.5)(den1)

	den2 = Dense(7, activation='relu')(drop1)
	drop2 = Dropout(rate=0.25)(den2)

	den3 = Dense(7, activation='relu')(drop2)
	# drop4 = Dropout(rate=0.5)(den3)

	out = Dense(7, activation='softmax')(den3)
	return out

if __name__ == '__main__':
	json_file = open('./settings.json')
	json_str = json_file.read()
	settings = json.loads(json_str)

	pickle_file = './db/FER.pickle'

	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		train_dataset = save['train_dataset']
		train_labels = save['train_labels']
		valid_dataset = save['valid_dataset']
		valid_labels = save['valid_labels']
		del save 
		print('Training set', train_dataset.shape, train_labels.shape)
		print('Validation set', valid_dataset.shape, valid_labels.shape)

	train_dataset = train_dataset.reshape(
		(-1, settings['image_size'], settings['image_size'], settings['num_channels'])).astype(np.float32)
	train_labels = to_categorical(train_labels)
	valid_dataset = valid_dataset.reshape(
		(-1, settings['image_size'], settings['image_size'], settings['num_channels'])).astype(np.float32)
	valid_labels = to_categorical(valid_labels)

	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)

	input_img = Input(shape = (settings['image_size'], settings['image_size'], settings['num_channels']))

	encode = encoder(input_img)
	full_model = Model(input_img,fc(encode))

	autoencoder = Model(input_img, decoder(encoder(input_img)))
	autoencoder.load_weights('autoencoder.h5')

	for l1,l2 in zip(full_model.layers[:16],autoencoder.layers[:16]):
		l1.set_weights(l2.get_weights())

	for layer in full_model.layers[:16]:
		print(layer)
		layer.trainable = False

	full_model.compile(
		loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=1e-4),metrics=['accuracy'])
	full_model.summary()
	plot_model(full_model, to_file='model.eps')


	classify_train = full_model.fit(
		train_dataset, train_labels, batch_size=256,epochs=125,verbose=1,validation_data=(valid_dataset, valid_labels))

	full_model.save_weights('autoencoder_classification.h5')

	for layer in full_model.layers[:16]:
		layer.trainable = True

	full_model.compile(
		loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=75e-6, decay=0.0001),metrics=['accuracy'])

	classify_train = full_model.fit(
		train_dataset, train_labels, batch_size=64,epochs=500,verbose=1,validation_data=(valid_dataset, valid_labels))

	full_model.save_weights('classification_complete.h5')

	accuracy = classify_train.history['acc']
	val_accuracy = classify_train.history['val_acc']
	loss = classify_train.history['loss']
	val_loss = classify_train.history['val_loss']
	epochs = range(len(accuracy))
	plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
	plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show() 