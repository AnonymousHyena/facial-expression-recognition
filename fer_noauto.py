import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import Input,Dense,Flatten,Dropout,Conv2D,MaxPooling2D, AveragePooling2D, SpatialDropout2D,Concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model
from keras import regularizers
from keras.backend.tensorflow_backend import set_session

from six.moves import cPickle as pickle
from autoencoder import encoder,decoder
import tensorflow as tf
import json

def fc(enco):
	flat = Flatten()(enco)
	drop0 = Dropout(rate=0.75)(flat)

	den1 = Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.008))(drop0)
	drop1 = Dropout(rate=0.65)(den1)

	den1 = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.008))(den1)
	drop1 = Dropout(rate=0.65)(den1)

	den2 = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.008))(drop1)
	den2 = Dropout(rate=0.65)(den2)

	den3 = Dense(7, activation='relu', kernel_regularizer=regularizers.l2(0.008))(den2)

	out = Dense(7, activation='softmax')(den3)
	return out

if __name__ == '__main__':
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
		(-1, settings['image_size'][0], settings['image_size'][1], settings['num_channels'])).astype(np.float32)
	train_labels = to_categorical(train_labels)
	valid_dataset = valid_dataset.reshape(
		(-1, settings['image_size'][0], settings['image_size'][1], settings['num_channels'])).astype(np.float32)
	valid_labels_oh = to_categorical(valid_labels)

	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels_oh.shape)

	input_img = Input(shape = (settings['image_size'][0], settings['image_size'][1], settings['num_channels']))

	encode = encoder(input_img)
	full_model = Model(input_img,fc(encode))

	full_model.compile(
		loss=keras.losses.categorical_crossentropy, 
		optimizer=keras.optimizers.Adam(lr=6e-5, decay=0.85e-7),
		metrics=['accuracy'])

	classify_train = full_model.fit(
		train_dataset, train_labels, batch_size=256, 
		epochs=600,
		verbose=1,validation_data=(valid_dataset, valid_labels_oh))

	full_model.save_weights('classification_complete_noauto.h5')

	accuracy = classify_train.history['acc']
	val_accuracy = classify_train.history['val_acc']
	loss = classify_train.history['loss']
	val_loss = classify_train.history['val_loss']
	epochs = range(len(accuracy))

	fig1 = plt.figure(dpi=200, figsize=(8,4.5))
	plt.plot(epochs, accuracy, 'b', label='Training accuracy', color='green')
	plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	fig1.savefig('./img/Training and validation accuracy no auto.jpg')

	fig2 = plt.figure(dpi=200, figsize=(8,4.5))
	plt.plot(epochs, loss, 'b', label='Training loss', color='green')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	fig2.savefig('./img/Training and validation loss no autop.jpg')