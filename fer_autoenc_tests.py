import os
import keras
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input
from keras.utils import to_categorical
from six.moves import cPickle as pickle
from autoencoder import encoder
from fer_autoenc import fc
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

image_size = 128
num_labels = 7
num_channels = 1 # grayscale

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
	return dataset, to_categorical(labels)

if __name__ == '__main__':
	translate_labels = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
	pickle_file = './db/FER.pickle'

	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		test_dataset = save['test_dataset']
		test_labels = save['test_labels']
		del save 
		print('Test set', test_dataset.shape, test_labels.shape)

	test_dataset, test_labels_oh = reformat(test_dataset, test_labels)
	print('Test set', test_dataset.shape, test_labels_oh.shape)

	input_img = Input(shape = (image_size, image_size, num_channels))

	encode = encoder(input_img)
	full_model = Model(input_img,fc(encode))
	full_model.load_weights('classification_complete.h5')
	full_model.compile(
		loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

	test_eval = full_model.evaluate(test_dataset, test_labels_oh, verbose=0)
	print('Test loss:', test_eval[0])
	print('Test accuracy:', test_eval[1])

	predicted_classes = full_model.predict(test_dataset)

	predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
	correct = np.where(predicted_classes==test_labels)[0]
	print("Found %d correct labels" % len(correct))
	for i, correct in enumerate(correct[:9]):
		plt.subplot(3,3,i+1)
		plt.imshow(test_dataset[correct].reshape(128,128), cmap='gray', interpolation='none')
		plt.title("Predicted {}, Class {}".format(translate_labels[predicted_classes[correct]], translate_labels[test_labels[correct]]))
	plt.show()

	incorrect = np.where(predicted_classes!=test_labels)[0]
	print("Found %d incorrect labels" % len(incorrect))
	for i, incorrect in enumerate(incorrect[:9]):
		plt.subplot(3,3,i+1)
		plt.imshow(test_dataset[incorrect].reshape(128,128), cmap='gray', interpolation='none')
		plt.title("Predicted {}, Class {}".format(translate_labels[predicted_classes[incorrect]], translate_labels[test_labels[incorrect]]))
	plt.show()

	from sklearn.metrics import classification_report
	target_names = ["Class {}".format(translate_labels[i]) for i in range(num_labels)]
	print(classification_report(test_labels, predicted_classes, target_names=target_names))