import matplotlib.pyplot as plt
import numpy as np
import os
from six.moves import cPickle as pickle
from pre_proccess import procc_image
import zipfile
import json

json_file = open('./settings.json')
json_str = json_file.read()
settings = json.loads(json_str)

original_image_size =  896 # Original Pixel width and height
image_size =  settings['image_size']
local_path = './db'

def maybe_extract(filename, force=False):
	root = os.path.splitext(os.path.splitext(filename)[0])[0]
	if os.path.isdir(root) and not force:
	# You may override by setting force=True.
		print('%s already present - Skipping extraction of %s.' % (root, filename))
	else:
		print('Extracting data for %s. This may take a while. Please wait.' % root)
		with zipfile.ZipFile(os.path.join(filename), 'r') as zip_ref:
			zip_ref.extractall('.')

def load_emotion(folder, min_num_images, augment):
	'''Load the data for a single emotion label. '''
	image_files = os.listdir(os.path.join(local_path,folder))
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

def merge_datasets(pickle_files, sizes, train_size, valid_size=0):
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
				# let's shuffle the expretions to have random validation and training set
				np.random.shuffle(emotion_set)
				vsize_per_class = int(valid_size*sizes[label])
				if valid_dataset is not None:
					valid_emotion = emotion_set[:vsize_per_class, :, :]
					valid_dataset[start_v:start_v+len(valid_emotion), :, :] = valid_emotion
					valid_labels[start_v:start_v+len(valid_emotion)] = label
					start_v += len(valid_emotion)
				tsize_per_class = int(train_size*sizes[label])
				train_emotion = emotion_set[vsize_per_class:, :, :]
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

if __name__ == '__main__':
	maybe_extract('db.zip')
	train_folders = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
	test_folders = ['test/anger', 'test/disgust', 'test/fear', 'test/happiness', 'test/neutral', 'test/sadness', 'test/surprise']
	train_datasets = maybe_pickle(train_folders, 10, True)
	test_datasets = maybe_pickle(test_folders, 0, False)
	sizes=list()

	f,emotion=plt.subplots(1,7,figsize=(18,3))
	for i,x in enumerate(train_datasets):
		print(x)
		ax = pickle.load(open(x, 'rb'))
		sizes.append(len(ax))
		emotion[i].imshow(ax[0],cmap='gray', interpolation='none')
		emotion[i].set_title(x[5:-7])
	f.savefig('./img/Train dataset examples.jpg')
	# plt.show()

	test_sizes = list()
	for x in test_datasets:
		ax = pickle.load(open(x, 'rb'))
		test_sizes.append(len(ax))

	train_size = 0.8

	valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
		train_datasets, sizes, train_size, 1-train_size)
	_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_sizes, 1)

	print('Training:', train_dataset.shape, train_labels.shape)
	print('Validation:', valid_dataset.shape, valid_labels.shape)
	print('Testing:', test_dataset.shape, test_labels.shape)

	train_dataset, train_labels = randomize(train_dataset, train_labels)
	test_dataset, test_labels = randomize(test_dataset, test_labels)
	valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

	f,emotion=plt.subplots(2,5,figsize=(12,5))
	for x in range(5):
		emotion[0][x].imshow(train_dataset[x],cmap='gray', interpolation='none')
		emotion[0][x].set_title(train_folders[train_labels[x]])
	for x in range(5,10):
		emotion[1][x-5].imshow(train_dataset[x],cmap='gray', interpolation='none')
		emotion[1][x-5].set_title(train_folders[train_labels[x]])
	f.savefig('./img/Train dataset examples proccessed.jpg')
	# plt.show()

	pickle_file = os.path.join(local_path, 'FER.pickle')

	try:
		f = open(pickle_file, 'wb')
		save = {
			'train_dataset': train_dataset,
			'train_labels': train_labels,
			'valid_dataset': valid_dataset,
			'valid_labels': valid_labels,
			'test_dataset': test_dataset,
			'test_labels': test_labels,
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
		f.close()
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise