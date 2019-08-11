import imageio
import numpy as np
from skimage import transform, exposure
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator

face_cascade = cv.CascadeClassifier(
	'./fer_env/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt.xml')

# create image data augmentation generator
datagen = ImageDataGenerator(
	brightness_range=[.2,.98],
	shear_range=20,
	rotation_range=30
	)

def procc_image(img_file,original_img_size,target_img_size,augment=False,aug_mul=9):
	# Read image from your local file system
	img_data = cv.imread(img_file)#.astype(float)
	# Convert color image to grayscale for Viola-Jones
	img_data = cv.cvtColor(img_data, cv.COLOR_BGR2GRAY)

	if img_data.shape != (original_img_size, original_img_size):
		raise Exception('Unexpected image shape: %s' % str(image_data.shape))

	detected_face = face_cascade.detectMultiScale(img_data)
	if detected_face == ():
		raise Exception('No face detected in: %s' % img_file)

	img_data = img_data[detected_face[0][1]:detected_face[0][1]+detected_face[0][3],
	detected_face[0][0]:detected_face[0][0]+detected_face[0][2]]
	images = list()
	if augment:
		# expand dimension to one sample
		samples = np.expand_dims(img_data, 0)
		samples = np.expand_dims(samples, 3)

		it = datagen.flow(samples[0:1,:,:,:], batch_size=1)

		# generate samples
		for i in range(aug_mul):
			# generate batch of images
			batch = it.next()
			# convert to float
			image = batch[0].astype('float')

			image = image[:,:,0]

			image = (image - np.mean(image)) / np.max(image)
			image = transform.resize(image, (target_img_size[0], target_img_size[1]), 
				mode='symmetric', preserve_range=True)

			images.append(image)
	else:
		img_data = (img_data - np.mean(img_data)) / np.max(img_data)
		img_data = transform.resize(img_data, (target_img_size[0], target_img_size[1]), 
			mode='symmetric', preserve_range=True)

		images.append(img_data)
	return images

	# p2, p98 = np.percentile(img_data, (12, 98))
	# img_data = exposure.rescale_intensity(img_data, in_range=(p2, p98))

	# img_data = img_data[250:762,256:640]
	# img_data = (img_data - np.mean(img_data)) / np.max(img_data)
	
	# return transform.resize(img_data, (target_img_size[0], target_img_size[1]), mode='symmetric', preserve_range=True)

if __name__ == '__main__':
	import matplotlib.pyplot as plt


	# Read image from your local file system
	original_image = cv.imread('./db/anger/001_an_002.jpg')

	# Convert color image to grayscale for Viola-Jones
	grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

	# Load the classifier and create a cascade object for face detection
	face_cascade = cv.CascadeClassifier('./fer_env/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt_tree.xml')

	detected_face = face_cascade.detectMultiScale(grayscale_image)

	for (column, row, width, height) in detected_face:
		cv.rectangle(
			original_image,
			(column, row),
			(column + width, row + height),
			(0, 255, 0),
			2
		)
	img_data = grayscale_image[detected_face[0][1]:detected_face[0][1]+detected_face[0][3],
	detected_face[0][0]:detected_face[0][0]+detected_face[0][2]]



	# expand dimension to one sample
	samples = np.expand_dims(img_data, 0)
	print(samples.shape)
	samples = np.expand_dims(samples, 3)
	print(samples.shape)

	# prepare iterator
	it = datagen.flow(samples[0:1,:,:,:], batch_size=1)

	# generate samples and plot
	for i in range(9):
		# define subplot
		plt.subplot(330 + 1 + i)
		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		image = batch[0].astype('float')
		# plot raw pixel data
		plt.imshow(image[:,:,0],cmap='gray', interpolation='none')
	# show the figure
	plt.show()
	print(image[:,:,0].shape)
	print(img_data.shape)
	cv.imshow('Image', img_data)
	cv.waitKey(0)
	cv.destroyAllWindows()
	# f,emotion=plt.subplots(1,5,figsize=(5,1))

	# img_data = rgb2gray((imageio.imread('./db/anger/001_an_002.jpg')).astype(float))

	# img_data = (img_data - np.mean(img_data)) / np.max(img_data)
	# emotion[0].imshow(img_data)

	# img_data2=exposure.equalize_hist(img_data)
	# emotion[1].imshow(img_data2)

	# p2, p98 = np.percentile(img_data, (2, 98))
	# img_data3 = exposure.rescale_intensity(img_data, in_range=(p2, p98))
	# emotion[2].imshow(img_data3)

	# img_data4 =	img_data[192:704,192:704]
	# emotion[3].imshow(img_data4)

	# image_file = procc_image('./db/anger/001_an_001.jpg',896,896)
	# emotion[4].imshow(image_file)

	# plt.show()

	im = procc_image('./db/anger/001_an_002.jpg',896,[128,128])
	print(len(im))
	print(im[0].shape)
	plt.imshow(im[0],cmap='gray', interpolation='none')
	plt.show()