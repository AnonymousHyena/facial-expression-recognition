import imageio
import numpy as np
from skimage import transform, exposure
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator

face_cascade = cv.CascadeClassifier(
	'./fer_env/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt.xml')

# create image data augmentation generator
datagen = ImageDataGenerator(
	rescale = 1./255,
	brightness_range=[.2,.98],
	shear_range=20,
	rotation_range=30,
	zoom_range=.2,
	)

def procc_image(img_file,original_img_size,target_img_size,augment=False,aug_mul=9):
	# Read image from your local file system
	img_data = cv.imread(img_file)
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
	img_data = (img_data - np.mean(img_data)) / 255.0
	img_data = transform.resize(img_data, (target_img_size[0], target_img_size[1]), 
		mode='symmetric', preserve_range=True)

	images.append(img_data)
	if augment:
		# expand dimension to one sample
		samples = np.expand_dims(img_data, 0)
		samples = np.expand_dims(samples, 3)

		it = datagen.flow(samples[0:1,:,:,:], batch_size=1)

		# generate samples
		for i in range(aug_mul-1):
			# generate batch of images
			batch = it.next()
			# convert to float
			image = batch[0].astype('float')

			image = image[:,:,0]

			image = (image - np.mean(image))
			image = transform.resize(image, (target_img_size[0], target_img_size[1]), 
				mode='symmetric', preserve_range=True)

			images.append(image)
	return images

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	# Read image from your local file system
	original_image = cv.imread('./db/test/anger/KA.AN1.39.tiff')

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

	cv.imshow('Image', original_image)
	cv.waitKey(0)
	cv.destroyAllWindows()

	im = procc_image('./db/test/anger/KA.AN1.39.tiff',896,[128,128],True)
	print(len(im))
	print(im[0].shape)
	f,a=plt.subplots(2,5,figsize=(10,4))
	for i in range(len(im)//2):
		a[0][i].imshow(im[i],cmap='gray', interpolation='none')
		a[1][i].imshow(im[-i],cmap='gray', interpolation='none')
	plt.show()