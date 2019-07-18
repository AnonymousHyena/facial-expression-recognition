import imageio
import numpy as np
from skimage import transform, exposure


def rgb2gray(rgb):
	return np.dot(rgb[...,:3],[0.2989, 0.587, 0.144])

def procc_image(img_file,original_img_size,target_img_size):
	img_data = rgb2gray((imageio.imread(img_file)).astype(float))
	if img_data.shape != (original_img_size, original_img_size):
		raise Exception('Unexpected image shape: %s' % str(image_data.shape))

	p2, p98 = np.percentile(img_data, (2, 98))
	img_data = exposure.rescale_intensity(img_data, in_range=(p2, p98))

	img_data = img_data[250:762,192:704]
	img_data = (img_data - np.mean(img_data)) / np.max(img_data)
	
	return transform.resize(img_data, (target_img_size, target_img_size), mode='symmetric', preserve_range=True)

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	f,emotion=plt.subplots(1,5,figsize=(5,1))

	img_data = rgb2gray((imageio.imread('./db/anger/001_an_001.jpg')).astype(float))

	img_data = (img_data - np.mean(img_data)) / np.max(img_data)
	emotion[0].imshow(img_data)

	img_data2=exposure.equalize_hist(img_data)
	emotion[1].imshow(img_data2)

	p2, p98 = np.percentile(img_data, (2, 98))
	img_data3 = exposure.rescale_intensity(img_data, in_range=(p2, p98))
	emotion[2].imshow(img_data3)

	img_data4 =	img_data[192:704,192:704]
	emotion[3].imshow(img_data4)

	image_file = procc_image('./db/anger/001_an_001.jpg',896,896)
	emotion[4].imshow(image_file)

	plt.show()

