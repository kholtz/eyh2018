import matplotlib.pyplot as plt

# Some functions to reduce duplicated code - displaying two or four images and
# auto-changing the color map to gray for single-channel images

def get_cmap(image):
	if len(image.shape) == 2:
		return 'gray'
	return 'jet'

def show_two_images(img1, img2, gray=False):
	# Displays two images side-by-side
	f, axarr = plt.subplots(1, 2, figsize=(20, 12))
	axarr[0].imshow(img1, cmap=get_cmap(img1))
	axarr[1].imshow(img2, cmap=get_cmap(img2))
	plt.show()

def show_images(img1, img2, img3=None, img4=None):
	# Displays images left to right, top to bottom, i.e.
	#	img1	img2
	#	img3	img4
	# In the case of only 2 images, will only have one row
	if img3 is None and img4 is None:
		show_two_images(img1, img2)
		return
	f, axarr = plt.subplots(2, 2, figsize=(20, 12))
	axarr[0, 0].imshow(img1, cmap=get_cmap(img1))
	axarr[0, 1].imshow(img2, cmap=get_cmap(img2))
	if img3 is not None:
		axarr[1, 0].imshow(img3, cmap=get_cmap(img3))
	if img4 is not None:
		axarr[1, 1].imshow(img4, cmap=get_cmap(img4))
	plt.show()

