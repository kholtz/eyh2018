import ipywidgets as widgets
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


def get_color_widget(color_name):
	return widgets.IntRangeSlider(value=[0, 255], 
		min=0,
		max=255,
		step=1, 
		description=color_name + ':',
		disabled=False,
		continuous_update=False,
		orientation='horizontal',
		readout=True,
		readout_format='d'
		)

red_widget = get_color_widget('Red')
blue_widget = get_color_widget('Blue')
green_widget = get_color_widget('Green')

filename = 'solidWhiteRight.jpg'
img = mpimg.imread(filename)

def apply_color_thresholds(red=(0, 255), blue=(0, 255), green=(0, 255)):
	color_select = np.copy(img)
	thresholds = (img[:,:,0] < red[0]) | (img[:,:,0] > red[1]) \
	| (img[:,:,1] < blue[0]) | (img[:,:,1] > blue[1]) \
	| (img[:,:,2] < green[0]) | (img[:,:,2] > green[1])
	color_select[thresholds] = [0, 0, 0]
	f, axarr = plt.subplots(1,2, figsize=(20, 10))
	axarr[0].imshow(img)
	axarr[1].imshow(color_select)
	plt.show()

def show_rgb():
	f, axarr = plt.subplots(1, 3, figsize=(20,10))
	red_img = np.copy(img)
	red_img[:,:,1] = 0
	red_img[:,:,2] = 0
	axarr[0].imshow(red_img)
	
	blue_img = np.copy(img)
	blue_img[:,:,0] = 0
	blue_img[:,:,2] = 0
	axarr[1].imshow(blue_img)

	green_img = np.copy(img)
	green_img[:,:,0] = 0
	green_img[:,:,1] = 0
	axarr[2].imshow(green_img)
	plt.show()
	
	f, axarr = plt.subplots(1, 3, figsize=(20,10))
	axarr[0].imshow(red_img[:,:,0], cmap='gray')
	axarr[1].imshow(blue_img[:,:,1], cmap='gray')
	axarr[2].imshow(green_img[:,:,2], cmap='gray')
	plt.show()

def apply_edges(kernel_size=3, low_threshold=0, high_threshold=100):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
	canny = cv2.Canny(blurred, low_threshold, high_threshold)
	f, axarr = plt.subplots(1, 2, figsize=(20,10))
	axarr[0].imshow(gray, cmap='gray')
	axarr[1].imshow(canny, cmap='gray')
	plt.show()

