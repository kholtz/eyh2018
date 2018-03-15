import ipywidgets as widgets
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

class Params:
	def __init__(self):
		self.red = (0, 255)
		self.blue = (0, 255)
		self.green = (0, 255)
		self.low_threshold = 0
		self.high_threshold = 0

params = Params()

def get_color_widget(color_name):
	return widgets.IntRangeSlider(
		value=[0, 255], 
		min=0,
		max=255,
		step=1, 
		description=color_name + ':',
		disabled=False,
		continuous_update=False,
		orientation='horizontal',
		readout=True,
		readout_format='d')

red_widget = get_color_widget('Red')
blue_widget = get_color_widget('Blue')
green_widget = get_color_widget('Green')

low_widget = widgets.IntSlider(
	value=1,
	min=0,
	max=200,
	step=1,
	description='Low:',
	disabled=False,
	continuous_update=False,
	orientation='horizontal',
	readout=True,
	readout_format='d'
)

high_widget = widgets.IntSlider(
	value=1,
	min=0,
	max=400,
	step=1,
	description='High:',
	disabled=False,
	continuous_update=False,
	orientation='horizontal',
	readout=True,
	readout_format='d'
)

filename = 'solidWhiteRight.jpg'
img = mpimg.imread(filename)

yellowfile = 'solidYellowCurve.jpg'
yellowimg = mpimg.imread(yellowfile)

def apply_color_thresholds(image, red, blue, green):
	out_of_range = (image[:, :, 0] < red[0]) | (image[:, :, 0] > red[1]) \
	| (image[:, :, 1] < blue[0]) | (image[:, :, 1] > blue[1]) \
	| (image[:, :, 2] < green[0]) | (image[:, :, 2] > green[1])
	result = np.copy(image)
	result[out_of_range] = [0, 0, 0]
	return result

def show_color_threshold_results(red=(0, 255), blue=(0, 255), green=(0, 255)):
	params.red = red
	params.blue = blue
	params.green = green

	f, axarr = plt.subplots(2,2, figsize=(20, 12))
	axarr[0, 0].imshow(img)
	axarr[0, 1].imshow(apply_color_thresholds(img, red, blue, green))
	axarr[1, 0].imshow(yellowimg)
	axarr[1, 1].imshow(apply_color_thresholds(yellowimg, red, blue, green))
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

def apply_edges(image, low_threshold, high_threshold):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	return cv2.Canny(gray, low_threshold, high_threshold)

def show_edge_detection_results(low_threshold=0, high_threshold=0):
	params.low_threshold = low_threshold
	params.high_threshold = high_threshold
	f, axarr = plt.subplots(2, 2, figsize=(20,12))
	axarr[0, 0].imshow(img, cmap='gray')
	axarr[0, 1].imshow(apply_edges(img, low_threshold, high_threshold), cmap='gray')
	axarr[1, 0].imshow(yellowimg, cmap='gray')
	axarr[1, 1].imshow(apply_edges(yellowimg, low_threshold, high_threshold), cmap='gray')
	plt.show()

def apply_masks():
	lower_color = np.array([params.red[0], params.blue[0], params.green[0]])
	higher_color = np.array([params.red[1], params.blue[1], params.green[1]])
	colormask_white = cv2.inRange(img, lower_color, higher_color)
	colormask_white = cv2.dilate(colormask_white, (10, 10))
	white_edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 
		params.low_threshold, params.high_threshold)
	white_masked_edges = cv2.bitwise_and(white_edges, colormask_white)

	colormask_yellow = cv2.inRange(yellowimg, lower_color, higher_color)
	colormask_yellow = cv2.dilate(colormask_yellow, (10, 10))
	yellow_edges = cv2.Canny(cv2.cvtColor(yellowimg, cv2.COLOR_RGB2GRAY), 
		params.low_threshold, params.high_threshold)
	yellow_masked_edges = cv2.bitwise_and(yellow_edges, colormask_yellow)

	f, axarr = plt.subplots(2,2, figsize=(20, 12))
	axarr[0, 0].imshow(img)
	axarr[0, 1].imshow(white_masked_edges, cmap='gray')
	axarr[1, 0].imshow(yellowimg)
	axarr[1, 1].imshow(yellow_masked_edges, cmap='gray')
	plt.show()

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)   
    
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_region(img, vertices):
	cp = np.copy(img)
	numl = len(vertices)
	for i, j in zip(range(numl), range(1, numl + 1)):
		cv2.line(cp, tuple(vertices[i % numl]), tuple(vertices[j % numl]), (255,0,0), 5)
	return cp

def show_regions_of_interest():
	height = img.shape[0]
	width = img.shape[1]
	vertices = np.array([(0.05,height),(0.45*width, 0.55*height), (0.55*width, 0.55*height),
		(0.95*width, height)], dtype=np.int32)
	f, axarr = plt.subplots(2, 2, figsize=(20, 12))
	axarr[0, 0].imshow(draw_region(img, vertices))
	axarr[0, 1].imshow(region_of_interest(img, vertices))
	axarr[1, 0].imshow(draw_region(yellowimg, vertices))
	axarr[1, 1].imshow(region_of_interest(yellowimg, vertices))
	plt.show()
