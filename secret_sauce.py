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

# Load Images
filename = 'solidWhiteRight.jpg'
img = mpimg.imread(filename)
yellowfile = 'solidYellowCurve.jpg'
yellowimg = mpimg.imread(yellowfile)

# Define region of interest (ROI) in images
height = img.shape[0]
width = img.shape[1]
vertices = np.array([(0.05,height),(0.45*width, 0.55*height), (0.55*width, 0.55*height),
	(0.95*width, height)], dtype=np.int32)

# Create widgets
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

# Color thresholding
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

# Edge Detection
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

# Cumulative color and edge detection
def apply_masks(image):
	lower_color = np.array([params.red[0], params.blue[0], params.green[0]])
	higher_color = np.array([params.red[1], params.blue[1], params.green[1]])

	colormask = cv2.inRange(image, lower_color, higher_color)
	colormask = cv2.dilate(colormask, (10, 10))
	edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
		params.low_threshold, params.high_threshold)
	return cv2.bitwise_and(edges, colormask)

def show_masking_results():
	f, axarr = plt.subplots(2,2, figsize=(20, 12))
	axarr[0, 0].imshow(img)
	axarr[0, 1].imshow(apply_masks(img), cmap='gray')
	axarr[1, 0].imshow(yellowimg)
	axarr[1, 1].imshow(apply_masks(yellowimg), cmap='gray')
	plt.show()

# ROI
# TODO: give them sliders for the ROI
def region_of_interest(image):
    mask = np.zeros_like(image)   
    
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_region(img):
	cp = np.copy(img)
	numl = len(vertices)
	for i, j in zip(range(numl), range(1, numl + 1)):
		cv2.line(cp, tuple(vertices[i % numl]), tuple(vertices[j % numl]), (255,0,0), 5)
	return cp

def show_regions_of_interest():
	f, axarr = plt.subplots(2, 2, figsize=(20, 12))
	axarr[0, 0].imshow(draw_region(img))
	axarr[0, 1].imshow(region_of_interest(img))
	axarr[1, 0].imshow(draw_region(yellowimg))
	axarr[1, 1].imshow(region_of_interest(yellowimg))
	plt.show()

# Hough Lines
def get_hough_line_image(image):
	# Define the Hough transform parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = 6*np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 12 #minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    masked_edges = region_of_interest(apply_masks(image))
    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), 
    	min_line_length, max_line_gap)

    for l in lines:
    	cv2.line(line_image, (l[0, 0], l[0, 1]), (l[0, 2], l[0, 3]), (255, 0, 0), 10)
    masked_edges_color = np.dstack((masked_edges, masked_edges, masked_edges))
    return cv2.addWeighted(masked_edges_color, 0.7, line_image, 1, 0)

def show_hough_lines():
	f, axarr = plt.subplots(1, 2, figsize=(20, 12))
	axarr[0].imshow(get_hough_line_image(img))
	axarr[1].imshow(get_hough_line_image(yellowimg))
	plt.show()

