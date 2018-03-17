import numpy as np
import matplotlib.image as mpimg
import cv2
from my_display_code import *
from lines_code import *

# Load Images
filename = 'solidWhiteRight.jpg'
white_lines = mpimg.imread(filename)
yellowfile = 'solidYellowCurve.jpg'
yellow_lines = mpimg.imread(yellowfile)




############## Color thresholding ##############
def apply_color_thresholds(image, red, blue, green):
	out_of_range = (image[:, :, 0] < red[0]) | (image[:, :, 0] > red[1]) \
	| (image[:, :, 1] < blue[0]) | (image[:, :, 1] > blue[1]) \
	| (image[:, :, 2] < green[0]) | (image[:, :, 2] > green[1])
	result = np.copy(image)
	result[out_of_range] = [0, 0, 0]
	return result

def show_color_threshold_results(red=(0, 255), blue=(0, 255), green=(0, 255)):
	show_images(white_lines, apply_color_thresholds(white_lines, red, blue, green), 
		yellow_lines, apply_color_thresholds(yellow_lines, red, blue, green))

############## Edge Detection ##############
def apply_edges(image, low_threshold, high_threshold):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	return cv2.Canny(gray, low_threshold, high_threshold)

def show_edge_detection_results(low_threshold=0, high_threshold=0):
	show_images(white_lines, apply_edges(white_lines, low_threshold, high_threshold), 
		yellow_lines, apply_edges(yellow_lines, low_threshold, high_threshold))

############## Cumulative color and edge detection ##############
def apply_masks(image, low_red, high_red, low_blue, high_blue, 
	low_green, high_green, low_edge, high_edge):
	lower_color = np.array([low_red, low_blue, low_green])
	higher_color = np.array([high_red, high_blue, high_green])

	colormask = cv2.inRange(image, lower_color, higher_color)
	colormask = cv2.dilate(colormask, (10, 10))
	edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), low_edge, high_edge)
	return cv2.bitwise_and(edges, colormask)

############## ROI ##############
def mask_roi(image, vertices):
    mask = np.zeros_like(image)   
    
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_region(image, vertices):
	cp = np.copy(image)
	numl = len(vertices)
	for i, j in zip(range(numl), range(1, numl + 1)):
		cv2.line(cp, tuple(vertices[i % numl]), tuple(vertices[j % numl]), (255,0,0), 5)
	return cp

def get_vertices(top_height, bottom_width, top_width):
	# Define region of interest (ROI) in images
	if top_height > 1.0 or bottom_width > 1.0 or top_width > 1.0:
		top_height /= 100.0
		bottom_width /= 100.0
		top_width /= 100.0
	height = white_lines.shape[0]
	width = white_lines.shape[1]
	vertices = np.array([
		((1.0 - bottom_width) * width / 2.0, height),
		((1.0 - top_width) * width / 2.0, (1.0 - top_height) * height), 
		((1.0 + top_width) * width / 2.0, (1.0 - top_height) * height),
		((1.0 + bottom_width) * width / 2.0, height)
		], dtype=np.int32)
	return vertices

def show_regions_of_interest(top_height=10, bottom_width=100, top_width=90):
	vertices = get_vertices(top_height, bottom_width, top_width)
	show_images(
		draw_region(white_lines, vertices),
		mask_roi(white_lines, vertices),
		draw_region(yellow_lines, vertices), 
		mask_roi(yellow_lines, vertices)
		)

############## Hough Lines ##############
def get_hough_lines(masked_image, min_line_length, max_line_gap):
	# Define the Hough transform parameters
    rho = 3 # distance resolution in pixels of the Hough grid
    theta = 4*np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)

    return cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]),
    	min_line_length, max_line_gap)

def draw_hough_lines(masked_image, min_line_length, max_line_gap):
	lines = get_hough_lines(masked_image, min_line_length, max_line_gap)
	masked_image_color = np.dstack((masked_image, masked_image, masked_image))
	line_image = np.copy(masked_image_color)*0
	for l in lines:
		cv2.line(line_image, (l[0, 0], l[0, 1]), (l[0, 2], l[0, 3]), (255, 0, 0), 5)
	return cv2.addWeighted(masked_image_color, 0.7, line_image, 1, 0)

def show_hough_lines(masked_image1, masked_image2, min_line_length=2, max_line_gap=0):
	show_images(
		masked_image1, 
		draw_hough_lines(masked_image1, min_line_length, max_line_gap),
		masked_image2,
		draw_hough_lines(masked_image2, min_line_length, max_line_gap)
	)

