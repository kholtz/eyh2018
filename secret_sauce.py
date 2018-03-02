import ipywidgets as widgets
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_color_widget(color_name):
	return widgets.IntRangeSlider(value=[0, 255], 
		min=0,
		max=255,
		step=1, 
		description=color_name + ':',
		disabled=False,
		continuous_update=True,
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
