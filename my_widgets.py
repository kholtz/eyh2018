import ipywidgets as widgets

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

min_line_widget = widgets.IntSlider(
	value=2,
	min=2,
	max=40,
	step=2,
	description='Minimum Pixels for a Line:',
	continuous_update=False,
	orientation='horizontal',
	readout=True,
	readout_format='d'
)
max_line_widget = widgets.IntSlider(
	value=0,
	min=0,
	max=50,
	step=2,
	description='Maximum Gap in a Line:',
	continuous_update=False,
	orientation='horizontal',
	readout=True,
	readout_format='d'
)