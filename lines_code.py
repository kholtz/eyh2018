import math
import numpy as np

def get_slopes(lines):
    """
    Go through array of lines and return array of corresponding slopes
    """
    m = []
    for line in lines:
        denom = line[0,2] - line[0,0]
        #if denom == 0:
        #    denom = 1e-9
        m.append((line[0,3]-line[0,1]) / denom)
        if (m[-1]*0) != 0:
            print("slope is NaN: ", m, line)
    return np.array(m)

def remove_outliers(slopes):
    """
    Go through array of slopes and remove outliers, inf, and nan. 
    Return array of valid indices.
    """
    
    top = 2.5*np.percentile(slopes, 75) - 1.5*np.percentile(slopes, 25)
    bottom = 2.5*np.percentile(slopes, 25) - 1.5*np.percentile(slopes, 75)
    
    valid_inds = []
    for ind in range(len(slopes)):
        s = slopes[ind]
        if abs(s) > 1e9:
            continue
        if math.isnan(s):
            continue
        if s <= top and s >= bottom:
            valid_inds.append(ind)

    return valid_inds
    

def get_intercept(mean_slope, lines):
    """
    Go through array of lines and return mean intercept given the 
    slope of the line as mean_slope
    """
    b = []
    for line in lines:
        b.append(line[0,1] - mean_slope*line[0,0])
        b.append(line[0,3] - mean_slope*line[0,2])
    return np.mean(b)
        

def get_lane_lines(lines, imsize):
    """
    Go through array of lines and find two lane lines
    """
    
    # Get slope of each line and split into two categories (positive and negative slopes)
    m = get_slopes(lines)
    posinds = m > 0.1
    neginds = m < -0.1
    res = []
    height = imsize[0]
    width = imsize[1]
    
    # for each of the two groups, remove outliers based on slope then find the mean slope 
    # of the lines and the mean intercept corresponding to that slope
    # ignore if there are no lines in that category
    
    if sum(posinds) > 0:
        poslines = lines[posinds]
        posm = get_slopes(poslines)
        val_pos_ind = remove_outliers(posm)
        posm = posm[val_pos_ind]
        poslines = poslines[val_pos_ind]
        meanposm = np.mean(posm)
        posb = get_intercept(meanposm, poslines)
        if meanposm == 0:
            meanposm = 1e-10
        if not math.isnan(posb) and not math.isnan(meanposm):
            line1 = [[min(max(int((height-posb)/meanposm), 0), width), height, 
                      min(max(int(((0.6*height)-posb)/meanposm), 0), width), int(0.6*height)]]
            res.append(line1)

    if sum(neginds) > 0:
        neglines = lines[neginds]
        negm = get_slopes(neglines)
        val_neg_ind = remove_outliers(negm)
        negm = negm[val_neg_ind]
        neglines = neglines[val_neg_ind]
        meannegm = np.mean(negm)
        negb = get_intercept(meannegm, neglines)
        if meannegm == 0:
            meannegm = 1e-10
        if not math.isnan(negb) and not math.isnan(meannegm):
            line2 = [[min(max(int((height-negb)/meannegm), 0), width), height, 
                      min(max(int(((0.6*height)-negb)/meannegm), 0), width), int(0.6*height)]]
            res.append(line2)
   
    return res