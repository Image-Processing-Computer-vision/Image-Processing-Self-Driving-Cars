#!/usr/bin/env python
# coding: utf-8

# ## Names and IDs:
# 
# - Yousef Adel Ismail Mohammed Shalaby (1802267)
# - John Magdy William Eskander Awadallah (1808270)
# - Farah Ihab Samir Abdo Mikhael  (1802653)

# # Phase 1 - Lane Line Detection

# ### Approach Steps:
# - Read photo
# - Apply Pre-processing Techniques
# - Try getting HLS channels first to get rid of lightness noise
# - Extract edges with Sobel from different informative views and stacking them together
# - Try to apply perspective transform to convert to Bird's eye view
# - Need to impelement a function to detect lane line using sliding window
# - Why not improve the prvious function instead of performing blind search with each new frame, it should use info extracted from the previous frame
# - Calculate radius of curvature and position from the center of the lane
# - Got all final outputs and implemented stacked image to use in debugging mode
# - Experiment with video
# 

# # Phase 2 - Vehicle Tracking

# ### Approach Steps:
# - Define a function that draw boxes around vehicles, trying manual inputs to draw boxes
# - Define a function to compute color histogram features 
# - Define a function to convert the image to any color space and get the spatial bin as 1D feature vector to feed to classifier
# - Define a function that extract info of the dataset that will be used to train the classifier
# - Define a function that return HOG features and it can visualize it
# - Import StandardScaler to scale features to make it ready for training the classifier
# - Implement a function to get features from images list to train the classifier model
# - Train a classifier model
# - Sliding window for the classifier to go through
# - Define a function to extract features from a single image window (will be used in the pipeline of video)
# - Define a function to search windows (output of slide_windows())
# - Define a complete function to extract features from a list of images it takes as input either hog features or histogram features or spatial features
# - Define some functions like using heat maps to improve the tracking of cars in pictures
# - Defined a class Parameters that will take all parameters and store it to use in different functions
# - Implement a collective functions that will make processing faster
# - Integrate Vehicle tracking part with lane detection part



# ----------------------------------------------------------

# ### Import Libraries:

# In[2]:


import numpy as np
import cv2
import glob
import time
import pickle
import os
import sys
import matplotlib.pyplot as plt
from skimage.feature import hog
from scipy.ndimage.measurements import label

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

#get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# --------------------------------

# ### Lane Line Detection - Implemented Functions:

# In[3]:


def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


# In[4]:


def gaussian(img, kernel_size):
    """Applies Guassian filter on the image"""
    blurry = cv2.GaussianBlur(img, (kernel_size,kernel_size), 0)
    return blurry


# In[5]:


def abs_sobel_thresh(img, orients='x', sobel_kernel=3, thresh=(0,255)):
    """
    This function takes the image and applies sobel filter to it to
    take gradients of the image
    """
    gray = grayscale(img)
    
    if(orients == 'x'):
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if(orients == 'y'):
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        
    # rescale to 8 bits
    scaled_sobel = np.uint8((255*abs_sobel)/np.max(abs_sobel))
    
    # create a copy and apply threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return binary_output


# In[6]:


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    """
    A function to get the magnitude sobel and output a binary 8 bits image
    """
    gray = grayscale(img)
    
    # Getting Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # get gradient magnitude
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # rescale to 8 bit
    scale_factor = np.max(grad_mag)/255
    
    grad_mag = (grad_mag/scale_factor).astype(np.uint8)
    
    # create a copy and apply threshold
    binary_output = np.zeros_like(grad_mag)
    binary_output[(grad_mag >= thresh[0]) & (grad_mag <= thresh[1])] = 1
    
    return binary_output


# In[7]:


def dir_thresh(img, sobel_kernel=3, thresh=(0,np.pi/2)):
    """
    a function to get the direction sobel and output a binary 8 bits image
    """
    gray = grayscale(img)
    
    # Getting Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # get absolute value of gradient direction
    abs_grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    
    # create a copy and apply threshold
    binary_output = np.zeros_like(abs_grad_dir)
    binary_output[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1
    
    return binary_output


# In[8]:


def hls_select(img, thresh=(0,255)):
    """
    A function that converts the image from RGB to HLS and apply thresholdd
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    
    return binary_output


# In[9]:


def warp_image(img):
    """""
    A function to transform the image to Bird's eye view to make it easy
    in radius of curvature and position calculations
    """""
    img_size = (img.shape[1], img.shape[0])
    
    #src = np.float32([
    #   [800,510],
    #   [1150,700],
    #   [270,700],
    #   [510,510]
    #])
    
    #dst = np.float32([
    #   [650,470],
    #   [640,700],
    #   [270,700],
    #   [270,510]
    #])
    
    src = np.float32(
        [[273,680],
         [1046,680],
         [585,455],
         [700,455]])

    dst = np.float32(
        [[300,720],
         [980,720],
         [300,0],
         [980,0]])
    
    # get perspective
    M = cv2.getPerspectiveTransform(src,dst)
    
    # get inverse perspective
    Minv = cv2.getPerspectiveTransform(dst,src)
    
    # warp image
    warped_image = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    # unwar image
    unwarped_image = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    
    return warped_image, unwarped_image, M, Minv


# In[10]:


def find_lane_pixels(warped_image, plot= False):
    
    #Try changing nwindows from 9 and see what is the difference
    
    # make a copy to work on
    binary_warped = warped_image.copy()
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    #midpoint = np.int(histogram.shape[0]//4)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    
    # we shouldn't take the range from midpoint to end because it will detect
    # the lane line in the end of the photo not the end of the same lane line
    rightx_base = np.argmax(histogram[midpoint:midpoint + 500]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    if plot == True:
        plt.figure(figsize=(10,10))
        fig = plt.figure()

        plt.imshow(out_img)
        plt.plot(leftx, ploty, color='yellow')
        plt.plot(rightx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, plot=False)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [255, 0, 0]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return left_fit, right_fit, out_img


# In[11]:


def fit_continuous(left_fit, right_fit, warped_image, plot = True):
    
    binary_warped = warped_image.copy
    
    #Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 50
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
                      & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
                       & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    if len(leftx) == 0:
        left_fit_new =[]
    else:
        left_fit_new = np.polyfit(lefty, leftx, 2)
    
    
    if len(rightx) == 0:
        right_fit_new =[]
    else:
        right_fit_new = np.polyfit(righty, rightx, 2)
        
    if plot == True:
    
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
   
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    
    return left_fit_new, right_fit_new, result


# In[12]:


def lane_detection(img_file, apply_blur=False):
    """""
    A pipeline function to get the output with one function,
    It is combining some lane detection the steps in one function
    """""
    
    # read image
    #image = cv2.imread(img_file)
    image = img_file
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # if blur argument is true define a kernel size of 5x5 and apply
    # gaussian blur
    if(apply_blur == True):
        kernel_size = 5
        image = gaussian(image, kernel_size)
        
    
    # applying gradient and color thresholding then combining
    sobelx_bin = abs_sobel_thresh(image, orients='x', sobel_kernel=3, thresh=(22,100))
    mag_bin = mag_thresh(image, sobel_kernel=3, thresh=(40, 100))
    dir_bin = dir_thresh(image, sobel_kernel=3, thresh=(0.7, 1.3))
    
    S_channel_bin = hls_select(image, thresh=(90,255))
    
    # Combine different binary thresholds
    combined_bin1 = np.zeros_like(sobelx_bin)
    combined_bin1[(sobelx_bin == 1) | (S_channel_bin == 1)] = 1
    
    combined_bin2 = np.zeros_like(sobelx_bin)
    combined_bin2[(sobelx_bin == 1) | (S_channel_bin == 1) | (mag_bin == 1)] = 1
    
    # Apply perspective transform
    # return warped_image, unwarped_image, M, Minv of warp_image(img) function
    # we will need the Minv later
    # combined_bin1 is the best to work with
    
    warped_image, _,_,_ = warp_image(combined_bin1)
    
    return image, sobelx_bin, S_channel_bin, combined_bin1, combined_bin2, warped_image


# In[13]:


#Calculate Curvature

def curvature(left_fit, right_fit, binary_warped, print_data = True):
    """""""""
    This function calculates the radius of curvature of the road and the position
    from the center of the lane
    """""""""
    #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    ploty = np.linspace(0, binary_warped.shape[0], binary_warped.shape[0])
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    #Define left and right lanes in pixels
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #Identify new coefficients in metres
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    #left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    #right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    #Calculation of center
    #left_lane and right lane bottom in pixels
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    # Lane center as mid of left and right lane bottom
                            
    lane_center = (left_lane_bottom + right_lane_bottom)/2.
    #center_image = 640
    #center = (lane_center - center_image)*xm_per_pix #Convert to meters
    car_position = binary_warped.shape[1]/2
    lane_center_position = (leftx[719] + rightx[719]) / 2
    center = (car_position - lane_center_position) * xm_per_pix
    
    #if print_data == True:
    #Now our radius of curvature is in meters
        #print(left_curverad, 'm ', right_curverad, 'm ', center, 'm ')

        #print("Left Curvature: ", left_curverad, 'm \t', "Right Curvature: ", right_curverad, 'm ')
        #print("Position from center: ", center, 'm ')

    return left_curverad, right_curverad, center


# In[14]:


def project_lane_line(warped_image, left_fit, right_fit, orig_img):
    """""""""
    A function to fill the lane and unwarp the image
    """""""""
    new_img = np.copy(orig_img)
    binary_img = np.copy(warped_image)
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
    ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0])
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,0), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255,0,0), thickness=15)
    
    filled_warp = np.copy(color_warp)
    _, newwarp,_,_ = warp_image(color_warp)
    
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result, filled_warp


# In[15]:


def write_on_Image(orig, left_curve,right_curve,center_dist):
    """""""""
    A function to print info on the original image
    """""""""
    test_img = np.copy(orig)
    font = cv2.FONT_HERSHEY_DUPLEX
    rds_of_curve = (left_curve +right_curve)/2
    
    text = 'Radius of curvature: ' + '{:04.2f}'.format(rds_of_curve) + 'm'
    #cv2.putText(test_img, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    cv2.putText(test_img, text, (40,70), font, 1.5, (228, 242, 68), 2, cv2.LINE_AA)
    if center_dist > 0:
        direction = 'right'
    else:
        direction = 'left'
    text = '{:04.3f}'.format(abs(center_dist)) + 'm '+ direction + ' of center'
    #cv2.putText(test_img, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    cv2.putText(test_img, text, (40,120), font, 1.5, (228, 242, 68), 2, cv2.LINE_AA)
    
    return test_img


# In[16]:


def Print_stacked_image(imgArray, resize_param):
    """""
    A function that will take an array of images and stack them together to get
    the stacked image that is used in debugging mode
    """""
    imgArray[0] = cv2.resize(imgArray[0], (0, 0), None, resize_param, resize_param)
    
    for i in range(1,6):
        imgArray[i] = cv2.resize(imgArray[i], (0, 0), None, resize_param/2, resize_param/2)

    h1, w1 = imgArray[0].shape[:2]
    h2, w2 = imgArray[1].shape[:2]
    
    imgStack = np.zeros((3*h2, 3*w2, 3), dtype=np.uint8)
    imgStack[:,:] = (255,255,255)


    for i in range(0,6):
        if imgArray[i].ndim == 2:
            imgArray[i] = 255 * np.expand_dims(imgArray[i], axis=2)
            

    imgStack[:h1, :w1,:3] = imgArray[0]
    imgStack[:h2, w1:w1+w2, :3] = imgArray[1]
    imgStack[h2:h1, w1:w1+w2, :3] = imgArray[2]
    imgStack[h1:h1+h2, w1:w1+w2, :3] = imgArray[3]
    imgStack[h1:h1+h2, :w2, :3] = imgArray[5]
    imgStack[h1:h1+h2, w2:w1, :3] = imgArray[4]

    #cv2.imshow('imgStack',imgStack)
    return imgStack


# ------------------------------------------------------------------

# ### Vehicle Detection and Tracking - Implemented Functions:

# In[17]:


def draw_boxes(img, bboxes, color=(255, 0, 0), thick=6):
    """""""""
    a function that takes an image, a list of bounding boxes, 
    and optional color tuple and line thickness as inputs
    then draws boxes in that color on the output
    """""""""
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # return the image copy with boxes drawn
    return draw_img # Change this line to return image copy with boxes


# In[18]:


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """""""""
    A function to compute the histograms of R,G and B channels seperately
    """""""""
    # Compute the histogram of the RGB channels separately
    channel1_hist = np.histogram(img[:,:,0], bins = nbins, range = bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins = nbins, range = bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins = nbins, range = bins_range)
    
    # Generating bin centers
    bin_edges = channel1_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
    # Return the individual histograms, bin_centers and feature vector
    return channel1_hist, channel2_hist, channel3_hist, bin_centers, hist_features


# In[19]:


def bin_spatial(img, size=(32, 32)):
    """""""""
    A function to get the spatial bin as 1D feature vector to feed to classifier
    """""""""
        
    # resize the image and use .ravel() to create the feature vector to feed into classifier
    features = cv2.resize(img, size).ravel() 
    
    # Return the feature vector
    return features


# In[20]:


def extract_features_spatial(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    """""""""
    A function to extract features from a list of images
    """""""""
    
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
    # Read in each one by one
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # may be reversed when integrate with video frames
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: 
            feature_image = np.copy(image)  
            
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)

        # Apply color_hist() to get color histogram features
        _, _, _, _, hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
        
    # Return list of feature vectors
    return features


# In[21]:


def extract_features_hog(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # may be reversed when integrate with video frames
        
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            
        # Append the new feature vector to the features list
        features.append(hog_features)
        
    # Return list of feature vectors
    return features


# In[22]:


def convert_image(image, cspace):
    '''''
    Param: input image in RGB format
    Param: Desired image colorspace
    Returns: image in desired colorspace
    '''''
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: 
        feature_image = np.copy(image) 
        
    return feature_image


# In[23]:


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()

def extract_features(img, params):
    
        file_features = []
        # Read in each one by one
        # apply color conversion if other than 'RGB'
        feature_image = convert_image(img, params.color_space)    

        if params.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=params.spatial_size)
            file_features.append(spatial_features)
            
        if params.hist_feat == True:
            # Apply color_hist()
            _, _, _, _, hist_features = color_hist(feature_image, nbins=params.hist_bins)
            file_features.append(hist_features)
            
        if params.hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if params.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        params.orient, params.pix_per_cell, params.cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)  
                
            else:
                hog_features = get_hog_features(feature_image[:,:,params.hog_channel], params.orient, 
                            params.pix_per_cell, params.cell_per_block, vis=False, feature_vec=True)
                
            # Append the new feature vector to the features list
            file_features.append(hog_features)
            
        # Return list of feature vectors
        return np.concatenate(file_features)


# In[40]:


def extract_features_train(img, params):
    
        file_features = []
        # Read in each one by one
        # apply color conversion if other than 'RGB'
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        feature_image = convert_image(img, params.color_space)    

        if params.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=params.spatial_size)
            file_features.append(spatial_features)
            
        if params.hist_feat == True:
            # Apply color_hist()
            _, _, _, _, hist_features = color_hist(feature_image, nbins=params.hist_bins)
            file_features.append(hist_features)
            
        if params.hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if params.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        params.orient, params.pix_per_cell, params.cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)  
                
            else:
                hog_features = get_hog_features(feature_image[:,:,params.hog_channel], params.orient, 
                            params.pix_per_cell, params.cell_per_block, vis=False, feature_vec=True)
                
            # Append the new feature vector to the features list
            file_features.append(hog_features)
            
        # Return list of feature vectors
        return np.concatenate(file_features)


# In[24]:


def data_look(car_list, notcar_list):
    """""""""
    A function that extract info of the dataset that will be used to train the classifier
    """""""""
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    
    # Read in a test image, either car or notcar
    example_img = cv2.imread(car_list[0])
    example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
    
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    
    return data_dict


# In[25]:


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    """""""""
    A function to return HOG features
    
    The Function accepts params and returns HOG features (optionally flattened) and an optional matrix for 
    visualization. Features will always be the first return (flattened if feature_vector= True).
    A visualization matrix will be the second return if visualize = True.
    """""""""
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                                  transform_sqrt=True, 
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualize=vis, feature_vector=feature_vec)
        return features


# In[26]:


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
        
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    
    # Initialize a list to append window positions to
    window_list = []
    
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
            
    # Return the list of windows
    return window_list


# In[27]:


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True): 
    
    #1) Define an empty list to receive features
    img_features = []
    
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)     
        
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        
        #4) Append features to list
        img_features.append(spatial_features)
        
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        _, _, _, _, hist_features = color_hist(feature_image, nbins=hist_bins)
        
        #6) Append features to list
        img_features.append(hist_features)
        
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True)) 
                
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


# In[28]:


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())

def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    
    #2) Iterate over all windows in the list
    for window in windows:
        
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        
        #5) Scale extracted features to be fed to classifier
        #test_features = scaler.transform(features)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
            
    #8) Return windows for positive detections
    return on_windows


# In[29]:


def add_heat(heatmap, bbox_list):
    
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes


# In[30]:


def apply_threshold(heatmap, threshold):
    
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    
    # Return thresholded map
    return heatmap


# In[31]:


def draw_labeled_bboxes(img, labels):
    
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        
    # Return the image
    return img


# In[32]:


# Define a function that will collect all processing to find cars in the frame with sliding window technique

def find_cars_hog_sub(img, ystart, ystop, svc, scaler, params, cells_per_step = 1):
    draw_img = np.copy(img)
    cspace = params.color_space
    
    # croping the image to a smaller frame
    img_tosearch = img[ystart:ystop,:,:]

    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
            
    else: feature_image = np.copy(img_tosearch)  
    
    if params.scale != 1:
        imshape = feature_image.shape
        feature_image = cv2.resize(feature_image, (np.int(imshape[1] / params.scale), np.int(imshape[0] / params.scale)))
        
    ch1 = feature_image[:,:,0]
    ch2 = feature_image[:,:,1]
    ch3 = feature_image[:,:,2]
    
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // params.pix_per_cell) - params.cell_per_block + 1
    nyblocks = (ch1.shape[0] // params.pix_per_cell) - params.cell_per_block + 1 
    nfeat_per_block = params.orient * params.cell_per_block**2
    
    window = 64
    nblocks_per_window = (window // params.pix_per_cell) - params.cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, params.orient, params.pix_per_cell, params.cell_per_block, vis=False, feature_vec=False)
    hog2 = get_hog_features(ch2, params.orient, params.pix_per_cell, params.cell_per_block, vis=False, feature_vec=False)
    hog3 = get_hog_features(ch3, params.orient, params.pix_per_cell, params.cell_per_block, vis=False, feature_vec=False)
    car_windows = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos*params.pix_per_cell
            ytop = ypos*params.pix_per_cell
            
            # Extract the image patch
            subimg = cv2.resize(feature_image[ytop:ytop + window, xleft:xleft + window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size = params.spatial_size)
            _, _, _, _, hist_features = color_hist(subimg, nbins = params.hist_bins, bins_range = params.hist_range)
            
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft * params.scale )
                ytop_draw = np.int(ytop * params.scale )
                win_draw = np.int(window * params.scale )
                cv2.rectangle(draw_img,(xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart),(0,0,255),6) 
                car_windows.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                
    return car_windows


# In[33]:


def heat_threshold(img, threshold, svc, X_scaler, windows_list, params):

    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat,windows_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    return draw_img, heatmap


# In[34]:


class Parameters():
    """""""""
    Defined a class Parameters that will take all parameters and store it to
    use in different functions
    """""""""
    
    color_space='RGB'
    spatial_size=(32, 32)
    hist_bins=8
    orient=9
    pix_per_cell=8
    cell_per_block=2
    hog_channel=0
    hist_range = (0, 256)
    spatial_feat=True
    hist_feat=True
    hog_feat=True
    
    def __init__(self, color_space='RGB', spatial_size=(32, 32),
                 hist_bins=8, orient=9, 
                 pix_per_cell=8, cell_per_block=2, hog_channel=0, scale = 1.5,hist_range = (0, 256),
                 spatial_feat=True, hist_feat=True, hog_feat=True):
        # HOG parameters
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.scale = scale
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.hist_range = hist_range


# In[35]:


def track_vehicle(img):
    
    ystart = 400
    ystop = 650
    threshold = 1 
    car_windows = find_cars_hog_sub(img, ystart, ystop, svc, X_scaler, params)
    draw_img, heat_map = heat_threshold(img, threshold, svc, X_scaler, car_windows, params)
    
    return draw_img


# -------------------------

# ### Pipline and Processing - Implemented Functions:

# In[36]:


def pipeline(img, resize_param):
    """""""""
    This will be the final collected function to apply all processing
    
    It should take:
    boolean: blur
    int: blur kernel size
    boolean: video or not
    
    To be completed
    and it will take input string from the user with file name to work on
    if video it should extract frames to work on and then recollect them again
    """""""""
    #file, debug = take_filename
    image = img
    
    counter = 0
    #global last_left 
    #global last_right
    #global left_fit
    #global right_fit
    
    #inputVideo = './project_video.mp4' 
    #outputVideo ='./output_videos/project_video.mp4'

    orig, sobelx_bin, S_channel_bin, combined_bin1, combined_bin2, warped_image = lane_detection(image, apply_blur=False)

    if counter == 0:
        left_fit, right_fit, out_img = fit_polynomial(warped_image)
    else:
        left_fit, right_fit, out_img  = fit_continuous(left_fit, right_fit, warped_image, plot = False)

    
    # output tracked vehicle frame
    #tracked = track_vehicle(orig)
    
    
    # Get radius of curvature and position
    # return left_curverad, right_curverad, center
    left_curverad, right_curverad, center = curvature(left_fit, right_fit, out_img)

    #fill lane and unwarp
    filled_img, filled_warp = project_lane_line(warped_image, left_fit, right_fit, orig)

    # Print radius of curvature and position on the original photo
    final = write_on_Image(filled_img, left_curverad, right_curverad, center)
    
    
    # output tracked vehicle frame                 #added new
    final = track_vehicle(final)
    

    #plt.imshow(out_img)
    img_stack = Print_stacked_image([final, image, sobelx_bin, warped_image, out_img, filled_warp], resize_param)
    
    #return final
    #return sobelx_bin, warped_image, out_img, final, filled_warp, img_stack
    return final, img_stack


# In[37]:


def Prcoess_video(filename, stacked = 0, resize_param = 0.5, save = 0, show = 1):
    """""
    A function that processes the video and you can choose whether to show in terminal, save output video
    without showing video in terminal, show video in terminal and save output video or do nothing

    This function also make the user choose whether to display normal mode or debugging mode
    """""
    
    video = cv2.VideoCapture(filename)

    # Naming output video without using new input parameters
    output_name = filename.replace(".mp4","")
    output_name = output_name + '_out.mp4'
    if(stacked == 1):
        output_name = output_name.replace(".mp4","")
        output_name = output_name + '_stacked.mp4'

    

    if (video.isOpened() == False): 
        print("Error reading video file")
    
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    if(stacked):
       frame_width = int(frame_width * resize_param * 1.5)
       frame_height = int(frame_height * resize_param * 1.5)
    
    if(save):   
        result = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'MJPG'), 25, (frame_width, frame_height))
        
    while(video.isOpened()):
        ret, frame = video.read()
        
        if(ret): 
            final, img_stack = pipeline(frame, resize_param)

            if(stacked):
                if(save & show):
                    result.write(img_stack)
                    cv2.imshow('img_stack', img_stack)
                elif(save):
                    result.write(img_stack)
                else:
                    cv2.imshow('img_stack', img_stack)
           
            else:
                if(save & show):
                    result.write(final)
                    cv2.imshow('final', final)
                elif(save):
                    result.write(final)
                else:
                    cv2.imshow('final', final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        else:
            break

    video.release()
    result.release()
    cv2.destroyAllWindows()
    print("The video was successfully saved")



# In[ ]:

# Cell for taking arguments from terminal
def play():
    if( len(sys.argv) == 2 ):
        Prcoess_video(sys.argv[1])

    elif(len(sys.argv) == 3):
        Prcoess_video(sys.argv[1], int(sys.argv[2]))

    else:
        Prcoess_video( sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) )

        #Prcoess_video(filename, stacked = 1, resize_param = 0.5, save = 0, show = 1)


# In[42]:


params = Parameters(
            color_space = 'YCrCb',
            spatial_size = (16, 16),
            orient = 8,
            pix_per_cell = 8,
            cell_per_block = 2,
            hog_channel = 'ALL',
            hist_bins = 32,
            scale = 1.5,
            spatial_feat=True, 
            hist_feat=True, 
            hog_feat=True
        )
    
    

# Divide up into cars and notcars
# Read in car and non-car images
images = glob.glob("dataset/*png")
car_list = []
notcar_list = []

for image in images:
    if 'image' in image or 'extra' in image:
        notcar_list.append(image)
    else:
        car_list.append(image)

        
car_features = list(map(lambda img: extract_features_train(img, params), car_list))
notcar_features = list(map(lambda img: extract_features_train(img, params), notcar_list))


X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
#rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=42)

print('Using:',params.orient,'orientations',params.pix_per_cell,
    'pixels per cell and', params.cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


# In[43]:


#Prcoess_video('project_video.mp4', resize_param = 0.5, stacked = 0, save = 0, show = 1)

play()


