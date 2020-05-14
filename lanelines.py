# Self-Driving Car Engineer Nanodegree
# 
# ## Project: **Finding Lane Lines on the Road** 

# ## Import Packages

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import moviepy


image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images  
# `cv2.cvtColor()` to grayscale or change color  
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # lists to store the slopes of lines which match our criteria
    left_slope = []
    right_slope = []
    
    # lists to store the calculate b intercepts of these lines
    left_b = []
    right_b = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
            # only select lines with specific slope range
            if(((slope < 0.8) and (slope > 0.5)) or ((slope > -0.8) and (slope < -0.5))):
                # check where the endpoints lie on the image...
                if (x1 < (img.shape[1]/2) and x2 < (img.shape[1]/2)):
                    left_slope.append(slope)
                    left_b.append(y1-slope*x1)
                    left_b.append(y2-slope*x2)
                else:
                    right_slope.append(slope)
                    right_b.append(y1-slope*x1)
                    right_b.append(y2-slope*x2)
    try:
        # we calculate average slope to draw the line
        avg_left_slope = sum(left_slope)/len(left_slope)
        avg_right_slope = sum(right_slope)/len(right_slope)
            
        avg_left_b = sum(left_b)/len(left_b)
        avg_right_b = sum(right_b)/len(right_b)
        
        # Y co-ordinate of the lane line will definitely be at the bottom of the image
        y1 = img.shape[0]
        y2 = 320
        y3 = 320
        y4 = img.shape[0]
        
        # X co-ordinate can be calculated by using the eqn of the line and y co-ordinate
        x1 = (y1 - avg_left_b)/avg_left_slope
        x2 = (y2 - avg_left_b)/avg_left_slope
        x3 = (y3 - avg_right_b)/avg_right_slope
        x4 = (y4 - avg_right_b)/avg_right_slope
        
        # draw the lines, converting values to integer for pixels
        
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), color, thickness)
        
    except ZeroDivisionError as error:
        
        pass


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

import os
directory = os.listdir("test_images/")

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
def lanelines(image):
    
    # 1. Grayscaling
    gray = grayscale(image)
    
    # 2. Gaussian Blur
    blur = gaussian_blur(gray, 5)
    
    # 3. Canny Detection
    canny_edges = canny(blur, 50, 150)
    
    # 4. Region Masking
    vertices = np.array([[(0,image.shape[0]),(460,320),(500,320),(image.shape[1],image.shape[0])]], dtype=np.int32)
    selected_region = region_of_interest(canny_edges, vertices)

    mpimg.imsave(os.path.join("test_images_output/" + "output-" + i), selected_region)
    # image.save(os.path.join("test_images_output/" + i + "-canny-region-output"), format=None, dpi=(540, 960))
    
    # Hough Transform Parameters- Identify lane lines in the masked region
    
    # execute Hough Transform
    lines_image = hough_lines(selected_region, 2, np.pi/180, 25, 20, 10)
    weighted_image = weighted_img(lines_image, image)
    
    return weighted_image
    
for i in directory:
    image = mpimg.imread(os.path.join("test_images/", i))
    weighted_image = lanelines(image)
    mpimg.imsave(os.path.join("test_images_output/" + "output+" + i), weighted_image)


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:

# `solidWhiteRight.mp4`
# `solidYellowLeft.mp4`
# 
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()

# Import everything needed to edit/save/watch video clips
import imageio
from moviepy.editor import VideoFileClip


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = lanelines(image)
    return result


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'

clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)


challenge_output = 'test_videos_output/challenge.mp4'

clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)

