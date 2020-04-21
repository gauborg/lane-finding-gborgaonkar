# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

### Overview
---
When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project code, we will detect lane lines in images using Python and OpenCV. OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.
We use Jupyter Notebook to run our code. I have also created an environment using Anaconda in which I have installed the necessary libraries such as OpenCV, NumPy, Matplotlib, Math, MoviePy libraries.

### 1. The Project pipeline explanation
---

I have defined separate functions for different image processing steps involved. First, we run our tool on individual images to adjust parameters and then we apply the code to videos. My code pipeline consists of following steps:

1. Convert the image to grayscale using [cv2.cvtColor()](https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor).
2. We apply Gaussian filter to suppress any noise present in our images using OpenCV inbuild gaussian_blur function.
3. Use Canny Edge Detection from OpenCV to detect edges in our test images. I experimented with different values and applied thresholds of 50 and 150 for edge detection.
4. Using region masking principles, we specify the points which mark the endpoints of our region of interest. We specify the vertices approximately by looking at the image (these values can be tweaked) and specify the points in the form of array in image space.
5. Apply Hough Transform to identify lines in the region of interest.

After this step, to draw the lanelines, we write our code in the **draw_lines()** function which takes as input an image, detected canny lines in our region of interest and color options for drawing the lines.

In the **draw_lines()** function, I used the following principles to identify lanelines. In order to draw lanelines, we have to calculate the equation of the two lanelines given by the formula [**y = mx + b**](https://en.wikipedia.org/wiki/Slope).

1. We calculate the slope of a line, we get the end co-ordinates of the line (x1, y1, x2, y2).

![slope of a line](/readMe_images/slope_formula.png)

2. After calculating the slope for the line, we check if the slope is within a specified range of 0.8 and 0.5 for left lanelines (since they have a positive slope) and _0.8 and -0.5 for right lanelines (as they have a negative slope).
3. Then, we check if the end points of a line have both endpoints on the left side of the image, then it is a left lane. If no, then it is a right lane.
4. We create two empty lists *left_slope* and *right slope* to store left and right lines. We apply the steps 2 and 3 to classify the lines using **left_slope.append(line)**. Using the equation **y = mx + b**, we calculate and store the b intercept in two lists *left_b* and *right_b*.
5. We calculate the average slope and b intercept for the left line and for the right line.
6. To draw the lines, we specify the y co-ordinates. The lower y co-ordinate for the laneline will be at the bottom of the image, i.e. image.shape[0] and higher y co-ordinate is specified as 330.
7. We draw the lines using **cv2.line()** from OpenCV.

**Note:** In our code, we have to take care that the loop doesn't fail, if it encounters a division by zero in the slope formula. Hence, we have included the **ZeroDivisionError** in our loop.

### 2. Potential Shortcomings of the Pipeline

1. It uses fixed parameters for thresholding and does not take into account the variance in road textures, shadows, etc.
2. The pipeline does not take into account any possible curvatures of the road.
3. The laneline detection algorithm we have developed does not separate between different types of lane lines (such as continuous lines, double yellow, single white line, etc.)

### 3. Scope for improvements to the pipeline

1. We should experiment more with multiple thresholding options for different channels in RGB/HSV/HLS formats.
2. We can use a curve equation like **ax^2 + bx + c** for capturing road curvatures.
3. We can also improve the parameters for higher accuracy.

