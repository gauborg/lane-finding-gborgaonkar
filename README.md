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

In the **draw_lines()** function, I used the following principles to identify lanelines.

<<<<<<< HEAD
||||||| merged common ancestors
## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
=======
1. We filter every line from our list of detected lines (using Canny) by specifying a slope range. The lane lines on the left side of the image will have positive slope whereas those on the right side have negative slope.
>>>>>>> a8f34c8afc132316c8e2b47797bfcf6e3cede36c

![slope of a line](/readMe_images/slope_formula.png)
