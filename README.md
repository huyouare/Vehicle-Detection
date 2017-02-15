## Vehicle Detection Project

##### The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./images/hog_features.png
[image3]: ./images/sliding_window.png
[image4]: ./images/heat_map.png
[image5]: ./images/all_heat_maps.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./result.mp4


##### References:

https://classroom.udacity.com/nanodegrees/nd013/
https://github.com/nhiddink/CarND_P5_Vehicle_Detection_and_Tracking
https://github.com/ksakmann/CarND-Vehicle-Detection

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `Vehicle Detection.ipynb` under the heading `2. Histogram of Oriented Gradients (HOG) Feature Extraction`. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Extracting features was straightforward.

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, using roughly a grid search where I changed the parameters by factors of 2. It was after verifying the result visually, and after verifying the accuracy and run time of running the HOG parameters in classification did I arrive at my final choice. It was also influenced by the parameters chosen in the course and reference projects.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in `Vehicle Detection.ipynb` under the headings `4. Color Classification` and `5. Classification using HOG features`. 
I trained a linear SVM using all the L channel of the HLS space. This was chosen through visual inspection of the images and color map, as well as through prior success with the HLS color space. The other parameters were chosen by default values, or suggested values from the course or other projects, and by manual grid search. The resulting accuracy was 97%. The time to predict was roughly 2.4 milliseconds, allowing for over 400 predictions per second. Assuming a video frame rate of 24 frames per second, this allows for only 16 windows to be classified in series. However, parallel computation would allow for more efficient classification.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started by inspecting the minimum and maximum X and Y values manually by observing the cars in the sample video. I then experimented with several approaches, starting from 0.5 overlap, and approaching 0.9 overlap. Although using windows of multiple sizes with varying offset was helpful initially, using a single window size with high overlap produced the best results in terms of maximizing true positives and minimizing false positives. The size was varied to best fit the expected car size in the video. The downside to having one window size is the possibility of missing cars that are of much larger or smaller size.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Reducing the color channel to use only L in HLS, and using a smaller bounds for X and Y in sliding windows helped to optimize performance. I also chose a larger size of HOG pixels per cell, and decided to only use one size of sliding window. From the example results, the pipeline seems to work great.

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./result.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I found that a threshold value of 3 worked best. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:
![alt text][image4]

#### Here are six frames and their corresponding heatmaps and resulting bounding boxes:

![alt text][image5]




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Similar to the previous project, the use of HOG and color features with many parameters requires careful hand-tuning. This hand-tuning is also particular to the data and data collection process. We would likely see the pipeline fail in other data sets, where the cars are located further to the left, or there are other obstructions and car types. The classifier also has issues with shadows and other objects with high gradient values. To make the video pipeline more robust, we can use bounding box values from previous frames to influence values in future frames, and use smoothing functions to avoid extraneous detections. To overcome issues with hand-tuning parameters, we can look at end-to-end and deep learning solutions.

