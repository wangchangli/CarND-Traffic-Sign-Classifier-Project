#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is len(X_train)
* The size of the validation set is len(X_valid)
* The size of test set is len(X_test)
* The shape of a traffic sign image is X_train[0].shape
* The number of unique classes/labels in the data set is len(set(y_train))

####2. Include an exploratory visualization of the dataset.

See the first code cell output of "Include an exploratory visualization of the dataset" in the traffic_Sign_Classifier.html for the visualization of the dataset(include data distribution chart).

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color of the traffic sign doesn't affect the result, and it also accelerates the training process. 

See the first code cell output of "Pre-process the Data Set"  in the traffic_Sign_Classifier.html for the gray images.

As a last step, I normalized the image data  to the range (-1,1) because it makes the optimizer to do its job easily as mentioned in the lesson.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16 |
| RELU	|                                        |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 	|
| fallten	|           Input = 5x5x16. Output = 400. |
| Fully connected		| Input = 400. Output = 120.  |
| RELU	|                                        |
| Fully connected		|Input = 120. Output = 84  |
| RELU	|                                        |
| Dropout	|                       keep_pro=0.55                 |
| Fully connected		|Input = 84. Output = 43  |
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an settings of:
* Adam Optimizer
* batch size 128
* eproch 100
* learining rate 0.0008
* mu 0
* sigma 0.1

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of  ?
* validation set accuracy of 0.937
* test set accuracy of 0.925

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
**I used the LaNet architecture first, because it is a well known CNN architecture for image classification.**
* What were some problems with the initial architecture?
**The input dataset is not normalized, the architecture may overfitting.**
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
**I normalized the input data, and add dropout to reduce the overfitting risk.**
* Which parameters were tuned? How were they adjusted and why?
**Eproch was tuned like this, 20 -> 50 -> 100, I stopped at 100, because I found the tranning begins to converge.**
**Learning rate of was set to 0.0008 finally after trying other values.**
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
**A dropout layer can reduce the overfitting risk.**
If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

See the Traffic_Sign_Classifier.html for details.
The images I picked contain three characters that I think may be difficult to classify:1) some images contain watermarks; 2) The traffic signs in some images are not in the center as the dataset show;3) some images are not token from the front.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The accuracy is terrible(0 out of 5), I think the main reason for this result is that the training dataset is so limited, as we can see from the dataset distribution, some traffic signs only got a few training datas. I will improve this by generate some fake data from the exist train dataset.

Another reason is that there must be some better architecture out there for this task, I will improve my architecture after some study and experiment.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

See the first code cell output of "Output Top 5 Softmax Probabilities ..." in the traffic_Sign_Classifier.html.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


