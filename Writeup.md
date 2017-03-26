
# **Traffic Sign Recognition** 


[//]: # (Image References)

[image1]: ./image/n_classes_distribution.png "Visualization"
[image2]: ./image/image.png "Grayscaling"
[image4]: ./image/image1.jpeg "Traffic Sign 1"
[image5]: ./image/image2.jpeg "Traffic Sign 2"
[image6]: ./image/image3.jpeg "Traffic Sign 3"
[image7]: ./image/image4.jpeg "Traffic Sign 4"
[image8]: ./image/image5.jpeg "Traffic Sign 5"
[image9]: ./image/images.png "Traffic Signs"



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the frequency distribution of traffic sign respect to the classes

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the classifier perform better working with only one parameter. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because to avoid gradient oscillatin due to the different distributions due of features value.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into training set and validation set. I did this by using train_test_split function that split the input sets in random subsets with size 80/20 of the input. 

My final training set had 27839 number of images. My validation set and test set had 6960 and 12630 number of images.


####  3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6  				    |
| Convolution 3x3	    | 1x1 strinde, outputs 10x10x16        		    |
| Max pooling	      	| 2x2 stride, outputs 5x5x16  				    |
| Flatten               | outputs 400                                   |
| Fully connected		| output = 120     								|
| RELU					|												|
| Fully connected		| output = 84     								|
| RELU					|												|
| Fully connected		| output = 43     								|


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used 15 epochs at the beginning of each epoch I shuffle the data in order to avoid the biases due to the position of the images.
I inizialize the variable of tensorflow, and I break training data into batches and train the model on each batch. At the end of each epoch I evaluate the model on our validation data.

The optimizer in order to minimize the loss function is a stochastic gradient descend (SGD) algorithm called Adam, that estimate the loss over small subset of data (batches).
I used a batch size of 128 images for a total of 218 iteraction to complete an epoch.

The value of hiperparameter learning rate that I use is 0.01. 

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.974
* validation set accuracy of 0.952
* test set accuracy of 0.867

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The architecture that I choise for resolve the problem of image classification is the LeNet  architecture a Convolutional Neural Network.

* What were some problems with the initial architecture?

There were some problems in tuning of the parameters and in the configuration of input and output dimensions of each layear.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The activation fuction used is ReLu function. I tried to add a dropout layer in order to avoid overfitting 

* Which parameters were tuned? How were they adjusted and why?

I tuned the mu = 0 and sigma = 0.1. I adapted the shape of weight and bias vector in each layer. I reshape the input image to make dimensionally consisten with weight matrix. I set the final filter depth to 43. I try different learning rate and I found that 0.01 work well.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The important design choices are: 
the choices of CNN wiht LeNet
the conversion of the images to grayscale reducing the network filter to make a more easy classification,
the max pooling layer 


If a well known architecture was chosen:
* What architecture was chosen?
The LeNet architecture 

* Why did you believe it would be relevant to the traffic sign application?
The LeNet architecture is a CNN and it is relevant to traffic sign application because it is an image classification problem

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The final accurancy of the test set is 86%. In order to estimate how well the model has been trained I look to the validation accurancy and in the 15 epochs is 95%. The training accurancy is comparable with the validation accurancy and at is 97%
 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The last three images might be difficult to classify because they are distorced during the automatic process of resizing and cropping. The signs are not in the same central location or resolution as the training samples. I suggest to augment the training set aumenting the data permorming trasformation as rotations and scaling.

![alt text][image9]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			                        |     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| General caution    		            | General caution   						    | 
| Right-of-way at the next intersection | Right-of-way at the next intersectio			|
| Speed limit (50km/h)				    | Speed limit (80km/h)      				    |
| Speed limit (30km/h)	      	        | Dangerous curve to the right 					|
| Stop		                            | Ahead only                  					|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 
The last three images are a bit distorted during the automatic process of resizing and cropping.


