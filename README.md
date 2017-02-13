
# **Behavioral Cloning** 



**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[c]: ./images/center.jpg "Center Image"
[l]: ./images/left.jpg "Left Image"
[r]: ./images/right.jpg "Right Image"
[mhl]:  ./images/max_hardleft.jpg "Max Left to hard left image"
[mhl.1]: ./images/max_hardleft1.jpg "Recovery Image"
[mhl.2]: ./images/max_hardleft2_dark.jpg "Recovery Image"
[hsl]: ./images/hard_soft_left.jpg "hard left to soft left image"
[ssl]: ./images/soft_straight_left.jpg "soft left to straight left image"
[ssr]: ./images/soft_straight_right.jpg "straight to soft right image"
[shr]: ./images/soft_hard.jpg "soft right to hard right image"
[hmr]: ./images/hard_max_right.jpg "hard right to max right image"







### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [model.h5](./model.h5) containing a trained convolution neural network 
* [Readme.md](./Readme.md) summarizing the results

#### 2 . Submssion includes functional code
Using the Udacity provided [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983558_beta-simulator-linux/beta-simulator-linux.zip) and my drive.py file, the car can be driven autonomously around the track by executing 
```sh 
 python drive.py model.json
```

#### 3. Submssion code is usable and readable

The [model.py](./model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
I have used python generator(`data_generator` in [model.py](./model.py)) to load images in batches and process them on the fly only when you need them, which is much more memory-efficient.


### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

I have used [NVIDIA's](https://arxiv.org/pdf/1604.07316v1.pdf) end to end learning for Self Driving Car architecture which consists of a normalization layer, convolution neural network layers with 3x3 and 5x5 filter sizes and depths between 24 and 64 ([model.py](./model.py) lines 446-488) followed by fully connected layers

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers to reduce overfitting. Dataset was balanced to avoid bias towars left turns (which are plenty in the collected training data)

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, with a default learning rate 0.001

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center,left,right camera images. Left and right camera views of same image simulate the effect of car being left or right and are useful for training car to recover from the left and right sides of the road. Steering values for left image are obtained by adding 0.2 to the steering value from the center camera view and subtracted by 0.2 for a right camera view of an image

For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

I preferred [NVIDIA's](https://arxiv.org/pdf/1604.07316v1.pdf) architecture since it was used to train Self Driving Cars and the model was able to detect roads based on steering angle without ever explicity training it to detect raods.  

I have initially used data augmentation techniques(like varying brightness of images,flipping images to remove bias to compensate for large number of left turns) to generate more images and the car went off the road and failed to recover after the bridge where there is no curb. I have tried collecting more data at the place where the car failed to recover but still faced the same problem. I have added dropout layers whenver I find that the car is wobbling and dropout stabilized the car but failed at places where it has to take steep turns. I have then performed histogram analysis on the steering angles and found that steering angles is largely imbalanced(even though I flipped images to add steering angles of opposite sign). This imbalance caused the car to be biased towards either hard left or hard right or soft left or soft right. To overcome this I then came up with an idea of seperating steering angles based on some thresholds and balancing the data which is explained below.

```
steering=0 implies driving straight
steering>0 implies driving straight
steering<0 implies driving straight
```
The thresholds I used to split the steering angles were:
```
max_left   =-1
hard_left  =-0.45    
soft_left  =-0.1
straight   = 0
soft_right = 0.1
hard_right = 0.45
max_right  = 1
```

Based on these thresholds the unbalanced data has

```
342  values between max_left and hard_left    [-1 -0.45) 
3709 values between hard_left and soft_left   [-0.45 -0.1)
1714 values between soft_left and straight    [-0.1 0)
2709 values between straight  and soft_right  [0 0.1)
4221 values between soft_right and hard_right [0.1 0.45) 
379  values between hard_right and max_right  [0.45 0.1) 
```
Based on these numbers the car prefers to take to take more left and right turns in range of soft and hard which makes car to go off the road

```
279  values between max_left and hard_left    [-1 -0.45) 
1026 values between hard_left and soft_left   [-0.45 -0.1)
1714 values between soft_left and straight    [-0.1 0)
2018 values between straight  and soft_right  [0 0.1)
1714 values between soft_right and hard_right [0.1 0.45) 
279  values between hard_right and max_right  [0.45 0.1) 
``` 
By balancing steering to above numbers the model prefers to predict mostly straight,is less likely to take soft left/right and even less likely takes hard left/right 

The model trained on this data proved to be good when the car completed several laps autonomously. The car was also able to recover when it was off the road.

#### 2. Final Model Architecture

The final model architecture ([model.py](./model.py) lines 446-488) consisted of a convolution neural network with the following layers and layer sizes ...

```
Lambda layer            : Normalizes data to the range -1 and 1
Convolution layer 1     : 24 filters of 5x5 size, stride-(2,2) valid padding, Relu activation 
Convolution layer 2     : 36 filters of 5x5 size, stride-(2,2) valid padding, Relu activation
Convolution layer 3     : 48 filters of 5x5 size, stride-(2,2) valid padding, Relu activation
Convolution layer 4     : 64 filters of 3x3 size, stride-(2,2) valid padding, Relu activation
Dropout layer 1         : dropout probability 0.5
Convolution layer 5     : 64 filters of 3x3 size, stride-(2,2) valid padding, Relu activation
Dropout layer 2         : dropout probability 0.5
Fully connected layer 1 : 100 neurons
Fully connected layer 2 :  50 neurons  
Fully connected layer 3 :  10 neurons
Fully connected layer 4 :   1 neuron
```
#### 3. Creation of the Training Set & Training Process
Here are the examples of some images used in the training data that were split based on steering thresholds as stated above. 

###### Center Camera view of Image (steering = -0.34)
![alt text][c]
###### Left Camera view of same Image  (steering = -0.14)
![alt text][l]
###### Right Camera of same Image (steering = -0.54)
![alt text][r]
###### Max left to hard left (steering = -0.51)
![alt text][mhl]
###### Hard left to soft left image(steering = -0.25)
![alt text][hsl]
###### Soft left to straight image (steering = -0.09)
![alt text][ssl]
###### Straight to soft right image (steering = 0.06)
![alt text][ssr]
###### Soft right to hard right image (steering = 0.32)
![alt text][shr]
###### Hard right to max right image (steering = 0.6)
![alt text][hmr]

The ratio of images split based on steering in the data set is choosen as follows

Steering Range  | sampling ratio of images
  :-------------: | :-------------:
  -1 to -0.45  | 0.04
  -0.45 to -0.1  | 0.16
  -0.1 to 0.1  | 0.6
   0.1 to 0.45  | 0.16
  0.45 to 1  | 0.04

  
