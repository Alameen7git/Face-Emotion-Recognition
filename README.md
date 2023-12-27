
# Face-Emotion-Recognition
Facial Emotion Recognition / FER2013 dataset/ using a Convolutional Neural Network and combining it with deep learning (OpenCV) for live emotion recognition

![image](https://github.com/Alameen7git/Face-Emotion-Recognition/assets/110742159/6ee816b8-39e1-4598-933d-a9be978342b9)

The dataset is imported from Kaggle 
Kaggle dataset- https://www.kaggle.com/datasets/subhaditya/fer2013plus

This model has a  final train accuracy = 75.31% and  validation accuracy = 71.98%
______________________________________________________________________________________________________________________________________________
GETTING STARTED:

These instructions will be based on training and validation of the dataset using Convolutional Neural Network  to recognize facial emotion recognition using images.
This project uses the trained model to detect facial emotion recognition through a webcam. In the future, it can be used in different applications.

________________________________________________________________________________________________________________________________________________
REQUIRED LIBRARIES

-----------------------------------------------------------------------------------------------------------------------------------------------
                  import matplotlib.pyplot as plt
                  import numpy as np
                  import pandas as pd
                  import seaborn as sns
                  import os
                  from tensorflow import keras
                  
#Importing deep learning libraries

                  from keras.preprocessing.image import ImageDataGenerator
                  from keras.preprocessing.image import load_img, img_to_array
                  from keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D
                  from keras.models import Model,Sequential
                  from keras.optimizers import Adam,SGD,RMSprop
-----------------------------------------------------------------------------------------------------------------------------------------------                 
________________________________________________________________________________________________________________________________________________
DOWNLOADING DATASET AND DEFINING FILE PATH:

#https://www.kaggle.com/datasets/subhaditya/fer2013plus

                  folder_path = "C:/Users/admin/Desktop/mfer2013/archive (20)/fer2013plus/fer2013/
___________________________________________________________________________________________________________________________________________________
DEVELOPING A CNN LAYER 

CNN (Convolutional Neural Networks) in image processing because they can effectively extract features from images and learn to recognize patterns, making them well-suited for image detection
This has 4 CNN layers and 2 fully connected layers

The layers in the Convolution Neural Network used in implementing this classifier can be summarized as follows. model.summar()

----------------------------------------------------------------------------------------------------------------------------------------------- 
                  Total params: 3141128 (11.98 MB)
                  Trainable params: 3136648 (11.97 MB)
                  Non-trainable params: 4480 (17.50 KB)
----------------------------------------------------------------------------------------------------------------------------------------------- 

____________________________________________________________________________________________________________________________________
reduce_learningrate - This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.

early_stopping - early stopping is a form of regularization used to avoid overfitting and save time
____________________________________________________________________________________________________________________________________
CONFUSION MATRIX

A confusion matrix is a table used in machine learning and statistics to assess the performance of a classification model. 
Why is a confusion matrix better than accuracy?
The confusion matrix can provide valuable insights into these crucial distinctions. But accuracy will not provide such an intuition on the correctly and incorrectly classified points

![Screenshot 2023-12-27 174643](https://github.com/Alameen7git/Face-Emotion-Recognition/assets/110742159/43e5f004-c00f-47e7-9034-8aeafdba4fb8)


ACCURACY:

-----------------------------------------------------------------------------------------------------------------------------------------------                                
                  final train accuracy = 75.31 , validation accuracy = 71.98
-----------------------------------------------------------------------------------------------------------------------------------------------                 
__________________________________________________________________________________________________________________________________________________
SUMMARY:

In terms of final train accuracy, the model correctly predicted 75.31% of the target variable values in the training dataset. This indicates that the model is effective in predicting the target variable based on the input features, especially for the examples it has already seen during training.

However, when it comes to validation accuracy, the model only achieved a slightly lower accuracy of 71.98%. This is a subtle but important difference, as it suggests that the model's performance is not only reliant on the examples it has seen during training but also generalizes well to unseen data

Overall, these accuracy scores suggest that the model is capable of learning from the training data and generalizing well to new, unseen data, resulting in a high validation accuracy. This indicates that the model has been trained effectively and is likely to perform well on real-world data
_____________________________________________________________________________________________________________________________________________________

COMPUTER VISION

# Emotion Detector

This is a Python-based program that utilizes OpenCV and a pre-trained Keras model to detect human emotions in real-time using a webcam. The model has been trained on the FER2013 dataset and is capable of identifying seven different emotions.

## Requirements

 ![Static Badge](https://img.shields.io/badge/Python%203.6%20-or%20later-white)
 
 ![Static Badge](https://img.shields.io/badge/OpenCV%203.4-%20or%20later-white)
 
 ![Static Badge](https://img.shields.io/badge/TensorFlow%202.0-or%20later-white)
 
 ![Static Badge](https://img.shields.io/badge/NumPy%201.16-or%20later-white)

 ![Static Badge](https://img.shields.io/badge/Webcam-White)



 
1. Import necessary libraries
   
        from keras.models import load_model
        from time import sleep
        from keras.preprocessing.image import img_to_array
        from keras.preprocessing import image
        import cv2
        import numpy as np
   
2. Load the pre-trained model 'model1.h5' and the haarcascade classifier for face detection.

3. Initialize the webcam and create a loop that will continuously capture frames from the webcam.

4. Inside the loop, convert each captured frame to grayscale and use the face classifier to detect faces.

5. For each detected face, create a region of interest (ROI) around the face. Then, resize the ROI to the 
   required size of 48x48 pixels.
   
   Get the region of interest (ROI)
   
         if np.sum([roi_gray])!=0:
              roi = roi_gray.astype('float')/255.0
              roi = img_to_array(roi)
              roi = np.expand_dims(roi,axis=0)
   
8. Convert the resized ROI to a float array and reshape it to the shape expected by the deep learning 
   model.

9. Predict the emotion of the person in the face using the pre-trained model. The emotion with the highest 
   predicted probability is considered as the emotion of the person.

10. Draw a rectangle around the face and display the predicted emotion label on the frame.
    
11. Display the frame with the face detection and emotion recognition results using OpenCV's imshow 
   function.
12.Break the loop and release the webcam when the user presses the 'q' key.

 Please note that the model's accuracy and speed will depend on the model used. In this case, it is a Keras model ('model1.h5') that has been trained to recognize emotions in facial expressions.
____________________________________________________________________________________________________________________________
NOTE

The dataset used for training model is from kaggle and I do not own the dataset
________________________________________________________________________________________________________________________________
-------------------------------------------------------------------------------------------------------------------------------




