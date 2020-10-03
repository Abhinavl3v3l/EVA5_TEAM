
1> Abhinav Rana (rabhinavcs@gmail.com)

2> Prashant Shinagare (techemerging1@gmail.com)

3> Pruthiraj Jayasingh (data.pruthiraj@gmail.com)

4> Madhu Charan (madhucharan512@gmail.com)

# Assignment 10 : Cifar-10,Restnet 18 Pluggin.

About the Code :

1. Data_utils.py : This files contain All the data Transforms , and data loading functions.
2. Model_cifar.py: Specific to model building and Designing . Every time we work in a new data set can build a new file with respective architecture  (only file we need to create in every data set change)
3. model_utils.py: this function is based with Training function , testing function , building_model function , getting model summary function , get_test_accuracy, Class based accuracy function etc. 
4. plot_utils.py: This is the function responsible for plotting. it is having the sample plotting function , miss classification function and the model performance plot function (accuracy,loss)
5. regularization.py: This file contents all the regularization functions.
6.Models.py : Restnet18 Model is here.

About Cifar - 10:

1. Total parameter used: 11,173,962

2. Number of epochs : 50

3. Best test accuracy : 91.54%

4. Best train accuracy : 94.76%

   ```

Analysis: the model is over fit as the gap between training and testing is high after cert-en epochs . can use image augmentation and batch norm to overcome this. 

plots:

<img src="https://github.com/pruthiraj/EVA5_TEAM/blob/master/session10/Train_and_Test_Curves.png" alt="Train and Test Curve" >
Miss classification images with gradcam applied :
<img src="https://github.com/pruthiraj/EVA5_TEAM/blob/master/session10/Misclassified_gradcam.png" alt="MisClassified images on GradCam" >


