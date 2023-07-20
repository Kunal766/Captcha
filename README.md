<a name="br1"></a> 

CS771 Assignment 2 Report

Group: CS771-LEARNERS

Name

Rick Ghosh

Saurabh

Roll No

200783

200902

200787

200534

200225

email

rickg20@iitk.ac.in

saurabha20@iitk.ac.in

rinku20@iitk.ac.in

Rinku

Kunal Nayak

Atul Kumar

kunalmayak20@iitk.ac.in

atulkumar20@iitk.ac.in

Preprint. Under review.



<a name="br2"></a> 

1 Model used: K-Nearest Neighbor(KNN)

An approach for non-parametric machine learning known as K-Nearest Neighbours (KNN) is

utilised for both classiﬁcation and regression problems. New data points are categorised in this sort

of instance-based learning according to how closely they resemble previously labelled data points.

The ”K” in KNN stands for how many closest neighbours are taken into account while generat-

ing predictions. Before using the algorithm, this hyperparameter must be supplied. Based on the

measured distances, KNN chooses the K closest neighbours and applies the class label that is most

common among them when predicting the class of a new data point. When doing regression tasks,

KNN uses the average or median of the values of the K closest neighbours as the predicted value.

Selecting a suitable value for K is a crucial KNN component. The hyperparameter K must be

appropriately set, because a lower value of K might result in overﬁtting and a higher value might

cause underﬁtting. Cross-validation or other optimisation methods are often used to identify the

ideal value of K, which depends on the particular dataset and situation at hand.

2 Training Set

The ﬁle ’reference’ had images of the sixteen hexadecimal numbers. We used the getRotationMa-

trix2D function from the opencv library to rotate (0◦, ±10◦, ±20◦, ±30◦) and generate seven images

of each digit. The digits in the images were then shifted to the top left corner.The folder containing

these one hundred twelve images, called ’ref rotated’ has been used as our training set.

We also created a numpy label vector for the training set called ’train label’ in predict.py

Figure 1: Digit zero and Digit zero, rotated 30◦ anticlockwise for the training set.

3 Isolating the Last Digit

The Captcha images are 500 x 100 pixels in size. We crop out the last digit from the image such that

the size of the new cropped part is 100 x 100 pixels. This is done since the last digit of the numbers

can determine whether they are odd or even on their own.

2



<a name="br3"></a> 

Figure 2: Captcha image, it’s isolated last digit and it’s grayscale version.

4 Converting Image into Grayscale

The ’V’ in HSV value of images represents their brightness and has values between 0 and 1. We

noticed that the brightness value of the obfuscating lines was 0.8, and the brightness value of the

background and interior color of the digits is 1. Hence all pixels with brightness value between 0.8

and 1 was replaced with white pixels. Anything that was left out was colored black.

This left us with Grayscale images of the last digit of the Captchas. The digits were then shifted to

the top left corner of the image.

5 Training our model

We used KNeighborsClassiﬁer, imported from the sklearn library to train our model. The training

set was the images contained in the folder ’ref rotated’.

The ’decaptcha’ function in predict.py used this model to classify each of the isolated last digits of

the captcha into one of the sixteen hexadecimal digits. From here we could easily predict whether

the numbers were even or odd and wrote it into the string ’labels’.

6 Testing our model and Hyperparameter

We ran our model on the two thousand captcha images given in the folder ’train’. We set the value

of hyperparameter K to be 1. We compared the predicted labels with those in the ﬁle ’labels.txt’

in the folder ’train’. We vectorized the labels (converting ’ODD’ to 1 and ’EVEN” to 0) and then

calculated the Mean Square Error (MSE) between them. The resulting MSE we got was 0.006,

which was quite clsoe to 0. It turned out that 1 was the ideal value of K, as increasing it gave us

higher MSE values.

Model Performance Comparison

Hyperparameter value K

Mean Square Error

1

0\.006

0\.162

0\.1575

0\.111

0\.127

0\.204

0\.31

2

3

4

5

10

25

50

0\.382

3

