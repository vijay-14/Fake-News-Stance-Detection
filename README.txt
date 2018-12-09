***This file gives you information about the Authors and step by step explanation of the execution of the project***

Team name: Evil Geniuses

Authors:

1) Nikhil Reddy Pathuri, George Mason University, Fairfax, VA, USA, npathuri@gmu.edu
   Role: Bulit the code for fnc_rf.py and fnc_doc2vec.py
2) Vijayasaradhi Muthavarapu, George Mason University, Fairfax, VA, USA, vmuthava@gmu.edu
   Role: Built the code for fnc_tfidf.py
3) Dilip Molugu, George Mason University, Fairfax, VA, USA, dmolugu@gmu.edu
   Role: Built the code for fnc_mlp.py

Date Created: 12/08/2018

a) This program is presented by us as part of the term project for the subject AIT-690 (Natural Language Processing) under the guidance of Dr.Ozlem Uzuner.
   This program also serves as the solution for the problem of fake news stance detection problem which was presented in a competition called Fake News Challenge conducted in 2017.

b) Hardware Requirements: RAM - 16 GB or higher, i5 7th Gen or higher processor and GPU - nvidea GTX-1060 or higher for faster performance. 
   Estimated time to run the program if you have above mentioned hardware is about 1 Hour for each Model.

c) Inorder to make this program work first read the INSTALL.txt and install the required softwares or libraries.
   -> First run the runit.sh file from your command line and follow the instructions presented on the screen.
   -> We have presented 4 different models each in a separate python file. You can run all these files in command line after executing runit.sh and selecting respective number for the model presented in the output.
   -> Make sure that the files runit.sh, fnc_rf.py, fnc_tfidf.py, fnc_doc2vec.py, fnc_mlp.py, test_bodies.csv, test_stances_unlabeled.csv, train_bodies.csv and train_stances.csv are present in the same folder.

d) About our code and algorithm:
   
   1) Our programs reads 4 csv files which are mentioned above. These files contains Headlines and bodies for train and test sets along with stances. The programs shows the performance of model as output in the form of Confusion Matrix, Accuracy, Precision, Recall and F-score.
   2) After reading the files we remove the stopwords. This step is common for all the models presented.
   3) For fnc_rf.py, fnc_tfidf.py, and fnc_mlp we have converted the sentences into TF-IDF vectors and for fnc_doc2vec.py we have converted sentences to Document Vectors using Doc2VeC. 
   4) In fnc_rf.py we have implemented Random Forest model. The model takes TF-IDF vectors as input and classifies them into 4 classes and gives the model performance on test data as output.
   5) In fnc_tfidf.py we have implemented Convolution Neural Network (CNN) model with tf-idf vectors as input for this model and classifies into 4 classes and gives the model performance on test data as output.
   6) In fnc_doc2vec.py we have implemented CNN model with doc2vec vectors as input for this model and classifies into 4 classes and gives the model performance on test data as output.
   7) In fnc_mlp.py we have implemente Multi-Layer Perceptron (MLP) model with TF-IDF vectors as input for this model and classifies into 4 classes and gives the model performance on test data as output.
      -> In this model in addition with removing stopwords we have also deal with bias by randomly selecting the number of rows equal to the class 'agree'. In order to do this we perform oversampling on the class 'disagree' and undersampling on the classes 'discuss' and 'unrealated'.

e) Results: 

The below mentioned results just specify Acuuracy and F1 scores for each class of all the models. We have listed more comprehensive results along with the confusion matrix in the documented paper. 

Confusion Matrix of Baseline predictions on testing data:
Accuracy: 72.22
F1 scores      Class
0.00		0- Agree
0.00		1- Disagree
0.00		2- Discuss
0.84		3-Unrelated

Our Model Results:
CNN with tf-idf:
Accuracy: 72.108
F1 scores	Class
0.00		0- Agree
0.00		1- Disagree
0.01		2- Discuss
0.84		3-Unrelated

CNN with Doc2vec:

Accuracy: 72.13
F1 scores	Class
0.00		0- Agree
0.00		1- Disagree
0.00		2- Discuss
0.84		3-Unrelated

MLP with Under Sampling:

Accuracy: 41.79
F1 scores	Class
0.12		0- Agree
0.01		1- Disagree
0.16		2- Discuss
0.60		3-Unrelated

CNN with Under Sampling:

Accuracy: 41.21
F1 scores	Class
0.12		0- Agree
0.02		1- Disagree
0.20		2- Discuss
0.58		3-Unrelated

 
