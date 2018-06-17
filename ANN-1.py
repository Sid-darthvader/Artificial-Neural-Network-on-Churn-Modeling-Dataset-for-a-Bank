#So in this dataset we would be dealing with Churn Modeling i.e. we would be writing a Artificial Neural Network
#to find out reasons as to why and which customers are actually leaving the bank and their dependencies on one another.
#This is a classification problem 0-1 classification(1 if Leaves 0 if customer stays)
#We might use Theano or Tensorflow but the thing is that these libraries require us to write most of the Ml code from
#scratch so instead we make use of Keras which enables us writing powerful Neural Networks with a few lines of code
#Keras runs on Theano and Tensorflow and you can think it of as a Sklearn for Deep Learning
# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset save it in Pycharm Projects/Name of Project
dataset = pd.read_csv('Churn_Modelling.csv')
#Looking at the features we can see that row no.,name will have no relation with a customer with leaving the bank
#so we drop them from X which contains the features Indexes from 3 to 12
X = dataset.iloc[:, 3:13].values
#We store the Dependent value/predicted value in y by storing the 13th index in the variable y
y = dataset.iloc[:, 13].values
#Printing out the values of X --> Which contains the features
#                           y --> Which contains the target variable
print(X)
print(y)

# Encoding categorical data
# Now we encode the string values in the features to numerical values
# The only 2 values are Gender and Region which need to converted into numerical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()#creating label encoder object no. 1 to encode region name(index 1 in features)
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])#encoding region from string to just 3 no.s 0,1,2 respectively
labelencoder_X_2 = LabelEncoder()#creating label encoder object no. 2 to encode Gender name(index 2 in features)
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])#encoding Gender from string to just 2 no.s 0,1(male,female) respectively
#Now creating Dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# Part 2 - Now let's make the ANN!
'''
Listing out the steps involved in training the ANN with Stochastic Gradient Descent
1)Randomly initialize the weights to small numbers close to 0(But not 0)
2)Input the 1st observation of your dataset in the Input Layer, each Feature in one Input Node
3)Forward-Propagation from Left to Right, the neurons are activated in a way that the impact of each neuron's activation
is limited by the weights.Propagate the activations until getting the predicted result y.
4)Compare the predicted result with the actual result. Measure the generated error.
5)Back-Propagation: From Right to Left, Error is back  propagated.Update the weights according to how much they are
responsible for the error.The Learning Rate tells us by how much such we update the weights.
6)Repeat Steps 1 to 5 and update the weights after each observation(Reinforcement Learning).
Or: Repeat Steps 1 to 5 but update the weights only after a batch of observations(Batch Learning)  
7)When the whole training set is passed through the ANN.That completes an Epoch. Redo more Epochs


'''
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential#For building the Neural Network layer by layer
from keras.layers import Dense#To randomly initialize the weights to small numbers close to 0(But not 0)

# Initialising the ANN
#So there are actually 2 ways of initializing a deep learning model
#------1)Defining each layer one by one
#------2)Defining a Graph
classifier = Sequential()#We did not put any parameter in the Sequential object as we will be defining the Layers manually

# Adding the input layer and the first hidden layer
#This remains an unanswered question till date that how many nodes of the hidden layer do we actually need
# There is no thumb rule but you can set the number of nodes in Hidden Layers as an Average of the number of Nodes in Input and Output Layer Respectively.
#Here avg= (11+1)/2==>6 So set Output Dim=6
#Init will initialize the Hidden Layer weights uniformly
#Activation Function is Rectifier Activation Function
#Input dim tells us the number of nodes in the Input Layer.This is done only once and wont be specified in further layers.
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#Sigmoid activation function is used whenever we need Probabilities of 2 categories or less(Similar to Logistic Regression)
#Switch to Softmax when the dependent variable has more than 2 categories

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)#if y_pred is larger than 0.5 it returns true(1) else false(2)
print(y_pred)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy=(1550+175)/2000#Obtained from Confusion Matrix
print(accuracy)