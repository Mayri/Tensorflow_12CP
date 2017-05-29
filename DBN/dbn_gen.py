#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

Import the math function for calculations
import math
#Tensorflow library. Used to implement machine learning models
import tensorflow as tf
#Numpy contains helpful functions for efficient mathematical calculations
import numpy as np
#Pandas contains functions needed to read csv files
import pandas as pd
#Sklearn library has some good evaluation functions 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#Operator contains basic math functions 
import operator
#Os contains Operating systemms calls which we use here to hide some unwanted messages from Tensorflow
import os
#Timeit contains function to calculate time
import timeit

#Please don`t change
os.system('mode con: cols=100 lines=40')
os.system('reset')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' 


##################################################################################################################################################
print("\t\t**********************************************************")
print("\t\t*                                                        *")
print("\t\t*                  The Deep Belief Network Program       *")
print("\t\t*   - A Program by Jay Vala, Akash Antony, Kartik Sareen *")
print("\t\t*                                                        *")
print("\t\t**********************************************************\n")

print("\t\tSelect the appropriate options and you will be good to go!\n")
##################################################################################################################################################

#Timer function to calculate the time taken to run the program
start = timeit.default_timer()

#Class that defines the behavior of the RBM
class RBM(object):
    
    def __init__(self, input_size, output_size):
        #Defining the hyperparameters
        self._input_size = input_size #Size of input
        self._output_size = output_size #Size of output
        self.epochs = rbm_epoch #Amount of training iterations
        self.learning_rate = lr_rate_rbm #The step used in gradient descent
        self.batchsize = rbm_epoch #The size of how much data will be used for training per sub iteration
        
        #Initializing weights and biases as matrices full of zeroes
        self.w = np.zeros([input_size, output_size], np.float32) #Creates and initializes the weights with 0
        self.hb = np.zeros([output_size], np.float32) #Creates and initializes the hidden biases with 0
        self.vb = np.zeros([input_size], np.float32) #Creates and initializes the visible biases with 0


    #Fits the result from the weighted visible layer plus the bias into a sigmoid curve
    def prob_h_given_v(self, visible, w, hb):
        if activation == 1:
            return tf.nn.relu(tf.matmul(visible, w) + hb)
        elif activation == 2:
            return tf.nn.sigmoid(tf.matmul(visible, w) + hb)
        elif activation == 3:
            return tf.nn.elu(tf.matmul(visible, w) + hb)
        elif activation == 4:
            return tf.nn.softplus(tf.matmul(visible, w) + hb)
        elif activation == 5:
            return tf.nn.softmax(tf.matmul(visible, w) + hb)
        else:
            print("Wrong option, exiting...")
            exit(0)

    #Fits the result from the weighted hidden layer plus the bias into a sigmoid curve
    def prob_v_given_h(self, hidden, w, vb):
        if activation == 1:
            return tf.nn.relu(tf.matmul(hidden, tf.transpose(w)) + vb)
        elif activation == 2:
            return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)
        elif activation == 3:
            return tf.nn.elu(tf.matmul(hidden, tf.transpose(w)) + vb)
        elif activation == 4:
            return tf.nn.softplus(tf.matmul(hidden, tf.transpose(w)) + vb)
        elif activation == 5:
            return tf.nn.softmax(tf.matmul(hidden, tf.transpose(w)) + vb)
        else:
            print("Wrong option, exiting...")
            exit(0)
    
    #Generate the sample probability
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    #Training method for the model
    def train(self, X):
        #Create the placeholders for our parameters
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])
        
        prv_w = np.zeros([self._input_size, self._output_size], np.float32) #Creates and initializes the weights with 0
        prv_hb = np.zeros([self._output_size], np.float32) #Creates and initializes the hidden biases with 0
        prv_vb = np.zeros([self._input_size], np.float32) #Creates and initializes the visible biases with 0

        
        cur_w = np.zeros([self._input_size, self._output_size], np.float32)
        cur_hb = np.zeros([self._output_size], np.float32)
        cur_vb = np.zeros([self._input_size], np.float32)
        v0 = tf.placeholder("float", [None, self._input_size])
        
        #Initialize with sample probabilities
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)
        
        #Create the Gradients
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)
        
        #Update learning rates for the layers
        update_w = _w + self.learning_rate *(positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vb = _vb +  self.learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb +  self.learning_rate * tf.reduce_mean(h0 - h1, 0)
        
        #Find the error rate
        err = tf.reduce_mean(tf.square(v0 - v1))
        
        #Training loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #For each epoch
            for epoch in range(self.epochs):
                #For each step/batch
                for start, end in zip(range(0, len(X), self.batchsize),range(self.batchsize,len(X), self.batchsize)):
                    batch = X[start:end]
                    #Update the rates
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error=sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                print ('Epoch: %d' % epoch,'error: %f' % error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb

    #Create expected output for our DBN
    def rbm_outpt(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)


#Loading the Dataset
df = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Test_41.csv')             # test set 
er = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_mat5.csv')     # test labels
ad = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Train_41.csv')            # train set 
qw = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat5.csv')    # train labels
tr = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Valid_41.csv')            # valid set
yu = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_ValidLabels_int3.csv')    # valid labels
rt = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TrainLabels_int.csv')
t = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_int.csv')

# Reading classes files for confusion matrics and classification reports
class2_for_test_data = pd.read_csv('/home/jay/Documents/Project/Classes_of_dataset/classes_for_KDD_test_data/class2.csv')
class3_for_test_data = pd.read_csv('/home/jay/Documents/Project/Classes_of_dataset/classes_for_KDD_test_data/class3.csv')
class4_for_test_data = pd.read_csv('/home/jay/Documents/Project/Classes_of_dataset/classes_for_KDD_test_data/class4.csv')

#taking the values from files
a = df.values
b = ad.values
c = qw.values
d = er.values
e = tr.values
f = yu.values
h = t.values


#Taking the values from csv files for classes datafiles
i = class2_for_test_data.values
j = class3_for_test_data.values
k = class4_for_test_data.values

#converting the values from files and converting them into numpy float types for our model
test_set = np.float32(a)
train_set = np.float32(b)
train_labels_set = np.float32(c)
valid_labels_set = np.float32(f)
valid_set = np.float32(e)
test_labels_set = np.float32(d)
test_set_for_CM =np.float32(h)

#Converting the class values into numpy float values
class2 = np.float32(i)
class3 = np.float32(j)
class4 = np.float32(k)

# Declaring the variabels and asking for user input (We have increased the number of layers for more accuracy)
print("Note : The best configration we found for this model was 300 -> 200 -> 100 -> 50 -> 30 -> 20 -> 10 -> 5. However you can use your own configrations as well.\n\n")
pre_n_hidden_1 = int(input("Please input the Pretraing network's Hidden layer 1'st Neurons : ")) # 1st layer num features
pre_n_hidden_2 = int(input("Please input the Pretraing network's Hidden layer 2'nd Neurons : ")) # 2nd layer num features 
pre_n_hidden_3 = int(input("Please input the Pretraing network's Hidden layer 3'rd Neurons : ")) # 3rd layer num features
pre_n_hidden_4 = int(input("Please input the Pretraing network's Hidden layer 4'th Neurons : ")) # 4th layer num features
pre_n_hidden_5 = int(input("Please input the Pretraing network's Hidden layer 5'th Neurons : ")) # 5th layer num features
pre_n_hidden_6 = int(input("Please input the Pretraing network's Hidden layer 6'th Neurons : ")) # 6th layer num features 
pre_n_hidden_7 = int(input("Please input the Pretraing network's Hidden layer 7'th Neurons : ")) # 7th layer num features
pre_n_hidden_8 = int(input("Please input the Pretraing network's Hidden layer 8'th Neurons : ")) # 8th layer num features


lr_rate_rbm = float(input("\nPlease input the learing rate for training RBM's(should be between 0 and 1) :"))
rbm_epoch = int(input("Please input the number of epochs for training RBM's(more >> better) : "))
rbm_batch =  int(input("Please input the batch size for training the RBM's(lower >> better) :"))
print("\nActivation Functions for the layers\n")
print("1 for relu\n2 for sigmoid\n3 for elu\n4 for softplus\n5 for softmax\n")
activation = int(input("Please enter the activation function for the encoder layers : "))
print("\n")



RBM_hidden_sizes = [pre_n_hidden_1, pre_n_hidden_2 ,pre_n_hidden_3, pre_n_hidden_4, pre_n_hidden_5, pre_n_hidden_6,pre_n_hidden_7, pre_n_hidden_8] #create layers of the RBM's 

#Since we are training, set input as training data
inpX = train_set

#Create list to hold our RBMs
rbm_list = []

#Size of inputs is the number of inputs in the training set
input_size = inpX.shape[1]

#For each RBM we want to generate
for i, size in enumerate(RBM_hidden_sizes):
    print ('RBM stack: ',i,' ',input_size,'->', size)
    rbm_list.append(RBM(input_size, size))
    input_size = size

#For each RBM in our list
for rbm in rbm_list:
    print ('New RBM stack:')
    #Train a new one
    rbm.train(inpX) 
    #Return the output layer
    inpX = rbm.rbm_outpt(inpX)

#Neural Network Parameter
lr_rate_nn = float(input("\nPlease input the learing rate for training Neural Networks's(should be between 0 and 1): "))
nn_epoch = int(input("Please input the number of epochs for training Neural Networks's(more >> better): "))
nn_batch =  int(input("Please input the batch size for training the Neural Networks's(lower >> better): "))


class NN(object):
    
    def __init__(self, sizes, X, Y):
        #Initialize hyperparameters
        self._sizes = sizes
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        self._learning_rate =  lr_rate_nn
        self._momentum = 0.0
        self._epoches = nn_epoch
        self._batchsize = nn_batch
        input_size = X.shape[1]
        
        #initialization loop
        for size in self._sizes + [Y.shape[1]]:
            #Define upper limit for the uniform distribution range
            max_range = 4 * math.sqrt(6. / (input_size + size))
            
            #Initialize weights through a random uniform distribution
            self.w_list.append(
                np.random.uniform( -max_range, max_range, [input_size, size]).astype(np.float32))
            
            #Initialize bias as zeroes
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size
      
    #load data from rbm
    def load_from_rbms(self, dbn_sizes,rbm_list):
        #Check if expected sizes are correct
        assert len(dbn_sizes) == len(self._sizes)
        
        for i in range(len(self._sizes)):
            #Check if for each RBN the expected sizes are correct
            assert dbn_sizes[i] == self._sizes[i]
        
        #If everything is correct, bring over the weights and biases
        for i in range(len(self._sizes)):
            self.w_list[i] = rbm_list[i].w
            self.b_list[i] = rbm_list[i].hb

    #Training method
    def train(self):
        #Create placeholders for input, weights, biases, output
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y = tf.placeholder("float", [None, self._Y.shape[1]])
        
        #Define variables and activation functoin
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])
        
        #Define the cost function
        cost = tf.reduce_mean(tf.square(_a[-1] - y))
        
        #Define the training operation (Momentum Optimizer minimizing the Cost function)
        train_op = tf.train.MomentumOptimizer(
            self._learning_rate, self._momentum).minimize(cost)
        
        #Prediction operation
        predict_op = tf.argmax(_a[-1], 1)
        
        #Training Loop
        with tf.Session() as sess:
            #Initialize Variables
            sess.run(tf.global_variables_initializer())
            
            #For each epoch
            for i in range(self._epoches):
                
                #For each step
                for start, end in zip(
                    range(0, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):
                    
                    #Run the training operation on the input data
                    sess.run(train_op, feed_dict={
                        _a[0]: self._X[start:end], y: self._Y[start:end]})
                
                for j in range(len(self._sizes) + 1):
                    #Retrieve weights and biases
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])
                
                print ("Accuracy rating for epoch " + str(i) + ": " + str(np.mean(np.argmax(self._Y, axis=1) ==
                              sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y}))))
            predicted_labels = sess.run(predict_op,feed_dict={_a[0]:test_set,y:test_labels_set})
            accuracy = accuracy_score(test_set_for_CM,predicted_labels)
            print("Accuracy of the model is :",accuracy)

            #creating confusion matrix for 5 classes 
            confusion_class5 = confusion_matrix(test_set_for_CM, predicted_labels)
            print("\nconfusion matrix for 5 classes\n",confusion_class5)
            #creating confusion matrix for 2 classes 
            confusion_class2 = confusion_matrix(class2_for_test_data, predicted_labels)
            print("confusion matrix for 2 classes\n",confusion_class2)
            #creating confusion matrix for 3 classes 
            confusion_class3 = confusion_matrix(class3_for_test_data, predicted_labels)
            print("confusion matrix for 3 classes\n",confusion_class3)
            #creating confusion matrix for 4 classes 
            confusion_class4 = confusion_matrix(class4_for_test_data, predicted_labels)
            print("confusion matrix for 4 classes\n",confusion_class4)
            #Classification Report for class 5
            classification_class_5 = classification_report(test_set_for_CM,predicted_labels, digits=4, target_names =['Normal','DoS','Probe','U2R','R2I'])
            print("The classification report for all the 5 classes "+"\n")
            print ("\t",classification_class_5)
            #Classification Report for class 2
            classification_class_2 = classification_report(class2_for_test_data,predicted_labels, digits=4, target_names =['Normal','Attack'])
            print("The classification report for the 2 classes "+"\n")
            print ("\t",classification_class_2)
            #Classification Report for class 3
            classification_class_3 = classification_report(class3_for_test_data,predicted_labels, digits=4, 
            	target_names =['Normal','DoS','OtherAttack'])
            print("The classification report for all the 3 classes "+"\n")
            print ("\t",classification_class_3)
            #Classification Report for class 4
            classification_class_4 = classification_report(class4_for_test_data,predicted_labels, digits=4, 
            	target_names =['Normal','DoS','Probe','OtherAttack'])
            print("The classification report for all the 4 classes "+"\n")
            print ("\t",classification_class_4)

nNet = NN(RBM_hidden_sizes, train_set, train_labels_set)
nNet.load_from_rbms(RBM_hidden_sizes,rbm_list)
nNet.train()
