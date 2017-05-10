#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import operator
import os
os.system('mode con: cols=100 lines=40')
from sys import exit
os.system('reset')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[31m'
    YELLOW = '\033[93m'
    CYAN ='\033[96m'
    PURPLE ='\033[35m'
    ORANGE ='\033[33m'
    CBLINK = '\33[5m'

## ############################## This is just for the specifing the script on the screen ##################################################################

print(bcolors.HEADER+"\t\t**********************************************************"+bcolors.ENDC)
print("\t\t*                                                        *")
print(bcolors.BOLD+bcolors.GREEN+"\t\t*                  The Autoencoder Program               *"+bcolors.ENDC)
print(bcolors.BOLD+bcolors.GREEN+bcolors.BLUE   +"\t\t*   - A Program by Jay Vala, Akash Antony, Kartik Sareen *"+bcolors.ENDC)
print("\t\t*                                                        *")
print(bcolors.HEADER+"\t\t**********************************************************\n"+bcolors.ENDC)

print(bcolors.BOLD+bcolors.GREEN+"\t\tSelect the appropriate options and you will be good to go!\n"+bcolors.ENDC)

########################################################### Dataset Inputs ################################################################################

print(bcolors.BOLD+bcolors.PURPLE+"\n File Specifications \n"+bcolors.ENDC)
print(bcolors.WARNING+"NOTE \nPlease input the file path in the format C:/Document...,Please use '/' as seperator"+bcolors.ENDC)

test_set_input = str(input("\nTest Set : "))
test_set_labels = str(input("\nTest Set Labels : "))
train_set_input = str(input("\nTrain Set : "))
train_set_labels = str(input("\nTrain Set Labels : "))
valid_set_input = str(input("\nValid Set : "))
valid_set_labels = str(input("\nValid Set Lables : "))

#Load Dataset
df = pd.read_csv(test_set)             # test set 
er = pd.read_csv(test_set_labels)     # test labels
ad = pd.read_csv(train_set)            # train set 
qw = pd.read_csv(train_set_labels)    # train labels
tr = pd.read_csv(valid_set)            # valid set
yu = pd.read_csv(valid_set_labels)    # valid labels
t = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_int.csv')

a = df.values
b = ad.values
c = qw.values
d = er.values
e = tr.values
f = yu.values
h = t.values
test_set = np.float32(a)
train_set = np.float32(b)
train_labels_set = np.float32(c)
valid_labels_set = np.float32(f)
valid_set = np.float32(e)
test_labels_set = np.float32(d)
test_set_for_CM =np.float32(h)

print(bcolors.YELLOW+bcolors.BOLD+"PRE-TRANING PHASE\n"+bcolors.ENDC)

##############################################################Pretraing Parameters #########################################################################

pre_learning_rate = float(input("Please input the Pretraining learning rate : ")) 
pre_training_epochs = int(input("Please input the Pretraining epochs : "))
pre_batch_size = int(input("Please input the Pretraining batch size : "))
display_step = 1


################################################# Pretraining Network Parameters############################################################################

pre_n_hidden_1 = int(input("Please input the Pretraing network's Hidden layer 1'st Neurons : ")) # 1st layer num features
pre_n_hidden_2 = int(input("Please input the Pretraing network's Hidden layer 2'nd Neurons : "))# 2nd layer num features 
pre_n_hidden_3 = int(input("Please input the Pretraing network's Hidden layer 3'rd Neurons : "))
pre_n_hidden_4 = int(input("Please input the Pretraing network's Hidden layer 4's Neurons : "))
pre_n_input = 41 
print("\n")

# tf Graph input
X = tf.placeholder("float", [None, pre_n_input])

######################################## Weights and Biases for Pre-Training layers ########################################################################
weights = {
    'encoder_pre_h1': tf.Variable(tf.random_normal([pre_n_input, pre_n_hidden_1])),
    'encoder_pre_h2': tf.Variable(tf.random_normal([pre_n_hidden_1, pre_n_hidden_2])),
    'encoder_pre_h3': tf.Variable(tf.random_normal([pre_n_hidden_2, pre_n_hidden_3])),
    'encoder_pre_h4': tf.Variable(tf.random_normal([pre_n_hidden_3, pre_n_hidden_4])),
    'decoder_pre_h1': tf.Variable(tf.random_normal([pre_n_hidden_4, pre_n_hidden_3])),
    'decoder_pre_h2': tf.Variable(tf.random_normal([pre_n_hidden_3, pre_n_hidden_2])),
    'decoder_pre_h3': tf.Variable(tf.random_normal([pre_n_hidden_2, pre_n_hidden_1])),
    'decoder_pre_h4': tf.Variable(tf.random_normal([pre_n_hidden_1, pre_n_input])),
}
biases = {
    'encoder_pre_b1': tf.Variable(tf.random_normal([pre_n_hidden_1])),
    'encoder_pre_b2': tf.Variable(tf.random_normal([pre_n_hidden_2])),
    'encoder_pre_b3': tf.Variable(tf.random_normal([pre_n_hidden_3])),
    'encoder_pre_b4': tf.Variable(tf.random_normal([pre_n_hidden_4])),
	  'decoder_pre_b1': tf.Variable(tf.random_normal([pre_n_hidden_3])),
    'decoder_pre_b2': tf.Variable(tf.random_normal([pre_n_hidden_2])),
    'decoder_pre_b3': tf.Variable(tf.random_normal([pre_n_hidden_1])),
    'decoder_pre_b4': tf.Variable(tf.random_normal([pre_n_input])),
}

################################ Activation Function Selection #############################################################################################

print(bcolors.BOLD+bcolors.YELLOW+"Activation Functions for the layers\n"+bcolors.ENDC)
print("1 for relu\n2 for sigmoid\n3 for elu\n4 for softplus\n5 for softmax\n")
activation = int(input("Please enter the activation function for the encoder layers : "))
print("\n")

if activation == 1: # Relu
  def pre_encoder(x):

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_pre_h2']),
                                   biases['encoder_pre_b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_pre_h3']),
                                   biases['encoder_pre_b3']))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_pre_h4']),
                                   biases['encoder_pre_b4']))

    return layer_4

  def pre_decoder(x):

      layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_pre_h1']),
                                     biases['decoder_pre_b1']))
      layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_pre_h2']),
                                     biases['decoder_pre_b2']))
      layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['decoder_pre_h3']),
                                     biases['decoder_pre_b3']))
      layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['decoder_pre_h4']),
                                     biases['decoder_pre_b4']))
      return layer_4

elif activation == 2:# Sigmoid
  def pre_encoder(x):

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_pre_h2']),
                                   biases['encoder_pre_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_pre_h3']),
                                   biases['encoder_pre_b3']))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_pre_h4']),
                                   biases['encoder_pre_b4']))

    return layer_4

  def pre_decoder(x):

      layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_pre_h1']),
                                     biases['decoder_pre_b1']))
      layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_pre_h2']),
                                     biases['decoder_pre_b2']))
      layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_pre_h3']),
                                     biases['decoder_pre_b3']))
      layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['decoder_pre_h4']),
                                     biases['decoder_pre_b4']))
      return layer_4

elif activation == 3: #elu
  def pre_encoder(x):

    layer_1 = tf.nn.elu(tf.add(tf.matmul(x, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
    layer_2 = tf.nn.elu(tf.add(tf.matmul(layer_1, weights['encoder_pre_h2']),
                                   biases['encoder_pre_b2']))
    layer_3 = tf.nn.elu(tf.add(tf.matmul(layer_2, weights['encoder_pre_h3']),
                                   biases['encoder_pre_b3']))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_pre_h4']),
                                   biases['encoder_pre_b4']))

    return layer_4

  def pre_decoder(x):

      layer_1 = tf.nn.elu(tf.add(tf.matmul(x, weights['decoder_pre_h1']),
                                     biases['decoder_pre_b1']))
      layer_2 = tf.nn.elu(tf.add(tf.matmul(layer_1, weights['decoder_pre_h2']),
                                     biases['decoder_pre_b2']))
      layer_3 = tf.nn.elu(tf.add(tf.matmul(layer_2, weights['decoder_pre_h3']),
                                     biases['decoder_pre_b3']))
      layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['decoder_pre_h4']),
                                     biases['decoder_pre_b4']))
      return layer_4

elif activation == 4: #Softplus
  
  def pre_encoder(x):
    
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
    layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, weights['encoder_pre_h2']),
                                   biases['encoder_pre_b2']))
    layer_3 = tf.nn.softplus(tf.add(tf.matmul(layer_2, weights['encoder_pre_h3']),
                                   biases['encoder_pre_b3']))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_pre_h4']),
                                   biases['encoder_pre_b4']))

    return layer_4

  def pre_decoder(x):
      
      layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['decoder_pre_h1']),
                                     biases['decoder_pre_b1']))
      layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, weights['decoder_pre_h2']),
                                     biases['decoder_pre_b2']))
      layer_3 = tf.nn.softplus(tf.add(tf.matmul(layer_2, weights['decoder_pre_h3']),
                                     biases['decoder_pre_b3']))
      layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['decoder_pre_h4']),
                                     biases['decoder_pre_b4']))
      return layer_4

elif activation == 5: #Softmax
  
  def pre_encoder(x):
   
    layer_1 = tf.nn.softmax(tf.add(tf.matmul(x, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
    layer_2 = tf.nn.softmax(tf.add(tf.matmul(layer_1, weights['encoder_pre_h2']),
                                   biases['encoder_pre_b2']))
    layer_3 = tf.nn.softmax(tf.add(tf.matmul(layer_2, weights['encoder_pre_h3']),
                                   biases['encoder_pre_b3']))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_pre_h4']),
                                   biases['encoder_pre_b4']))

    return layer_4

  def pre_decoder(x):
    
    layer_1 = tf.nn.softmax(tf.add(tf.matmul(x, weights['decoder_pre_h1']),
                                     biases['decoder_pre_b1']))
    layer_2 = tf.nn.softmax(tf.add(tf.matmul(layer_1, weights['decoder_pre_h2']),
                                     biases['decoder_pre_b2']))
    layer_3 = tf.nn.softmax(tf.add(tf.matmul(layer_2, weights['decoder_pre_h3']),
                                     biases['decoder_pre_b3']))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['decoder_pre_h4']),
                                     biases['decoder_pre_b4']))
    return layer_4
else:
  print(bcolors.FAIL+"Wrong Option selected, exiting..."+bcolors.ENDC)
  exit(0)
  
# Construct model
encoder_pre_op = pre_encoder(X)
decoder_pre_op = pre_decoder(encoder_pre_op)

# Prediction
y_pred = decoder_pre_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
print(bcolors.BOLD+bcolors.YELLOW+"\nOptimizer Selection for Pre-training"+bcolors.ENDC)
print("\n1 for Adam Optimizer,\n2 for RMSprop,\n3 for Gradient Descent Optimizer,\n4 for Momentum Optimizer,\n")
command = int(input("Please input the optimizer selection :"))

if command == 1:
    optimizer = tf.train.AdamOptimizer(pre_learning_rate).minimize(cost)
elif command == 2:
    optimizer = tf.train.RMSpropOptimizer(pre_learning_rate).minimize(cost)
elif command == 3:
    optimizer = tf.train.GradientDescentOptimizer(pre_learning_rate).minimize(cost)
elif command == 4:    
    optimizer = tf.train.MomentumOptimizer(pre_learning_rate,momentum).minimize(cost)
else:
   print(bcolors.FAIL+"You have entered a wrong option, exiting...\n"+bcolors.ENDC)
   exit(0)

#optimizer = tf.train.AdamOptimizer(pre_learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(len(test_set)/pre_batch_size)
    print(bcolors.BOLD+bcolors.ORANGE+"\n\tPRE-TANING MODEL...\n"+bcolors.ENDC)
    # Training cycle
    for epoch in range(pre_training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: valid_set})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

            print(bcolors.BOLD+bcolors.CBLINK+"Pre Training Complete..."+bcolors.ENDC)
    
#Saving the weights and biases to be used in the fine tuning of the model

weights_encoder_post_1 = weights['encoder_pre_h1']
weights_encoder_post_2 = weights['encoder_pre_h2']
weights_encoder_post_3 = weights['encoder_pre_h3']
weights_encoder_post_4 = weights['encoder_pre_h4']
    
bias_encoder_post_1 = biases['encoder_pre_b1']
bias_encoder_post_2 = biases['encoder_pre_b2']
bias_encoder_post_3 = biases['encoder_pre_b3']
bias_encoder_post_4 = biases['encoder_pre_b4']

print(bcolors.YELLOW+bcolors.BOLD+"Fine Tuning Phase"+bcolors.ENDC)
#Finetuning Parameters
post_learning_rate = float(input("Please input the Fintuning learning rate : ")) 
post_training_epochs = int(input("Please input the Finetuning epochs : "))
post_batch_size = int(input("Please input the Finetuing batch size : \n"))
display_step = 1

print(bcolors.BOLD+bcolors.YELLOW+"Network Parameters For Fine Tuning Process"+bcolors.ENDC)
#Finetuning Network Parameters
post_n_hidden_1 = int(input("Please input the Finetuning network's Hidden layer 1'st Neurons : ")) # 1st layer num features
post_n_hidden_2 = int(input("Please input the Finetuning network's Hidden layer 2'nd Neurons : "))# 2nd layer num features 
post_n_hidden_3 = int(input("Please input the Finetuning network's Hidden layer 3'rd Neurons : "))
post_n_hidden_4 = int(input("Please input the Finetuning network's Hidden layer 4'th Neurons : "))
post_n_input = 41
# Placeholder for the Labels data
Y = tf.placeholder("float", [None,5])

print(bcolors.WARNING+"\n\t For making this program more exprimental there is also an option to change the activation after the network is pretraining\n"+bcolors.ENDC)
print(bcolors.WARNING+"\t You can change the activation or you can select the same as you have selected above!!\n"+bcolors.ENDC)
print(bcolors.BOLD+bcolors.YELLOW+"Activation Function For Fine Tuning"+bcolors.ENDC)
print("1 for relu\n2 for sigmoid\n3 for elu\n4 for softplus\n5 for softmax")

activation = int(input("Please enter the activation function for the encoder layers : "))
if activation == 1:
  def pre_encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_pre_h2']),
                                   biases['encoder_pre_b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_pre_h3']),
                                   biases['encoder_pre_b3']))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_pre_h4']),
                                   biases['encoder_pre_b4']))

    return layer_4


elif activation == 2:
  def pre_encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_pre_h2']),
                                   biases['encoder_pre_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_pre_h3']),
                                   biases['encoder_pre_b3']))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_pre_h4']),
                                   biases['encoder_pre_b4']))

    return layer_4

 

elif activation == 3: #Sigmoid
  def pre_encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.elu(tf.add(tf.matmul(x, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.elu(tf.add(tf.matmul(layer_1, weights['encoder_pre_h2']),
                                   biases['encoder_pre_b2']))
    layer_3 = tf.nn.elu(tf.add(tf.matmul(layer_2, weights['encoder_pre_h3']),
                                   biases['encoder_pre_b3']))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_pre_h4']),
                                   biases['encoder_pre_b4']))

    return layer_4

  d
elif activation == 4: #Softplus
  
  def pre_encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, weights['encoder_pre_h2']),
                                   biases['encoder_pre_b2']))
    layer_3 = tf.nn.softplus(tf.add(tf.matmul(layer_2, weights['encoder_pre_h3']),
                                   biases['encoder_pre_b3']))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_pre_h4']),
                                   biases['encoder_pre_b4']))

    return layer_4

  

elif activation == 5: #Softmax
  
  def pre_encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.softmax(tf.add(tf.matmul(x, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.softmax(tf.add(tf.matmul(layer_1, weights['encoder_pre_h2']),
                                   biases['encoder_pre_b2']))
    layer_3 = tf.nn.softmax(tf.add(tf.matmul(layer_2, weights['encoder_pre_h3']),
                                   biases['encoder_pre_b3']))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_pre_h4']),
                                   biases['encoder_pre_b4']))

    return layer_4
else:
  print(bcolors.FAIL+"Wrong Option selected, exiting..."+bcolors.ENDC)
  exit(0)

# Construct model
encoder_post_op = pre_encoder(X)
# Prediction
y_pred = encoder_post_op
# Targets (Labels) are the input data.
y_true = train_labels_set

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
################################# Optimizer Selection Function ############################################################################

print(bcolors.BOLD+bcolors.YELLOW+"\nOptimizer Selection for Fine Tuining"+bcolors.ENDC)
print("\n1 for Adam Optimizer,\n2 for RMSprop,\n3 for Gradient Descent Optimizer,\n4 for Momentum Optimizer,\n")
command = int(input("Please input the optimizer selection :"))

if command == 1:
    optimizer = tf.train.AdamOptimizer(pre_learning_rate).minimize(cost)
elif command == 2:
    optimizer = tf.train.RMSpropOptimizer(pre_learning_rate).minimize(cost)
elif command == 3:
    momnetum = float(input("\nPlease enter the value of momentum (only floating point values are allowed): "))
    optimizer = tf.train.MomentumOptimizer(pre_learning_rate,momentum).minimize(cost)
else:
   print(bcolors.FAIL+"You have entered a wrong option, exiting...\n"+bcolors.ENDC)
   exit(0)

############################################################################################################################################

# Initializing the variables
init = tf.global_variables_initializer()
predict_op = tf.argmax(y_pred,1)
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(len(test_set)/post_batch_size)
    print(bcolors.BOLD+bcolors.ORANGE+"\nFINE-TUNING MODEL...\n"+bcolors.ENDC)
    # Training cycle
    for epoch in range(post_training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: train_set,Y: train_labels_set})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print(bcolors.BOLD+bcolors.GREEN+"\t\tFine Tuning Finised..."+bcolors.ENDC)


    predicted_labels = sess.run(predict_op, feed_dict ={X:test_set, Y:test_labels_set})

#####################################################The Accuracy Function ###################################################################

accuracy = accuracy_score(test_set_for_CM, predicted_labels)
b = 100
printaccuracy = operator.mul(accuracy,b)
print("\nThe Accuracy of the model is :"+ bcolors.BOLD+bcolors.RED+printaccuracy+bcolors.ENDC)

##############################################################################################################################################


