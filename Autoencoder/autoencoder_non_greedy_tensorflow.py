from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import operator
import os
from sys import exit
os.system('clear')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#Load Dataset
df = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Test_41.csv')             # test set 
er = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_mat5.csv')     # test labels
ad = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Train_41.csv')            # train set 
qw = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat5.csv')    # train labels
tr = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Valid_41.csv')            # valid set
yu = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_ValidLabels_int3.csv')    # valid labels
rt = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TrainLabels_int.csv')
t = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_int.csv')

a = df.values
b = ad.values
c = qw.values
d = er.values
e = tr.values
f = yu.values
g = rt.values
h = t.values
test_set = np.float32(a)
train_set = np.float32(b)
train_labels_set = np.float32(c)
valid_labels_set = np.float32(f)
valid_set = np.float32(e)
test_labels_set = np.float32(d)
test_set_for_CM =np.float32(h)

#Pretraing Parameters
pre_learning_rate = float(input("Please input the Pretraining learning rate : ")) 
pre_training_epochs = int(input("Please input the Pretraining epochs : "))
pre_batch_size = int(input("Please input the Pretraining batch size : "))
display_step = 1


# Pretraining Network Parameters
pre_n_hidden_1 = int(input("Please input the Pretraing network's Hidden layer 1'st Neurons : ")) # 1st layer num features
pre_n_hidden_2 = int(input("Please input the Pretraing network's Hidden layer 2'nd Neurons : "))# 2nd layer num features 
pre_n_hidden_3 = int(input("Please input the Pretraing network's Hidden layer 3'rd Neurons : "))
pre_n_hidden_4 = int(input("Please input the Pretraing network's Hidden layer 4's Neurons : "))
pre_n_input = 41 
print("\n\n")

# tf Graph input
X = tf.placeholder("float", [None, pre_n_input])

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
# Building the encoder
def encoder(x):
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


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_pre_h1']),
                                   biases['decoder_pre_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_pre_h2']),
                                   biases['decoder_pre_b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['decoder_pre_h3']),
                                   biases['decoder_pre_b3']))
    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['decoder_pre_h4']),
                                   biases['decoder_pre_b4']))
    return layer_4

# Construct model
encoder_pre_op = encoder(X)
decoder_pre_op = decoder(encoder_pre_op)

# Prediction
y_pred = decoder_pre_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

command = int(input("1 for Adam Optimizer,\n2 for RMSprop,\n3 for Gradient Descent Optimizer,\n4 for Momentum Optimizer,\n"))

if command == 1:
    optimizer = tf.train.AdamOptimizer(pre_learning_rate).minimize(cost)
elif command == 2:
    optimizer = tf.train.RMSpropOptimizer(pre_learning_rate).minimize(cost)
elif command == 3:
    momnetum = float(input("\nPlease enter the value of momentum (only floating point values are allowed): "))
    optimizer = tf.train.MomentumOptimizer(pre_learning_rate,momentum).minimize(cost)
else:
   print("You have entered a wrong option, exiting...\n")
   exit(0)

#optimizer = tf.train.AdamOptimizer(pre_learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(len(test_set)/pre_batch_size)
    print("Pretraing the model...\n")
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
    print("\nFine-tuning Finished!")
    predicted_labels = sess.run(predict_op, feed_dict ={X:test_set, Y:test_labels_set})

accuracy = accuracy_score(test_set_for_CM, predicted_labels)
b = 100
printaccuracy = operator.mul(accuracy,b)
print("\nThe Accuracy of the model is :", printaccuracy)
 Generated the Confusion Matrix 
confusionMatrix = confusion_matrix(test_set_for_CM, predicted_labels)
print("\n"+"\t"+"The confusion Matrix is ")
print ("\n",confusionMatrix)

# Classification_report in Sklearn provide all the necessary scores needed to succesfully evaluate the model. 
classification = classification_report(test_set_for_CM,predicted_labels, digits=4, target_names =['class 0','class 1','class 2','class 3','class 4'])
print("\n"+"\t"+"The classification report is ")

print ("\n",classification)
