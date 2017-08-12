from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import operator
import os
from sys import exit
os.system('clear')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#Load Dataset
test = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Test_41.csv')             # test set 
testLables = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_mat5.csv')     # test labels
train = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Train_41.csv')            # train set 
trainLables = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat5.csv')    # train labels
valid = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Valid_41.csv')            # valid set
validLabels = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_ValidLabels_int3.csv')    # valid labels
trainLabelsOHE = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TrainLabels_int.csv')
testLabelsOHE = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_int.csv')

test_set = np.float32(test)
test_labels_set = np.float32(testLables)
train_set = np.float32(train)
train_labels_set = np.float32(trainLables)
valid_set = np.float32(valid)
valid_labels_set = np.float32(validLabels)

#for classification purpose
test_set_for_CM =np.float32(testLabelsOHE)
train_set_for_CM = np.float32(trainLabelsOHE)

# Reading KDD train classes files for confusion matrics and classification reports
class2_for_train_data = pd.read_csv('/home/jay/Documents/Project/Classes_of_dataset/classes_for_KDD_train_data/class2.csv')
class3_for_train_data = pd.read_csv('/home/jay/Documents/Project/Classes_of_dataset/classes_for_KDD_train_data/class3.csv')
class4_for_train_data = pd.read_csv('/home/jay/Documents/Project/Classes_of_dataset/classes_for_KDD_train_data/class4.csv')

#classes
class2 = np.float32(class2_for_train_data)
class3 = np.float32(class3_for_train_data)
class4 = np.float32(class4_for_train_data)


#Pretraing Parameters
learning_rate = 0.1 #float(input("Please input the learning rate of the model: ")) 
training_epochs = 1 #int(input("Please input the training epochs of the model : "))
batch_size = 100 #int(input("Please input the training batch size of model : "))
display_step = 1


# Pretraining Network Parameters
pre_n_hidden_1 = 30 #int(input("Please input the training network's Hidden layer 1'st Neurons : ")) # 1st layer num features
pre_n_hidden_2 = 20 #int(input("Please input the training network's Hidden layer 2'nd Neurons : "))# 2nd layer num features 
pre_n_hidden_3 = 10 #int(input("Please input the training network's Hidden layer 3'rd Neurons : "))
pre_n_hidden_4 = 5 #int(input("Please input the training network's Hidden layer 4'th Neurons : "))
pre_n_input = 41 
print("\n\n")

# tf Graph input
X = tf.placeholder("float", [None, pre_n_input])
# Placeholder for the Labels data
Y = tf.placeholder("float", [None, 5])

#weights and biases for pretraining with random normal

weights = {
    'encoder_pre_h1': tf.Variable(tf.random_normal([pre_n_input, pre_n_hidden_1])),
    'encoder_pre_h2': tf.Variable(tf.random_normal([pre_n_hidden_1, pre_n_hidden_2])),
    'encoder_pre_h3': tf.Variable(tf.random_normal([pre_n_hidden_2, pre_n_hidden_3])),
    'encoder_pre_h4': tf.Variable(tf.random_normal([pre_n_hidden_3, pre_n_hidden_4])),
    'decoder_pre_h1': tf.Variable(tf.random_normal([pre_n_hidden_1,pre_n_input])),
    'decoder_pre_h2': tf.Variable(tf.random_normal([pre_n_hidden_2, pre_n_hidden_1])),
    'decoder_pre_h3': tf.Variable(tf.random_normal([pre_n_hidden_3, pre_n_hidden_2])),
    'decoder_pre_h4': tf.Variable(tf.random_normal([pre_n_hidden_4, pre_n_hidden_3])),
}
biases = {
    'encoder_pre_b1': tf.Variable(tf.random_normal([pre_n_hidden_1])),
    'encoder_pre_b2': tf.Variable(tf.random_normal([pre_n_hidden_2])),
    'encoder_pre_b3': tf.Variable(tf.random_normal([pre_n_hidden_3])),
    'encoder_pre_b4': tf.Variable(tf.random_normal([pre_n_hidden_4])),
    'decoder_pre_b1': tf.Variable(tf.random_normal([pre_n_input])),
    'decoder_pre_b2': tf.Variable(tf.random_normal([pre_n_hidden_1])),
    'decoder_pre_b3': tf.Variable(tf.random_normal([pre_n_hidden_2])),
    'decoder_pre_b4': tf.Variable(tf.random_normal([pre_n_hidden_3])),
}
#weights and biases for fine tune defined with zeros
post_weights = {
    'encoder_pre_h1': tf.Variable(tf.zeros([pre_n_input, pre_n_hidden_1])),
    'encoder_pre_h2': tf.Variable(tf.zeros([pre_n_hidden_1, pre_n_hidden_2])),
    'encoder_pre_h3': tf.Variable(tf.zeros([pre_n_hidden_2, pre_n_hidden_3])),
    'encoder_pre_h4': tf.Variable(tf.zeros([pre_n_hidden_3, pre_n_hidden_4])),
}
post_biases = {
    'encoder_pre_b1': tf.Variable(tf.zeros([pre_n_hidden_1])),
    'encoder_pre_b2': tf.Variable(tf.zeros([pre_n_hidden_2])),
    'encoder_pre_b3': tf.Variable(tf.zeros([pre_n_hidden_3])),
    'encoder_pre_b4': tf.Variable(tf.zeros([pre_n_hidden_4])),
}

encoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))

#cost and optimizer for layer
decoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_1, weights['decoder_pre_h1']),
                                   biases['decoder_pre_b1']))


cost_layer_1 = tf.reduce_mean(tf.pow(X - decoder_layer_1, 2))
optimizer_layer_1 = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_layer_1)

inti = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(inti)
    total_batch = int(len(train_set)/batch_size)
    print("training encoder layer 1")
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
             _,c = sess.run([optimizer_layer_1,cost_layer_1], feed_dict={X: valid_set})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
    #saving weights and biases for fine tuning
    print("Saving weight and biases for fine tune of layer 1")
    

encoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))

encoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_1, weights['encoder_pre_h2']),
                               biases['encoder_pre_b2']))

decoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_2, weights['decoder_pre_h2']),
                                   biases['decoder_pre_b2']))                               

cost_layer_2 = tf.reduce_mean(tf.pow(encoder_layer_1 - decoder_layer_2, 2))
optimizer_layer_2 = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_layer_2)

inti = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(inti)
    total_batch = int(len(train_set)/batch_size)
    print("training encoder layer 2")    
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
              _,c = sess.run([optimizer_layer_2,cost_layer_2], feed_dict={X: valid_set})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
    #saving weights and biases for fine tuning
    print("Saving weight and biases for fine tune of layer 2")
    

encoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
                                   
encoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_1, weights['encoder_pre_h2']),
                               biases['encoder_pre_b2']))

encoder_layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_2, weights['encoder_pre_h3']),
                               biases['encoder_pre_b3']))

decoder_layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_3, weights['decoder_pre_h3']),
                                   biases['decoder_pre_b3']))                               

cost_layer_3 = tf.reduce_mean(tf.pow(encoder_layer_2 - decoder_layer_3, 2))
optimizer_layer_3 = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_layer_3)

inti = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(inti)
    total_batch = int(len(train_set)/batch_size)
    print("training encoder layer 3")    
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
             _,c = sess.run([optimizer_layer_3,cost_layer_3], feed_dict={X: valid_set})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
    #saving weights and biases for fine tuning
    print("Saving weight and biases for fine tune of layer 3")
    

encoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_pre_h1']),
                                   biases['encoder_pre_b1']))
                                   
encoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_1, weights['encoder_pre_h2']),
                               biases['encoder_pre_b2']))

encoder_layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_2, weights['encoder_pre_h3']),
                               biases['encoder_pre_b3']))
encoder_layer_4 = tf.nn.softmax(tf.add(tf.matmul(encoder_layer_3, weights['encoder_pre_h4']),
                                   biases['encoder_pre_b4']))

decoder_layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_4, weights['decoder_pre_h4']),
                                   biases['decoder_pre_b4']))   

cost_layer_4 = tf.reduce_mean(tf.pow(encoder_layer_3 - decoder_layer_4, 2))
optimizer_layer_4 = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_layer_4)


inti = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(inti)
    total_batch = int(len(train_set)/batch_size)
    print("training encoder layer 4")    
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
             _,c = sess.run([optimizer_layer_4,cost_layer_4], feed_dict={X: valid_set})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
    #saving weights and biases for fine tuning
    print("Saving weight and biases for fine tune of layer 4")
    post_weights['encoder_pre_h1'] = weights['encoder_pre_h1']
    post_biases['encoder_pre_b1'] = biases['encoder_pre_b1']
    post_weights['encoder_pre_h2'] = weights['encoder_pre_h2']
    post_biases['encoder_pre_b2'] =biases['encoder_pre_b2']
    post_weights['encoder_pre_h3'] = weights['encoder_pre_h3']
    post_biases['encoder_pre_b3'] =biases['encoder_pre_b3']
    post_weights['encoder_pre_h4'] = weights['encoder_pre_h4']
    post_biases['encoder_pre_b4'] =biases['encoder_pre_b4']

#building the neural network


def encoder(x):
	# Encoder Hidden layer with relu activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, post_weights['encoder_pre_h1']),
                                   post_biases['encoder_pre_b1']))
    # Decoder Hidden layer with relu activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, post_weights['encoder_pre_h2']),
                                   post_biases['encoder_pre_b2']))

    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, post_weights['encoder_pre_h3']),
                                   post_biases['encoder_pre_b3']))

    layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, post_weights['encoder_pre_h4']),
                                   post_biases['encoder_pre_b4']))
    return layer_4

# Construct model
encoder_post_op = encoder(X)
# Prediction
y_pred = encoder_post_op
# Targets (Labels) are the input data.
y_true = Y

# Define loss and optimizer, minimize the squared error
cost_nn = tf.reduce_mean(tf.pow(y_pred - y_true, 2))

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_nn)
#optimizer = tf.train.MomentumOptimizer(post_learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
predict_op = tf.argmax(y_pred,1)
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(len(test_set)/batch_size)
    print("\nFine-tuning the model...\n")
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost_nn], feed_dict={X: test_set, Y:test_labels_set})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("\nFine-tuning Finished!")
    predicted_labels = sess.run(predict_op, feed_dict ={X:train_set, Y:train_labels_set})

#accuracay function
accuracy = accuracy_score(train_set_for_CM, predicted_labels)
b = 100
printaccuracy = operator.mul(accuracy,b)
print("\nThe Accuracy of the model is :", printaccuracy)

#confusion matrix for all the classes
confusionMatrix = confusion_matrix(train_set_for_CM, predicted_labels)
print("\n"+"\t"+"The confusion Matrix is ")
print ("\n",confusionMatrix)


# Classification_report in Sklearn provide all the necessary scores needed to succesfully evaluate the model. 
classification = classification_report(train_set_for_CM,predicted_labels, digits=4, 
				target_names =['Normal','DoS','Probe','U2R','I2R'])
print("\n"+"\t"+"The classification report is ")

print ("\n",classification)
