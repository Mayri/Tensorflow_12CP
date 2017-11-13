from __future__ import division, print_function, absolute_import

import timeit
import numpy as np
import tflearn
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from random import randint

#timer 
start = timeit.default_timer()
#Load Dataset
df = pd.read_csv('C:/DeepLearning_lib/NSL/NSL-KDD Processed/KDD_Test_41.csv')
ad = pd.read_csv('C:/DeepLearning_lib/NSL/NSL-KDD Processed/KDD_Train_41.csv')
qw = pd.read_csv('C:/DeepLearning_lib/NSL/NSL-KDD Processed/NSL_TrainLabels_mat5.csv')
er = pd.read_csv('C:/DeepLearning_lib/NSL/NSL-KDD Processed/NSL_TestLabels_mat5.csv')
tr = pd.read_csv('C:/DeepLearning_lib/NSL/NSL-KDD Processed/KDD_Valid_41.csv')
yu = pd.read_csv('C:/DeepLearning_lib/NSL/NSL-KDD Processed/NSL_TestLabels_mat5.csv')
rt = pd.read_csv('C:/DeepLearning_lib/NSL/NSL-KDD Processed/NSL_TrainLabels_int.csv')

a = df.values
b = ad.values
c = qw.values
d = er.values
e = tr.values
f = yu.values
g = rt.values
test_set = np.float32(a)
train_set = np.float32(b)
train_labels_set = np.float32(c)
valid_labels_set = np.float32(d)
valid_set = np.float32(e)
test_labels_set = np.float32(f)
test_set_for_CM =np.float32(g)


f = randint(0,20)

# Placeholders to hold data before feeding it into network
x = tf.placeholder("float",[None, 41]) 	#for images with shape of None,
y = tf.placeholder("float",[None, 5])		#for lables with shape of None,10
#z = tflearn.layers.core.one_hot_encoding(test_labels_set, n_classes = 5, name = 'one_hot_encoded_testlables')

# Building the encoder Network
encoder = tflearn.input_data(shape=[None, 41])
encoder = tflearn.fully_connected(encoder, 41)
encoder = tflearn.fully_connected(encoder, 30)
encoder = tflearn.fully_connected(encoder, 20)
encoder = tflearn.fully_connected(encoder, 10)
encoder = tflearn.fully_connected(encoder, 5, activation='softmax')

#For calculating Accuracy at every step of model training 
acc= tflearn.metrics.Accuracy()

# Regression, with mean square error (learn about it more here http://tflearn.org/layers/estimator/)
net = tflearn.regression(encoder, optimizer='adam', learning_rate=0.001,
                         loss='mean_square', metric=acc, shuffle_batches=True)


# Mpdeling the Neural Network (for details http://tflearn.org/models/dnn/)
model = tflearn.DNN(net, tensorboard_verbose=0)

sess = tf.Session()


# Training the Neural Network (for details http://tflearn.org/models/dnn/)
model.fit(test_set, test_labels_set, n_epoch=20, validation_set=(valid_set, valid_labels_set),
          run_id="auto_encoder", batch_size=1,show_metric=True, snapshot_epoch=False)

# Here I evaluate the model with Test Images and Test Lables, calculating the Mean Accuracy of the model.
evaluation= model.evaluate(train_set,train_labels_set)
print("\n")
print("\t"+"Mean accuracy of the model is :", evaluation)

# Prediction the Lables of the Images that we give to the model just to have a clear picture of Neural Netwok
lables = model.predict_label(train_set)
print("\n")
print("\t"+"The predicted labels are :",lables)

# Predicted probailites 
y = model.predict(train_set)
print("\n")
print("\t"+"\t"+"\t"+"The predicted probabilities are :" )
print("\n")
print (y[f])

# Running a session to feed calculate the confusion matrix
sess = tf.Session()
# taking the argumented maximum of the predicted probabilities for generating confusion matrix 
prediction = tf.argmax(y,1)
# displaying length of predictions and evaluating them in a session 
with sess.as_default():
	print (len(prediction.eval()))
	predicted_labels = prediction.eval()
# Again importing the mnist data with one hot as false because we need to know the truepositive and other values for evaluation
#Images, Lables, testImages, targetLables = mnist.load_data(one_hot=False)

# Used Sklearn library for evaluation as tensorflows library was not documented properly 
# Generated the Confusion Matrix 
confusionMatrix = confusion_matrix(test_set_for_CM, predicted_labels)
print("\n"+"\t"+"The confusion Matrix is ")
print ("\n",confusionMatrix)

# Classification_report in Sklearn provide all the necessary scores needed to succesfully evaluate the model. 
classification = classification_report(test_set_for_CM,predicted_labels, digits=4, 
				target_names =['class 0','class 1','class 2','class 3','class 4'])
print("\n"+"\t"+"The classification report is ")

print ("\n",classification)
stop = timeit.default_timer()
mins = (stop - start)/60
print ("The total time for this algorithm is ",mins,"min")
