import timeit
import os
import math
#Tensorflow library. Used to implement machine learning models
import tensorflow as tf
#Numpy contains helpful functions for efficient mathematical calculations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
import operator

#Please don`t change
os.system('mode con: cols=100 lines=40')
os.system('reset')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#timer 
start = timeit.default_timer()
class RBM(object):
    
    def __init__(self, input_size, output_size):
        #Defining the hyperparameters
        self._input_size = input_size #Size of input
        self._output_size = output_size #Size of output
        self.epochs = 10 #Amount of training iterations
        self.learning_rate = 0.1 #The step used in gradient descent
        self.batchsize = 10 #The size of how much data will be used for training per sub iteration
        
        #Initializing weights and biases as matrices full of zeroes
        self.w = np.zeros([input_size, output_size], np.float32) #Creates and initializes the weights with 0
        self.hb = np.zeros([output_size], np.float32) #Creates and initializes the hidden biases with 0
        self.vb = np.zeros([input_size], np.float32) #Creates and initializes the visible biases with 0


    #Fits the result from the weighted visible layer plus the bias into a sigmoid curve
    def prob_h_given_v(self, visible, w, hb):
        #Sigmoid 
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    #Fits the result from the weighted hidden layer plus the bias into a sigmoid curve
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)
    
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

# ## Creating the Deep Belief Network

# With the RBM class created and MNIST Datasets loaded in, we can start creating the DBN. For our example, we are going to use a 3 RBMs, one with 500 hidden units, the second one with 200 and the last one with 50. We are generating a **deep hierarchical representation of the training data**. The cell below accomplishes this:

# In[ ]:

RBM_hidden_sizes = [30,20,10,5 ] #create 2 layers of RBM with size 400 and 100

#Since we are training, set input as training data
inpX = train_set

#Create list to hold our RBMs
rbm_list = []

#Size of inputs is the number of inputs in the training set
input_size = inpX.shape[1]
print("...building model")
#For each RBM we want to generate
for i, size in enumerate(RBM_hidden_sizes):
    print ('RBM: ',i,' ',input_size,'->', size)
    rbm_list.append(RBM(input_size, size))
    input_size = size


# ## RBM Train

# We will now begin the pre-training step and train each of the RBMs in our stack by individiually calling the train function, getting the current RBMs output and using it as the next RBM's input.

print("...pretraining model")
#For each RBM in our list
for rbm in rbm_list:
    print ('New RBM:')
    #Train a new one
    rbm.train(inpX) 
    #Return the output layer
    inpX = rbm.rbm_outpt(inpX)




class NN(object):
    
    def __init__(self, sizes, X, Y):
        #Initialize hyperparameters
        self._sizes = sizes
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        self._learning_rate =  0.1
        #self._momentum = 0.0
        self._epoches = 5
        self._batchsize = 10
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
        j = tf.placeholder("float", [None,1]) #test labels set
        k = tf.placeholder("float", [None,0])  # placeholder for predicted labels
        #Define variables and activation functoin
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.softmax(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])
            
        #Define the cost function
        cost = tf.reduce_mean(tf.square(_a[-1] - y))
        
        #Define the training operation (Momentum Optimizer minimizing the Cost function)
        train_op = tf.train.GradientDescentOptimizer(
            self._learning_rate).minimize(cost)
        
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
                    
                    #Run the training operation on the input data i.e the train data
                    sess.run(train_op, feed_dict={
                        _a[0]: self._X[start:end], y: self._Y[start:end]})
                
                for j in range(len(self._sizes) + 1):
                    #Retrieve weights and biases
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])
                
                print ("Accuracy rating for epoch " + str(i) + ": " + str(np.mean(np.argmax(self._Y, axis=1) ==
                              sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y})))
            #Predicted lables on the test data
            predicted_labels = sess.run(predict_op, feed_dict ={_a[0]:test_set, y:test_labels_set})

            #Calculating the means accuracy of the model on the test data 
            accuracy_mode = accuracy_score(test_set_for_CM, predicted_labels)
            accuracy = operator.mul(accuracy_mode,100)
            print("Accuracy of the models is: ", accuracy,"%")
            #warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            #creating confusion matrix for 5 classes 
            confusion_class5 = confusion_matrix(test_set_for_CM, predicted_labels)
            print("confusion matrix for 5 classes\n",confusion_class5)
            #creating confusion matrix for 2 classes 
            confusion_class2 = confusion_matrix(class2, predicted_labels)
            print("confusion matrix for 2 classes\n",confusion_class2)
            #creating confusion matrix for 3 classes 
            confusion_class3 = confusion_matrix(class3, predicted_labels)
            print("confusion matrix for 3 classes\n",confusion_class3)
            #creating confusion matrix for 4 classes 
            confusion_class4 = confusion_matrix(class4, predicted_labels)
            print("confusion matrix for 4 classes\n",confusion_class4)
            #Classification Report for class 5
            classification_class_5 = classification_report(test_set_for_CM,predicted_labels, digits=4, 
				target_names =['Normal','DoS','Probe','U2R','R2I'])
            print("The classification report for all the 5 classes "+"\n")
            print ("\t",classification_class_5)
            #Classification Report for class 2
            classification_class_2 = classification_report(class2,predicted_labels, digits=4, 
				target_names =['Normal','Attack'])
            print("The classification report for the 2 classes "+"\n")
            print ("\t",classification_class_2)
            #Classification Report for class 3
            classification_class_3 = classification_report(class3,predicted_labels, digits=4, 
				target_names =['Normal','DoS','OtherAttack'])
            print("The classification report for all the 3 classes "+"\n")
            print ("\t",classification_class_3)
            #Classification Report for class 4
            classification_class_4 = classification_report(class4,predicted_labels, digits=4, 
				target_names =['Normal','DoS','Probe','OtherAttack'])
            print("The classification report for all the 4 classes "+"\n")
            print ("\t",classification_class_4)
                
nNet = NN(RBM_hidden_sizes, train_set, train_labels_set)
nNet.load_from_rbms(RBM_hidden_sizes,rbm_list)
nNet.train()
stop = timeit.default_timer()
time = ((stop-start)/60)
print ("The total time for this algorithm is ",time,"min")
