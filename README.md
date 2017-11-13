# Tensorflow Models

This repository contains two tensorflow models namely Stacked Autoencoders and Deep Belief Network.

There are various variants of both networks.


# Usage
The links to the dataset are here.

Download all the files

[KDD_Test](https://github.com/jdvala/incomplete_project/blob/master/autoencoder/Kdd_Test_41.csv)

[KDD_Train](https://github.com/jdvala/incomplete_project/blob/master/autoencoder/Kdd_Train_41.csv)

[KDD_Valid](https://github.com/jdvala/incomplete_project/blob/master/autoencoder/Kdd_Valid_41.csv)

[KDD_TestLabels](https://github.com/jdvala/incomplete_project/blob/master/autoencoder/NSL_TestLabels_mat5.csv)

[KDD_TrainLabels](https://github.com/jdvala/incomplete_project/blob/master/autoencoder/NSL_TrainLabels_mat5.csv)

[KDD_ValidLabels](https://github.com/jdvala/incomplete_project/blob/master/autoencoder/NSL_ValidLabels_int2.csv)

Clone this repository and open the desired algorithm in any text editor to access settings 

Change the path of the files

```python 
df = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Test_41.csv')             # test set 
er = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_mat5.csv')     # test labels
ad = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Train_41.csv')            # train set 
qw = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TrainLabels_mat5.csv')    # train labels
tr = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/Kdd_Valid_41.csv')            # valid set
yu = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_ValidLabels_int3.csv')    # valid labels
rt = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TrainLabels_int.csv')
t = pd.read_csv('/home/jay/Documents/Project/incomplete_project/DBNKDD/dataset/NSL-KDD_Processed/NSL_TestLabels_int.csv')
```

You can also hard code the parameters as well as you can input them when you run the program

```python 
pre_learning_rate = float(input("Please input the Pretraining learning rate(should be between 0 and 1) : ")) 
pre_training_epochs = int(input("Please input the Pretraining epochs(more >> better) : "))
pre_batch_size = int(input("Please input the Pretraining batch size(lower >> better) : "))
display_step = 1
```

Also you can change the network architecture

```python 
pre_n_hidden_1 = int(input("\nPlease input the Pretraing network's Hidden layer 1'st Neurons : ")) # 1st layer num features
pre_n_hidden_2 = int(input("Please input the Pretraing network's Hidden layer 2'nd Neurons : "))# 2nd layer num features 
pre_n_hidden_3 = int(input("Please input the Pretraing network's Hidden layer 3'rd Neurons : "))
pre_n_hidden_4 = int(input("Please input the Pretraing network's Hidden layer 4'th Neurons : "))
pre_n_input = 41 
```
