#!/bin/bash

# Modify "train_x.csv" to your training x file path (Gene expr data)
train_x_filename="./train_x.csv" 

# Modify "train_y.csv" to your training y file path (having cell type label in integer type)
train_y_filename="./train_y.csv" 

# Modify "test_x.csv" to your testing x file path (Gene expr data)
test_x_filename="./test_x.csv" 

# Parameter for number of neighbors to select during cell-specific network construction (Default : 10)
num_neighbor="10" 


python3 construct_CSN.py $train_x_filename $train_y_filename $test_x_filename $num_neighbor
python3 predict_subtype.py $train_x_filename $train_y_filename $test_x_filename