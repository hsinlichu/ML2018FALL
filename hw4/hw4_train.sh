#!/bin/bash
#bash hw4_train.sh <train_x file> <train_y file> <test_x.csv file> <dict.txt.big file>
python3 preprocess.py $4 $1 train_x_processed.plk
python3 model.py 0 $2
