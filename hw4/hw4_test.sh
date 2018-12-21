#!/bin/bash
#bash hw4_test.sh <test_x file> <dict.txt.big file> <output file>
wget 'https://www.dropbox.com/s/rhtcgk7twt5fg27/compress.tar.bz2?dl=1' -O compress.tar.bz2
tar -jxv -f compress.tar.bz2 -C .
python3 preprocess.py $2 $1 test_x_processed.plk
python3 bagging.py $3 
