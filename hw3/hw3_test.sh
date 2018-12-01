#!/bin/bash
wget 'https://www.dropbox.com/s/l4xsh1te0t8z1ml/bagging0.h5?dl=1' -O bagging0.h5
wget 'https://www.dropbox.com/s/lgdvp26ud9cjq2e/bagging1.h5?dl=1' -O bagging1.h5
wget 'https://www.dropbox.com/s/c7nk237cm0st6my/bagging3.h5?dl=1' -O bagging3.h5
wget 'https://www.dropbox.com/s/8sh5gzrt1xxzejb/bagging4.h5?dl=1' -O bagging4.h5
wget 'https://www.dropbox.com/s/yrk1aavnu9nneqt/bagging5.h5?dl=0' -O bagging5.h5
wget 'https://www.dropbox.com/s/b838sl7k5wb2lgn/bagging6.h5?dl=1' -O bagging6.h5
wget 'https://www.dropbox.com/s/l32xzr4245mr4ho/bagging7.h5?dl=1' -O bagging7.h5

cat bag8* >> bagging8.h5
cat bag9* >> bagging9.h5
python3 bagging_predict.py $1 $2
