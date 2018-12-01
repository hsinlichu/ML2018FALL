#!/bin/bash
cat bag8* >> bagging8
cat bag9* >> bagging9
python3 bagging_predict.py $1 $2
