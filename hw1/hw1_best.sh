#!/bin/bash

#python3 model_best.py $1 model/model_best.h5
python3 test_best.py $1 model/model_best.h5 > $2
