#!/bin/bash

wget -O ./model.h5 "https://www.dropbox.com/s/am5tf8en0zj2mc5/model.h5?dl=1"
python ./model_seq2seq.py $1 $2 $3