#!/bin/bash

set -e
set -x

python nnreduce/main.py --suite $1 --method std --backend tvm
python nnreduce/main.py --suite $1 --method nnreduce-t --backend tvm
python nnreduce/main.py --suite $1 --method nnreduce --backend tvm --save_min
python nnreduce/main.py --suite $1 --method linear --backend tvm
python nnreduce/main.py --suite $1 --method bisect --backend tvm
