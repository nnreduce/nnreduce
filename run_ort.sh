#!/bin/bash

set -e
set -x

python nnreduce/main.py --suite $1 --method std --backend onnxruntime --device cuda
python nnreduce/main.py --suite $1 --method nnreduce-t --backend onnxruntime --device cuda
python nnreduce/main.py --suite $1 --method nnreduce --backend onnxruntime --device cuda --save_min
python nnreduce/main.py --suite $1 --method linear --backend onnxruntime --device cuda
python nnreduce/main.py --suite $1 --method bisect --backend onnxruntime --device cuda
