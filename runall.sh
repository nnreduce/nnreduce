#!/bin/bash

set -e
set -x

export PYTHONPATH=$PWD:$PWD/tvm-builds/tvm-0.10.0/python
bash run_ort.sh "$PWD"/fuzzd/ort-1.14.0-4h-nnsmith-debug-bugs
bash run_ort.sh "$PWD"/fuzzd/ort-1.14.0-4h-nnsmith-vulops-bugs

bash run_tvm.sh "$PWD"/fuzzd/tvm-0.10.0-4h-nnsmith-bugs
bash run_tvm.sh "$PWD"/fuzzd/tvm-0.10.0-4h-nnsmith-vulops-bugs

export PYTHONPATH=$PWD:$PWD/tvm-builds/tvm-0.9.0/python
bash run_tvm.sh "$PWD"/fuzzd/tvm-0.9.0-4h-nnsmith-bugs
