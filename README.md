# NNReduce: Minimize and Deduplicate DL Compiler Bugs

## Quick Start

```shell
pip install -r requirements.txt
# Evaluate minimization methods over tvm-nnsmith dataset
python nnreduce/main.py --suite samples --method nnreduce --backend tvm
python nnreduce/main.py --suite samples --method std --backend tvm
python nnreduce/main.py --suite samples --method linear --backend tvm
python nnreduce/main.py --suite samples --method bisect --backend tvm
# [[Other methods]]
# `nnreduce-t`: nnreduce with temporal reduction; (Use it for fuzzing bugs)
# `std`: topological delta debugging;
# `linear`: Polygraphy linear mode;
# `bisect`: Polygraphy bisect mode;
# [[Other backends]]
# `onnxruntime`

# Output will be generated as an `.csv` file under `results/`:
# Such as:
# - results/tvm-0.9.0-nnreduce.csv
# - results/tvm-0.9.0-std.csv
# - results/tvm-0.9.0-linear.csv
# - results/tvm-0.9.0-bisect.csv
python experiments/barplot.py \
       --inps results/tvm-0.9.0-nnreduce.csv \
              results/tvm-0.9.0-std.csv      \
              results/tvm-0.9.0-linear.csv   \
              results/tvm-0.9.0-bisect.csv   \
       --label "NNReduce" "Topological DD" "Polygraph-Linear" "Polygraph-Bisect" \
       --system tvm
# Check `results/tvm-speed.png` and `results/tvm-reduction`.
```

## Evaluate the whole fuzzing dataset

### S1: Run reduction

- Install tvm-0.9 and tvm-0.10 as `tvm-builds/tvm-0.9.0` and `tvm-builds/tvm-0.10.0`;
- Install the fuzzing dataset as `fuzzd/`;
- Run `./runall.sh`;

### S2: Visualization

```shell
# - results/ort-reduction.png
# - results/ort-speed.png
python experiments/barplot.py \
       --inps results/onnxruntime-1.14.0-nnreduce.csv   \
              results/onnxruntime-1.14.0-nnreduce-t.csv \
              results/onnxruntime-1.14.0-std.csv        \
              results/onnxruntime-1.14.0-linear.csv     \
              results/onnxruntime-1.14.0-bisect.csv     \
              --label "NNReduce" "NNReduce-T" "Topological DD" "Polygraph-Linear" "Polygraph-Bisect" \
              --system ort

# - results/tvm-reduction.png
# - results/tvm-speed.png
python experiments/barplot.py \
       --inps results/tvm-0.10.0-nnreduce.csv   \
              results/tvm-0.10.0-nnreduce-t.csv \
              results/tvm-0.10.0-std.csv        \
              results/tvm-0.10.0-linear.csv     \
              results/tvm-0.10.0-bisect.csv     \
              --label "NNReduce" "NNReduce-T" "Topological DD" "Polygraph-Linear" "Polygraph-Bisect" \
              --system tvm
```

### S3: Deduplication

```shell
python experiments/dedup.py --cfg results/tvm-0.10.0-nnreduce.csv --system tvm
python experiments/dedup.py --cfg results/onnxruntime-1.14.0-nnreduce.csv --system onnxruntime
```

## Developer Note

- `pip install -r requirements.txt`
- `pip install -r requirements-dev.txt`
- `pre-commit install`
