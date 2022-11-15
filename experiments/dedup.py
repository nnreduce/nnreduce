# Deduplication experiment
import os

import onnx
import onnx.helper
import pandas as pd


from nnreduce.utils import succ_print, note_print
from nnreduce.dedup import check_isomorphism
from nnreduce.onnx_utils import readable_graph


CAP = ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deduplication analyzer.")
    parser.add_argument("--cfg", type=str, required=True, help="The output csv files")
    parser.add_argument(
        "--keep-unmin", action="store_true", help="analyze those unminimized."
    )
    parser.add_argument(
        "--system", type=str, required=True, help="the system under test"
    )
    args = parser.parse_args()

    d = pd.read_csv(args.cfg)

    tag = f"{args.system}-"

    if "tvm" in tag.lower():
        CAP = "T"
    elif "ort" in tag.lower() or "onnx" in tag.lower():
        CAP = "O"

    files = []
    for _, row in d.iterrows():
        if args.keep_unmin or row["before[real]"] > row["after[real]"]:
            files.append(os.path.join(row["path"], "min.onnx"))
        else:
            note_print(f"skipping {row['path']}")

    models = {}
    id2file = {}
    for i, f in enumerate(files):
        assert os.path.exists(f), f"{f} does not exist"
        label = f"{CAP}{1+i}"
        id2file[label] = f
        succ_print("\t", label, f)
        m = onnx.load(f)
        models[label] = m

    def cluster(debug=False, **kwargs):
        clusters = []
        for label in models:
            m = models[label]
            merged = False
            for c in clusters:
                if check_isomorphism(m, models[c[0]], **kwargs):
                    succ_print(label, "is matched with ", *c)
                    c.append(label)
                    merged = True
                    break
            if not merged:
                clusters.append([label])

        print([len(c) for c in clusters])
        succ_print(f"Get {len(clusters)} clusters with {kwargs}")

        if debug:
            for c in clusters:
                succ_print(f"cluster {c}")
                for cc in c:
                    succ_print(f"\t{id2file[cc]}")
                note_print(readable_graph(models[c[0]]))

    # Most relax.
    cluster(
        debug=True,
        op_aggr=True,
        strict_dtype=False,
        check_shape=False,
        check_attr=False,
    )
    # OpType exact match.
    cluster(op_aggr=False, strict_dtype=False, check_shape=False, check_attr=False)
    # DType exact match.
    cluster(op_aggr=True, strict_dtype=True, check_shape=False, check_attr=False)
    # Shape & Attr exact match.
    cluster(op_aggr=True, strict_dtype=True, check_shape=True, check_attr=True)
