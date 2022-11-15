# To get the statistics of operator patterns.
import os

import onnx
import onnx.helper
import pandas as pd
from onnx.helper import printable_type, make_tensor_type_proto
from onnx.numpy_helper import to_array

from nnreduce.utils import succ_print, note_print


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
            files.append(row["path"])
        else:
            note_print(f"skipping {row['path']}")

    def get_stat(graph, log: str):
        stat = {
            "#attr": 0,
            "scalar": 0,
            "log": 0,
            "fpcmp": 0,
            "#ttype": 0,
        }
        nrealop = 0
        for node in graph.node:
            if node.op_type == "Constant":
                continue

            nrealop += 1

            if len(node.attribute) > 0:
                stat["#attr"] += len(node.attribute)

            itypes = set()
            otypes = set()

            for mi in graph.input:
                if mi.name in node.input:
                    itypes.add(printable_type(mi.type))
                if mi.name in node.output:
                    otypes.add(printable_type(mi.type))

            for mo in graph.output:
                if mo.name in node.input:
                    itypes.add(printable_type(mo.type))
                if mo.name in node.output:
                    otypes.add(printable_type(mo.type))

            for mv in graph.value_info:
                if mv.name in node.input:
                    itypes.add(printable_type(mv.type))
                if mv.name in node.output:
                    otypes.add(printable_type(mv.type))

            for ini in graph.initializer:
                ini_arr = to_array(ini)
                t = printable_type(
                    make_tensor_type_proto(
                        elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ini_arr.dtype],
                        shape=ini_arr.shape,
                    )
                )
                if ini.name in node.input:
                    itypes.add(t)
                if ini.name in node.output:
                    otypes.add(t)

            stat["#ttype"] += len(itypes.union(otypes))

            stat["scalar"] += any(["scalar" in t for t in itypes.union(otypes)])

            stat["log"] += node.op_type in log

            def has_fp_inp():
                for it in itypes:
                    if "float" in it.lower() or "double" in it.lower():
                        return True
                return False

            stat["fpcmp"] += has_fp_inp() and node.op_type in [
                "Equal",
                "Greater",
                "Less",
                "Not",
                "Or",
                "And",
            ]

        return stat

    total_stat_min = {}
    total_stat_full = {}

    for i, f in enumerate(files):
        assert os.path.exists(f), f"{f} does not exist"
        label = f"{CAP}{1+i}"
        succ_print("\t", label, f)

        original = onnx.load(os.path.join(f, "model.onnx")).graph
        minimized = onnx.load(os.path.join(f, "min.onnx")).graph
        log = open(os.path.join(f, "err.log")).read()

        stat_full = get_stat(original, log)
        stat_min = get_stat(minimized, log)

        # stat["path"].append(f)
        for k in stat_full:
            total_stat_full.setdefault(k, []).append(stat_full[k])
            total_stat_min.setdefault(k, []).append(stat_min[k])

        total_stat_min.setdefault("#op", []).append(
            len([n for n in minimized.node if n.op_type != "Constant"])
        )
        total_stat_full.setdefault("#op", []).append(
            len([n for n in original.node if n.op_type != "Constant"])
        )

    df_min = pd.DataFrame(total_stat_min)
    res_min = df_min.sum() / df_min["#op"].sum()
    print(res_min)

    df_full = pd.DataFrame(total_stat_full)
    res_full = df_full.sum() / df_full["#op"].sum()
    print(res_full)

    for k in res_min.index:
        print(
            k,
            "\t",
            f"$\\nicefrac{{{res_min[k]:.2g}}}{{{res_full[k]:.2g}}}={res_min[k] / res_full[k]:.3g}\\times$",
        )
