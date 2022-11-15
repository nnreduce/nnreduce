import time
from typing import Type, Tuple, Dict
import sys

from nnsmith.materialize import TestCase, Stage, Symptom, BugReport, Oracle
from nnsmith.backends import BackendFactory
from nnsmith.materialize.onnx import ONNXModel
import pandas as pd
import onnx
from onnx import ModelProto
import onnx.checker
import onnx.mapping
import onnx.helper

from nnreduce.onnx_utils import nrealop
from nnreduce.minimizer import (
    DDMin,
    NNReduce,
    TemporalNNReduce,
    PolyGraphBisect,
    PolyGraphLinear,
    Reducer,
)
from nnreduce.utils import *
from nnreduce.minimizer import onnx2nnsmith
from nnreduce.onnx_utils import readable_graph, mark_interm_as_outputs

# make a graphp
"""
graph_def = helper.make_graph(
    [node_def],        # nodes
    'test-model',      # name
    [X, pads, value],  # inputs
    [Y],               # outputs
)
"""


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Partition a model into multiple models."
    )
    parser.add_argument("--input", type=str, help="Input model folder.")
    parser.add_argument(
        "--backend",
        type=str,
        default="tvm",
        help="The type of backend as the executor.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="The device to run the model."
    )
    parser.add_argument("--suite", type=str, help="Evaluate a whole bench suite.")
    parser.add_argument("--method", type=str, default="std", help="Partition method.")
    parser.add_argument("--output", type=str, help="Path to output the result csv.")
    parser.add_argument(
        "--debug_min",
        type=str,
        help="Path to a duplicated bug report to debug NNReduce-T.",
    )
    parser.add_argument("--save_min", action="store_true", help="Save minimized model.")
    parser.add_argument(
        "--hack_ref", action="store_true", help="Save intermediate outputs."
    )
    parser.add_argument(
        "--ref", type=str, help="Path to a reference intermediate values."
    )
    args = parser.parse_args()

    if args.suite is not None:
        suite = [os.path.join(args.suite, d) for d in os.listdir(args.suite)]
    elif args.input is not None:
        suite = [args.input]
    else:
        raise ValueError("Please specify either --suite or --input.")

    if args.output is None:
        print("WARNING: output is None. Results will only be printed to stdout.")

    suite = sorted(suite)

    results = {
        "path": [],
        "before[full]": [],
        "after[full]": [],
        "before[real]": [],
        "after[real]": [],
        "n-exec": [],
        "time": [],
    }

    ReducerType: Type[Reducer] = {
        "nnreduce": NNReduce,
        "nnreduce-t": TemporalNNReduce,
        "std": DDMin,
        "linear": PolyGraphLinear,
        "bisect": PolyGraphBisect,
    }[args.method]

    failed_cases = []
    minimized: Dict[Tuple[Symptom, Stage], ModelProto] = {}

    if args.debug_min is not None:
        min_path = os.path.join(args.debug_min, "min.onnx")
        assert os.path.exists(min_path)
        min_model = onnx.load(min_path)
        succ_print(readable_graph(min_model))
        dup = BugReport.load(model_type=ONNXModel, root_folder=args.debug_min)
        minimized[(dup.symptom, dup.stage)] = [min_model]
        succ_print(f"Loaded minimized model from : {min_path}")

    if args.hack_ref and args.input is not None:
        testcase = TestCase.load(
            model_type=ONNXModel, root_folder=args.input, allow_no_oracle=True
        )
        fac = BackendFactory.init(name=args.backend, target=args.device, optmax=True)
        dbg_testcase = TestCase(
            model=onnx2nnsmith(mark_interm_as_outputs(testcase.model.native_model)),
            oracle=Oracle(input=testcase.oracle.input, output=None),
        )
        outputs = fac.checked_compile_and_exec(dbg_testcase)
        assert not isinstance(outputs, BugReport), outputs

        import pickle

        with open(os.path.join(args.input, "ref.pkl"), "wb") as f:
            pickle.dump(outputs, f)

        sys.exit(0)

    ref = None
    if args.ref is not None:
        import pickle

        with open(args.ref, "rb") as f:
            ref = pickle.load(f)

    for idx, folder in enumerate(suite):
        print(f"------------- Evaluating {folder} ------------- {idx + 1}/{len(suite)}")

        testcase = TestCase.load(
            model_type=ONNXModel, root_folder=folder, allow_no_oracle=True
        )
        factory = BackendFactory.init(
            name=args.backend, target=args.device, optmax=True
        )
        try_org_succ = False
        try:
            nnmin = ReducerType(
                testcase=testcase,
                factory=factory,
                log_path=os.path.join(folder, "ddmin.log"),
                minimized=minimized,
                ref=ref,
            )
            tstart = time.time()
            minimized_model = nnmin.reduce()
            try_org_succ = True
        except Exception:
            import traceback

            traceback.print_exc()

        if not try_org_succ:
            try:
                nnmin = ReducerType(
                    testcase=testcase,
                    factory=factory,
                    do_simplify=True,
                    log_path=os.path.join(folder, "ddmin.log"),
                )
                tstart = time.time()
                minimized_model = nnmin.reduce()
            except Exception:
                fail_print("Failed for simplified model...")
                traceback.print_exc()
                failed_cases.append(folder)
                continue

        print(minimized_model)
        minimized.setdefault((nnmin.bug_symptom, nnmin.bug_stage), []).append(
            minimized_model.onnx_model
        )
        if nnmin.value_oracle and nnmin.bug_stage == Stage.VERIFICATION:
            for v in minimized_model.onnx_model.graph.input:
                if v.name in nnmin.value_oracle:
                    succ_print(f"<== Input {v.name}:")
                    succ_print(nnmin.value_oracle[v.name])
            for v in minimized_model.onnx_model.graph.output:
                if v.name in nnmin.value_oracle:
                    succ_print(f"==> Oracle Output {v.name}:")
                    succ_print(nnmin.value_oracle[v.name])

        results["path"].append(folder)
        results["before[full]"].append(len(testcase.model.native_model.graph.node))
        results["after[full]"].append(len(minimized_model.onnx_model.graph.node))
        results["before[real]"].append(nrealop(testcase.model.native_model.graph.node))
        results["after[real]"].append(nrealop(minimized_model.onnx_model.graph.node))
        results["n-exec"].append(nnmin.counter)
        results["time"].append(time.time() - tstart)
        if args.save_min:
            onnx.save(
                minimized_model.onnx_model,
                os.path.join(folder, "min.onnx"),
            )

    print(f"{len(suite) - len(failed_cases)} / {len(suite)} cases minimized")
    if failed_cases:
        print(f"Failed cases: {failed_cases}")

    if len(results["path"]) > 0:
        df = pd.DataFrame(results)
        if args.suite is not None:
            if args.output is not None:
                df.to_csv(args.output, index=False)
            else:
                extra = ""
                if "vulops" in args.suite:
                    extra = "vulops-"
                df.to_csv(
                    f"results/{factory.system_name}-{factory.version}-{extra}{args.method}.csv",
                    index=False,
                )

        df["path"] = df["path"].apply(lambda x: x.split("/")[-1])
        print(df)

    print_stats()  # Profiler. Show if profiled.
