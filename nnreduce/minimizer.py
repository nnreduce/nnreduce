from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import time
import os

from rich.console import Console
from rich.table import Table
import numpy as np
from onnx import ModelProto
from nnsmith.materialize import TestCase, BugReport, Oracle, Symptom, Stage
from nnsmith.backends import BackendFactory
from nnsmith.backends.tvm import TVMFactory
from nnsmith.backends.onnxruntime import ORTFactory
from nnsmith.materialize.onnx import ONNXModel, analyze_onnx_io
from difflib import SequenceMatcher
import onnx
import onnx.checker
import onnx.mapping
import onnx.helper
from onnx.helper import printable_type, make_tensor_type_proto
from onnx.numpy_helper import to_array
from onnx import shape_inference
from onnx import ModelProto, ValueInfoProto
from onnxsim import simplify

from nnreduce.utils import *
from nnreduce.onnx_utils import (
    readable_graph,
    mark_interm_as_outputs,
    realop,
    nrealop,
    is_dyn_shape,
    check_well_formed,
)
from nnreduce.partitioner import ONNXPartitioner
from nnreduce.dedup import try_match


def onnx2nnsmith(onnx_model: ModelProto) -> ONNXModel:
    nnsmith_onnx_model = ONNXModel(with_torch=False)
    full_input_like, full_output_like = analyze_onnx_io(onnx_model)
    nnsmith_onnx_model.onnx_model = onnx_model
    nnsmith_onnx_model.full_input_like = full_input_like
    nnsmith_onnx_model.full_output_like = full_output_like
    nnsmith_onnx_model.masked_output_like = nnsmith_onnx_model.full_output_like
    return nnsmith_onnx_model


def debugger_run(
    model: ModelProto, original_testcase: TestCase = None, sysname: str = None
) -> Dict[str, np.ndarray]:
    debug_model = onnx2nnsmith(model)

    tvm_dbg = TVMFactory(target="cpu", optmax=False, executor="debug")
    ort_dbg = ORTFactory(target="cpu", optmax=False)
    debug_facs: List[BackendFactory] = [
        tvm_dbg,
        ort_dbg,
        TVMFactory(target="cuda", optmax=False, executor="debug"),
    ]

    if sysname == tvm_dbg.system_name:
        debug_facs.reverse()  # Always try a different compiler first.

    input = None
    if original_testcase is not None:
        input = original_testcase.oracle.input
    testcase = TestCase(
        model=debug_model,
        oracle=Oracle(input=input, output=None),
    )

    for debug_fac in debug_facs:
        outputs = debug_fac.checked_compile_and_exec(testcase)
        if not isinstance(outputs, BugReport):
            succ_print(f"Succeeded with {debug_fac}")
            if (
                original_testcase
                and original_testcase.oracle.output
                and debug_fac.verify_results(
                    {k: outputs[k] for k in original_testcase.oracle.output},
                    original_testcase,
                )
            ):
                # Shall we check
                note_print("DBG result != oracle. Can be a bug in the oracle ref.")

            return outputs
        fail_print(f"Failed to run with {debug_fac}")
        # Only show the last 4 lines of error:
        note_print("...\n" + "\n".join(outputs.log.splitlines()[-4:]))

    raise RuntimeError("TVM-debug and ORT-debug cannot compile this onnx model.")


def make_value_info(values: Dict[str, np.ndarray]) -> List[ValueInfoProto]:
    """
    https://github.com/onnx/onnx/blob/main/docs/IR.md#graphs
    > Used to store the type and shape information of values that are not inputs or outputs.
    But we include inputs & outputs anyways.
    """
    return [
        onnx.helper.make_tensor_value_info(
            k, onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[v.dtype], v.shape
        )
        for k, v in values.items()
    ]


def annotate_value_info(
    model: ModelProto, value_info: List[ValueInfoProto]
) -> ModelProto:
    name2vinfo = {vi.name: vi for vi in value_info}
    onames = set()
    for n in model.graph.node:
        for o in n.output:
            onames.add(o)

    graph_def = onnx.helper.make_graph(
        nodes=model.graph.node,  # nodes
        name="amplified_graph",  # name
        inputs=[name2vinfo[i.name] for i in model.graph.input],  # inputs
        outputs=[name2vinfo[o.name] for o in model.graph.output],  # outputs
        initializer=model.graph.initializer,
        value_info=[name2vinfo[oname] for oname in onames],
    )
    model_def = onnx.helper.make_model(graph=graph_def, producer_name="onnx-ddmin")

    # This might fail as ONNX's current shape inference is weak.
    # onnx.checker.check_model(model_def, full_check=True)
    check_well_formed(model_def)
    return model_def


def get_all_vinfo(model: ModelProto) -> Dict[str, ValueInfoProto]:
    n2vinfo = {vi.name: vi for vi in model.graph.value_info}
    n2ini = {ini.name: ini for ini in model.graph.initializer}
    n2vinfo.update({vi.name: vi for vi in model.graph.input})
    n2vinfo.update({vi.name: vi for vi in model.graph.output})
    n2vinfo.update(
        {
            k: onnx.helper.make_tensor_value_info(k, ini.data_type, ini.dims)
            for k, ini in n2ini.items()
        }
    )
    return n2vinfo


def split_static_dynamic_vinfo(
    vinfo: Dict[str, ValueInfoProto]
) -> Tuple[Dict[str, ValueInfoProto], Dict[str, ValueInfoProto]]:
    dyn_vinfo = {}
    sta_vinfo = {}

    for vn, vi in vinfo.items():
        if is_dyn_shape(vi):
            dyn_vinfo[vn] = vi
        else:
            sta_vinfo[vn] = vi

    return sta_vinfo, dyn_vinfo


def make_vinfo_model(
    onnx_model: ModelProto, n2vinfo: Dict[str, ValueInfoProto], dyn_names: List[str]
) -> Optional[ModelProto]:
    """Given a model and a list of value info, construct a minimal submodel that outputs such value info. The implementation is
    to construct the model in bottom-up fashion where the vinfo is marked as outputs and we add its dependencies until all dependencies
    are concrete values."""

    n2node = {node.name: node for node in onnx_model.graph.node}
    o2node = {}  # TopK has multiple outputs
    for node in onnx_model.graph.node:
        for oname in node.output:
            o2node[oname] = node
    inputs = dyn_names
    export_node_names = []
    masked_vinfo = set()

    changed = True
    while changed:
        changed = False
        for vname in inputs:
            if vname in masked_vinfo:
                continue
            val = n2vinfo[vname]
            if is_dyn_shape(val):
                producer = o2node[vname]
                if producer.name in export_node_names:
                    continue
                export_node_names.append(producer.name)
                if len(producer.input) == 0:  # post-leaf node
                    raise ValueError("WARNING: post-leaf node still has dynamic shape")
                masked_vinfo.add(vname)
                for pi in producer.input:
                    if len(pi) > 0:
                        inputs.append(pi)
                changed = True

    inputs = [inp for inp in inputs if len(inp) > 0 and inp not in masked_vinfo]
    outputs = set(dyn_names)
    for node_name in export_node_names:
        for oname in n2node[node_name].output:
            outputs.add(oname)

    varnames = []
    for n in export_node_names:
        varnames.extend(n2node[n].input)
        varnames.extend(n2node[n].output)
    varnames = list(set(varnames) - set([""]))

    newgraph = onnx.helper.make_graph(
        nodes=[n2node[n] for n in export_node_names],
        name="vinfo_graph",
        inputs=[n2vinfo[vi] for vi in inputs],
        outputs=[n2vinfo[vi] for vi in outputs],
        initializer=[
            ini for ini in onnx_model.graph.initializer if ini.name in varnames
        ],
    )

    newmodel = onnx.helper.make_model(graph=newgraph, producer_name="onnx-ddmin")
    onnx.checker.check_model(newmodel, full_check=True)

    return newmodel


class Reducer(ABC):
    def __init__(
        self,
        testcase: TestCase,
        factory: BackendFactory,
        do_simplify=False,
        log_path=None,
        minimized=None,
        ref=None,
    ):
        self.opt_fac = factory
        bug_report = self.opt_fac.verify_testcase(testcase)
        assert (
            bug_report
        ), "The first execution did not fail (no need to reduce a 'bug')!"

        if do_simplify:
            # try to see if simplifed model can reproduce the bug.
            model_simp, check = simplify(
                testcase.model.native_model, perform_optimization=False
            )
            assert check, "Simplified ONNX model could not be validated"
            model_simp = onnx2nnsmith(model_simp)
            testcase = TestCase(model_simp, testcase.oracle)
            bug_report_simp = self.opt_fac.verify_testcase(testcase)
            assert bug_report_simp and bug_report_simp.log == bug_report.log
            succ_print("Simplified model can reproduce the bug!")

        note_print("The original bug report:")
        print(bug_report)
        note_print(f"Reduction Type: {bug_report.symptom} at {bug_report.stage}")

        self.bug_stage = bug_report.stage
        self.bug_symptom = bug_report.symptom
        self.oracle_msg = bug_report.log

        onnx_model = shape_inference.infer_shapes(
            testcase.model.native_model, strict_mode=True
        )

        n2vinfo = get_all_vinfo(onnx_model)

        self.value_oracle = ref
        self.value_info = None
        if self.value_oracle is None:
            if bug_report.stage == Stage.VERIFICATION:
                self.value_oracle = debugger_run(
                    mark_interm_as_outputs(onnx_model),
                    testcase,
                    self.opt_fac.system_name,
                )
            else:
                try:  # Try full inference first.
                    self.value_oracle = debugger_run(
                        mark_interm_as_outputs(onnx_model),
                        testcase,
                        self.opt_fac.system_name,
                    )
                except Exception:
                    sta_vinfo, dyn_vinfo = split_static_dynamic_vinfo(n2vinfo)
                    if dyn_vinfo:  # Fall back: partial inference.
                        # Do some partial inference to get the static shape of `dyn_info`.
                        note_print(
                            "Full inference failed, falling back to partial inference."
                        )
                        partial_graph = make_vinfo_model(
                            onnx_model, n2vinfo, list(dyn_vinfo.keys())
                        )
                        concrete_values = debugger_run(partial_graph)
                        self.value_info = list(sta_vinfo.values()) + [
                            val
                            for val in make_value_info(concrete_values)
                            if val.name not in sta_vinfo
                        ]
                    else:
                        self.value_info = list(sta_vinfo.values())

        if self.value_info is None and self.value_oracle is not None:
            self.value_info = make_value_info(self.value_oracle)

        if self.value_oracle is not None and self.bug_stage == Stage.VERIFICATION:
            # check the bug with debugger generated oracle.
            bug_report = self.opt_fac.verify_testcase(
                TestCase(
                    testcase.model,
                    oracle=Oracle(
                        input=testcase.oracle.input,
                        output={
                            k: self.value_oracle[k] for k in testcase.oracle.output
                        },
                    ),
                )
            )
            assert (
                bug_report
            ), "Cannot reproduce the bug with debugger generated oracle!"

        amplified_onnx_model = annotate_value_info(onnx_model, self.value_info)
        self.base_model = ONNXPartitioner(
            amplified_onnx_model, value_oracle=self.value_oracle
        )

        if log_path is None:
            log_path = "_ddmin.log"
        self.logger = open(log_path, "w")
        self.logger.write(f"[ORACLE]{self.oracle_msg}--\n")

        self.counter = 0
        self.minimized = []
        if minimized is not None and (self.bug_symptom, self.bug_stage) in minimized:
            self.minimized = minimized[(self.bug_symptom, self.bug_stage)]

    def text_similarity(self, msg) -> float:
        return SequenceMatcher(None, self.oracle_msg, msg).ratio()

    def check_onnx_model(self, onnx_model: ModelProto) -> Optional[BugReport]:
        model = onnx2nnsmith(onnx_model)

        if self.bug_stage == Stage.VERIFICATION:
            input = {k: self.value_oracle[k] for k in model.input_like}
            output = {k: self.value_oracle[k] for k in model.output_like}
            oracle = Oracle(input=input, output=output)
            testcase = TestCase(model=model, oracle=oracle)
            return self.opt_fac.verify_testcase(testcase)
        else:
            bug_or_tcase = self.opt_fac.make_testcase(model)
            if isinstance(bug_or_tcase, BugReport):
                return bug_or_tcase
            return None

    def step(
        self, model: ONNXPartitioner, groups: List[List[str]]
    ) -> Optional[ONNXPartitioner]:
        for subg_names in groups:
            subg = model.make_graph(node_names=subg_names, model_name=f"subg")
            if subg is None:
                continue
            assert (
                nrealop(subg.graph.node) < model.op_size()
            ), "Internal Error: split subgraph has the same size as the original one!"
            checked_g = self.evaluate(subg)
            if checked_g is not None:
                return checked_g

        return None

    def evaluate(self, subg) -> Optional[ONNXPartitioner]:
        bug_report = self.check_onnx_model(subg)
        self.counter += 1
        if bug_report:
            if (
                self.bug_stage == bug_report.stage
                and self.bug_symptom == bug_report.symptom
                and (
                    self.bug_stage == Stage.VERIFICATION
                    or self.text_similarity(bug_report.log) >= 0.9
                )  # Thresh.
            ):
                new_model = ONNXPartitioner(subg, value_oracle=self.value_oracle)
                succ_print(f"Bug minimized a bit! Now #real op: {new_model.op_size()}")
                return new_model
            else:
                note_print("Found bug but the log looks not like the oracle log.")
                line = "-" * 24 + "\n"
                self.logger.write(line)
                self.logger.write(line)
                self.logger.write(bug_report.log)
                self.logger.write(
                    "~" * 8
                    + "[^] Mismatched Log; Corresponding Subgraph [v]"
                    + "~" * 8
                    + "\n"
                )
                self.logger.write(readable_graph(subg) + "\n")
                self.logger.write(line)
                self.logger.write(line)
                self.logger.flush()
        return None

    @abstractmethod
    def reduce(self) -> ONNXPartitioner:
        pass


class DDMin(Reducer):
    @staticmethod
    def cut(model: ONNXPartitioner, granularity=2) -> Optional[List[List[str]]]:
        """
        The standard delta-debugging algorithm follows the following steps:
        1. Reduce to subset: Reduce the graph to $granularity parts, and try each of them;
        2. Reduce to complement: Try all possible subsets of the remaining nodes;
        3. Increase granularity if failed.
        """
        real_nodes = realop(model.onnx_model.graph.node)
        op_size = len(real_nodes)
        if op_size <= 1:
            print("WARNING: op_size <= 1")
            return None, None

        # partition into $granularity parts

        all_nodes = [node.name for node in real_nodes]
        pars = list(split(all_nodes, granularity))

        # Reduce to subset
        subgraphs = pars.copy()

        # Reduce to complement
        if granularity > 2:  # No need to reduce to complement if granularity == 2
            for par in pars:
                nodes = []
                for n in all_nodes:
                    if n not in par:
                        nodes.append(n)
                if nodes:  # if not empty
                    subgraphs.append(nodes)

        return subgraphs

    def reduce(self, start_point=None) -> ONNXPartitioner:
        cur_model = cur_model = self.base_model if start_point is None else start_point
        last_granularity = None
        granularity = min(2, cur_model.op_size())
        while cur_model.op_size() > 1 and last_granularity != granularity:
            subgs = self.cut(cur_model, granularity)
            if len(subgs) == 2:
                subsets = subgs
                complements = None
            else:
                subsets = subgs[: len(subgs) // 2]
                complements = subgs[len(subgs) // 2 :]

            new_model_or = self.step(cur_model, subsets)

            if new_model_or:  # "reduce to subset"
                cur_model = new_model_or
                granularity = 2
                last_granularity = None
                continue

            if complements:
                new_model_or = self.step(cur_model, complements)
                if new_model_or:  # "reduce to complement"
                    cur_model = new_model_or
                    granularity = max(granularity - 1, 2)
                    last_granularity = None
                    continue

            # "increase granularity"
            last_granularity = granularity
            granularity = min(cur_model.op_size(), 2 * granularity)  # following DD.
            print(f"Granularity: {last_granularity}->{granularity}")

        succ_print(f"Minimized to {cur_model.op_size()} op after {self.counter} runs.")
        return cur_model


class PolyGraphLinear(Reducer):
    @staticmethod
    def cut(model) -> Optional[List[List[str]]]:
        """
        Reproduced from https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/tools/README.md#reducing-failing-onnx-models
        Reduce the model using "linear" mode
        #TODO: decide to use index or not
        """

        if len(model.graph.node) <= 1:
            print("WARNING: node_size <= 1")
            return None, None

        all_nodes = [
            node.name for node in model.graph.node if node.op_type != "Constant"
        ]

        return [all_nodes[1:], all_nodes[:-1]]

    def reduce(self) -> ONNXPartitioner:
        cur_model = self.base_model
        while cur_model.op_size() > 1:
            new_model_or = self.step(
                cur_model, PolyGraphLinear.cut(cur_model.onnx_model)
            )
            if not new_model_or:
                break  # Cannot minimize anymore.
            cur_model = new_model_or

        return cur_model


class PolyGraphBisect(Reducer):
    @staticmethod
    def cut(model) -> Optional[List[List[str]]]:
        """
        Reproduced from https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/tools/README.md#reducing-failing-onnx-models
        Reduce the model using "bisect" mode
        """

        node_size = len(model.graph.node)
        if node_size <= 1:
            print("WARNING: node_size <= 1")
            return None, None

        mid = node_size // 2
        all_nodes = [
            node.name for node in model.graph.node if node.op_type != "Constant"
        ]
        return [all_nodes[:mid], all_nodes[mid:]]

    def reduce(self) -> ONNXPartitioner:
        cur_model = self.base_model
        while cur_model.op_size() > 1:
            new_model_or = self.step(
                cur_model, PolyGraphBisect.cut(cur_model.onnx_model)
            )
            if not new_model_or:
                break  # Cannot minimize anymore.
            cur_model = new_model_or

        return cur_model


class NNReduce(Reducer):
    # Node weights
    W_SCALAR_INP = 1.1
    W_FP_COMPARATOR = 1.5
    W_OP_IN_LOG = 2

    def init_weight_map(self) -> Dict[str, float]:
        if os.getenv("DEBUG"):
            print(readable_graph(self.base_model.onnx_model))
        self.nconstops = [
            node
            for node in self.base_model.onnx_model.graph.node
            if node.op_type != "Constant"
        ]
        opname2idx = {node.name: idx for idx, node in enumerate(self.nconstops)}

        self.adjgraph = np.zeros(
            (len(self.nconstops), len(self.nconstops)), dtype=np.bool_
        )

        # ==[BEGIN]== init node weight
        self.nmap = np.ones(len(self.nconstops), dtype=np.float32)
        for idx in range(len(self.nmap)):
            node = self.nconstops[idx]

            itypes = []
            otypes = []

            for inp in node.input:
                if inp == "":
                    continue

                if inp in self.base_model.oname2node:
                    inp_node = self.base_model.oname2node[inp]
                    if inp_node.op_type != "Constant":
                        src_node_id = opname2idx[inp_node.name]
                        self.adjgraph[src_node_id, idx] = self.adjgraph[
                            idx, src_node_id
                        ] = True

                if inp in self.base_model.name2input:
                    itypes.append(printable_type(self.base_model.name2input[inp].type))
                elif inp in self.base_model.name2output:
                    itypes.append(printable_type(self.base_model.name2output[inp].type))
                elif inp in self.base_model.name2value:
                    itypes.append(printable_type(self.base_model.name2value[inp].type))
                else:
                    ini = to_array(self.base_model.name2initializer[inp])

                    itypes.append(
                        printable_type(
                            make_tensor_type_proto(
                                elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[
                                    ini.dtype
                                ],
                                shape=ini.shape,
                            )
                        )
                    )

            for out in node.output:
                if out in self.base_model.name2input:
                    otypes.append(printable_type(self.base_model.name2input[out].type))
                elif out in self.base_model.name2output:
                    otypes.append(printable_type(self.base_model.name2output[out].type))
                elif out in self.base_model.name2value:
                    itypes.append(printable_type(self.base_model.name2value[out].type))
                else:
                    ini = to_array(self.base_model.name2initializer[out])
                    itypes.append(
                        printable_type(
                            make_tensor_type_proto(
                                elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[
                                    ini.dtype
                                ],
                                shape=ini.shape,
                            )
                        )
                    )

            # HEURISTIC
            # attribute complexity
            if len(node.attribute) > 0:
                ATTR_FACTOR = 1 + 0.1 * len(node.attribute)
                self.nmap[idx] *= ATTR_FACTOR

            # HEURISTIC
            nio_t = len(set(itypes + otypes))
            if nio_t > 1:
                self.nmap[idx] *= 1 + 0.1 * (nio_t - 1)

            # HEURISTIC
            if any(["scalar" in t for t in itypes + otypes]):
                self.nmap[idx] *= self.W_SCALAR_INP

            # HEURISTIC
            if self.bug_symptom != Symptom.INCONSISTENCY:
                if node.op_type in self.oracle_msg:
                    self.nmap[idx] *= self.W_OP_IN_LOG

            def has_fp_inp():
                for it in itypes:
                    if "float" in it.lower() or "double" in it.lower():
                        return True
                return False

            # HEURISTIC
            if self.bug_symptom == Symptom.INCONSISTENCY and has_fp_inp():
                if node.op_type in ["Equal", "Greater", "Less", "Not", "Or", "And"]:
                    self.nmap[idx] *= self.W_FP_COMPARATOR

        # VIZ :: Debug
        # table = Table(title="Weight Map")
        # table.add_column("Node", justify="center", style="cyan", no_wrap=True)
        # table.add_column("Weight", justify="center", style="magenta")
        if os.getenv("DEBUG"):
            tbegin = time.time()
            console = Console()
            # console.print(table)
            adj_table = Table(title="Adjacency Graph")
            adj_table.add_column("Node", justify="center", style="cyan", no_wrap=True)
            for idx in range(len(self.nmap)):
                adj_table.add_column(self.nconstops[idx].name, justify="center")
            adj_table.add_row(
                "Score",
                *[f"{v:.2f}" for v in self.nmap],
                style="magenta",
                end_section=True,
            )
            for idx in range(len(self.nmap)):
                row = [self.nconstops[idx].name]
                for jdx in range(len(self.nmap)):
                    row.append("X" if self.adjgraph[idx, jdx] else " ")
                adj_table.add_row(*row)
            console.print(adj_table)
            print(f"VIZ :: Adjacency Graph: {1000 * (time.time() - tbegin):.2f}ms")
        # ==[END]== init node weight

        # Maybe just go nodes for simplicity?
        # ==[BEGIN]== init edge weight
        # self.emap = np.ones((len(self.name2idx), len(self.name2idx)))
        # ==[END]== init edge weight

    def cut(self, model: ONNXPartitioner, granularity=2) -> Optional[List[List[str]]]:
        op_size = model.op_size()
        if op_size <= 1:
            print("WARNING: op_size <= 1")
            return None, None

        worklist = self.nmap.copy()
        for idx, op in enumerate(self.nconstops):
            if op.name not in model.name2node:
                worklist[idx] = 0

        subgraphs = []

        def search(batch_sizes, worklist, adjgraph):
            # partition into $granularity parts
            subgs = []
            for target_size in batch_sizes:
                subg = []
                adjs = []
                while len(subg) < target_size:
                    if len(adjs) == 0:
                        subg.append(np.argmax(worklist))
                        worklist[subg[-1]] = 0
                        adjs = np.where(
                            np.logical_and(adjgraph[subg[-1]], worklist > 0)
                        )[0]
                        continue
                    # get max neighbor
                    neighbors = worklist[adjs]
                    adjs_amaxes = np.where(neighbors == np.amax(neighbors))[0]
                    adjs_amaxes = adjs_amaxes[: target_size - len(subg)]
                    # remove adjs_amax from adjs
                    nexts = adjs[adjs_amaxes]
                    worklist[adjs[adjs_amaxes]] = 0  # Erase next in worklist
                    adjs = np.delete(adjs, adjs_amaxes)
                    # add neighbors of next to adjs
                    for next in nexts:
                        # assert next not in subg, f"{next} already in subg"
                        subg.append(next)
                        adjs = np.append(
                            adjs,
                            np.where(np.logical_and(adjgraph[next], worklist > 0))[0],
                        )
                    # remove duplicates
                    adjs = np.unique(adjs)

                subg.sort()
                subgs.append(subg)
            return subgs

        batch_sizes = np.array([len(s) for s in split(np.arange(op_size), granularity)])
        subgs = search(batch_sizes, worklist, self.adjgraph)
        subgraphs = [[self.nconstops[idx].name for idx in subg] for subg in subgs]

        # Reduce to complement
        if granularity > 2:  # No need to reduce to complement if granularity == 2
            for i in range(len(subgraphs)):
                nodes = [
                    node.name
                    for node in model.onnx_model.graph.node
                    if node.op_type != "Constant" and node.name not in subgraphs[i]
                ]
                if nodes:  # if not empty
                    subgraphs.append(nodes)

        if os.getenv("DEBUG"):
            for subg in subgraphs:
                print(subg)

        return subgraphs

    def reduce(self, start_point=None) -> ONNXPartitioner:
        self.init_weight_map()
        return DDMin.reduce(self, start_point)


class TemporalNNReduce(NNReduce):
    def reduce(self) -> ONNXPartitioner:
        # Try fast solve by looking at minimized.
        start_point = None
        for m in self.minimized:
            match = try_match(m, self.base_model.onnx_model)
            if match:
                subg = self.base_model.make_graph(node_names=match, model_name=f"subg")
                if subg is None:
                    continue
                start_point = self.evaluate(subg)
                if start_point:
                    succ_print("+++++++++++++++++++++++++++++++++++++++++++")
                    succ_print("+++ Found a matched to minimized model! +++")
                    succ_print("+++++++++++++++++++++++++++++++++++++++++++")
                    succ_print(start_point)
                    break

        return NNReduce.reduce(self, start_point=start_point)
