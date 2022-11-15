import os
from typing import Dict

import numpy as np
import onnx
import onnx.checker
import onnx.mapping
import onnx.helper
from onnx import NodeProto

from nnreduce.onnx_utils import readable_graph
from nnreduce.utils import *
from nnreduce.onnx_utils import is_dyn_shape, check_well_formed


class ONNXPartitioner:
    def __init__(self, onnx_model, value_oracle=None):
        self.onnx_model = onnx_model
        self.name2node: Dict[str, NodeProto] = {
            node.name: node for node in self.onnx_model.graph.node
        }
        self.name2initializer = {
            initializer.name: initializer
            for initializer in self.onnx_model.graph.initializer
        }
        self.name2input = {input.name: input for input in self.onnx_model.graph.input}
        self.name2output = {
            output.name: output for output in self.onnx_model.graph.output
        }
        self.name2value = {
            value.name: value for value in self.onnx_model.graph.value_info
        }
        self.oname2node = {}
        for node in self.onnx_model.graph.node:
            for output in node.output:
                self.oname2node[output] = node
        self.constantable = set()
        for node in self.onnx_model.graph.node:
            if node.op_type == "Constant" or all(
                [inp in self.constantable for inp in node.input if inp != ""]
            ):
                for output in node.output:
                    self.constantable.add(output)
        self.value_oracle = value_oracle
        self._op_size = len(
            [n for n in self.onnx_model.graph.node if n.op_type != "Constant"]
        )

    def op_size(self) -> int:
        return self._op_size

    def node_size(self) -> int:
        return len(self.onnx_model.graph.node)

    def __repr__(self) -> str:
        return readable_graph(self.onnx_model.graph)

    def __str__(self) -> str:
        return readable_graph(self.onnx_model.graph)

    def make_graph(self, node_names, model_name="unamed", verbose=False):
        assert len(node_names) > 0, "node_names is empty!"
        if os.getenv("DEBUG"):
            check_well_formed(self.onnx_model)

        nodes = []
        inputs = []
        outputs = []
        initializer = []

        consumed_node_names = set()
        for node_name in node_names:
            node = self.name2node[node_name]
            for inp in node.input:
                consumed_node_names.add(inp)

        produced_node_names = set()
        for node_name in node_names:
            node = self.name2node[node_name]
            for out in node.output:
                produced_node_names.add(out)

        for node_name in node_names:
            node = self.name2node[node_name]
            nodes.append(node)

            # requires
            for ninp_name in node.input:
                if ninp_name and ninp_name not in produced_node_names:
                    # inputs or weights...
                    if ninp_name in self.name2initializer:
                        # reuse weights
                        initializer.append(self.name2initializer[ninp_name])
                    elif ninp_name in self.name2input:
                        # reuse inputs
                        inputs.append(self.name2input[ninp_name])
                    elif ninp_name in self.name2output:
                        # reuse outputs
                        inputs.append(self.name2output[ninp_name])
                    else:  # Create a new input node from original node.
                        inputs.append(self.name2value[ninp_name])
                    produced_node_names.add(ninp_name)
                    consumed_node_names.add(ninp_name)

            # produces
            for nout_name in node.output:
                # set as an output if it is an leaf node.
                # not consumed -> leaf!
                if nout_name not in consumed_node_names:
                    if nout_name in self.name2output:
                        outputs.append(self.name2output[nout_name])
                    else:
                        outputs.append(self.name2value[nout_name])
                    consumed_node_names.add(nout_name)

        # constantable => input
        constantized = set()
        for inp in inputs:
            if inp.name in self.constantable and inp.name not in constantized:
                if self.oname2node[inp.name].op_type == "Constant":
                    nodes.insert(0, self.oname2node[inp.name])
                else:
                    assert not is_dyn_shape(
                        inp
                    ), f"{onnx.helper.printable_value_info(inp)} has dynamic shape!"
                    dims = [v.dim_value for v in inp.type.tensor_type.shape.dim]
                    etype = inp.type.tensor_type.elem_type
                    vals = None
                    if self.value_oracle and inp.name in self.value_oracle:
                        vals = self.value_oracle[inp.name]
                    elif self.oname2node[inp.name].op_type == "ConstantOfShape":
                        vname = self.oname2node[inp.name].input[0]
                        vinfo = self.name2value[vname]
                        vals = np.array(
                            [v.dim_value for v in vinfo.type.tensor_type.shape.dim]
                        )
                    else:
                        note_print(f"Initializing constant {inp.name} from scratch")
                        # Don't use random here for determinism.
                        vals = np.ones(
                            shape=dims, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[etype]
                        )

                    if len(vals.shape) == 0:
                        vals = np.array([vals])
                    nodes.insert(
                        0,
                        onnx.helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[inp.name],
                            value=onnx.helper.make_tensor(
                                name=inp.name,
                                data_type=etype,
                                dims=dims,
                                vals=vals,
                                raw=False,
                            ),
                        ),
                    )
                constantized.add(inp.name)

        inputs = [inp for inp in inputs if inp.name not in constantized]

        if len(inputs) == 0:
            # In our world, models must have inputs.
            # (A model cannot be just a constant expression.)
            print(f"WARNING: no inputs... return None")
            return None

        needed_values = set()
        for node in nodes:
            for inp in node.input:
                if inp not in produced_node_names and inp in self.name2value:
                    needed_values.add(inp)
            for out in node.output:
                if out not in produced_node_names and out in self.name2value:
                    needed_values.add(out)

        vs = []
        for n in nodes:
            for o in n.output:
                vs.append(self.name2value[o])

        graph_def = onnx.helper.make_graph(
            nodes=nodes,  # nodes
            name=model_name,  # name
            inputs=inputs,  # inputs
            outputs=outputs,  # outputs
            initializer=initializer,
            value_info=vs,
        )

        if verbose:
            print("====== INPUTS ======")
            for inode in graph_def.input:
                print(onnx.helper.printable_value_info(inode))
            print("====== OUTPUTS ======")
            for onode in graph_def.output:
                print(onnx.helper.printable_value_info(onode))
            print("====== NODES ======")
            for nnode in graph_def.node:
                print(onnx.helper.printable_node(nnode))

        model_def = onnx.helper.make_model(graph=graph_def, producer_name="onnx-ddmin")

        check_well_formed(model_def)
        return model_def
