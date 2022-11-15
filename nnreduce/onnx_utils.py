from typing import List, Union, Dict, Tuple
from collections import namedtuple
import os
import traceback
import sys

import onnx
import onnx.helper
import onnx.checker
from onnx import GraphProto, ModelProto, ValueInfoProto
from onnx import shape_inference

ShapeType = namedtuple("ShapeType", ["shape", "dtype"])


def realop(nodes):
    return [n for n in nodes if n.op_type != "Constant"]


def nrealop(nodes):
    return len(realop(nodes))


def dtype_str(id: int) -> str:
    if id == 1:
        return "float32"
    elif id == 2:
        return "uint8"
    elif id == 3:
        return "int8"
    elif id == 4:
        return "uint16"
    elif id == 5:
        return "int16"
    elif id == 6:
        return "int32"
    elif id == 7:
        return "int64"
    elif id == 8:
        return "string"
    elif id == 9:
        return "bool"
    elif id == 10:
        return "float16"
    elif id == 11:
        return "double"
    elif id == 12:
        return "uint32"
    elif id == 13:
        return "uint64"
    elif id == 14:
        return "complex64"
    elif id == 15:
        return "complex128"
    elif id == 16:
        return "bfloat16"
    else:
        raise ValueError(
            f"Unknown dtype id: {id}. See https://deeplearning4j.org/api/latest/onnx/Onnx.TensorProto.DataType.html"
        )


def analyze_onnx_io(model: onnx.ModelProto) -> Tuple[Dict[str, ShapeType], List[str]]:
    """Analyze the input and output shapes of an ONNX model.

    Args:
        model (onnx.ModelProto): The model to be analyzed.

    Returns:
        Tuple[Dict[str, ShapeType], List[str]]: Input specifications (name -> {shape, dtype}) and output names.
    """
    inp_analysis_ret = {}
    out_analysis_names = [node.name for node in model.graph.output]

    # Note that there are 2 kinds of "inputs":
    # 1. The inputs provided by the user (e.g., images);
    # 2. The inputs provided by the model (e.g., the weights).
    # We only consider the first kind of inputs.
    weight_names = [node.name for node in model.graph.initializer]

    # Analyze the input shapes
    # Expected information:
    #   For each input:
    #     1. name
    #     2. shape (Note: `-1` stands for unknown dimension)
    #     3. data type
    # iterate through inputs of the graph
    for input_node in model.graph.input:
        if input_node.name in weight_names:
            continue
        # get type of input tensor
        tensor_type = input_node.type.tensor_type

        shape = []
        dtype = dtype_str(tensor_type.elem_type)

        # check if it has a shape:
        if tensor_type.HasField("shape"):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if d.HasField("dim_value"):
                    shape.append(d.dim_value)  # known dimension
                elif d.HasField("dim_param"):
                    # unknown dimension with symbolic name
                    shape.append(-1)
                else:
                    shape.append(-1)  # unknown dimension with no name
        else:
            raise ValueError("Input node {} has no shape".format(input_node.name))

        inp_analysis_ret[input_node.name] = ShapeType(shape, dtype)

    return inp_analysis_ret, out_analysis_names


def mark_interm_as_outputs(onnx_model):
    onnx_model = shape_inference.infer_shapes(onnx_model, strict_mode=True)
    name2initializer = {
        initializer.name: initializer for initializer in onnx_model.graph.initializer
    }
    name2input = {input.name: input for input in onnx_model.graph.input}
    name2output = {output.name: output for output in onnx_model.graph.output}
    name2value = {value.name: value for value in onnx_model.graph.value_info}

    onames = set()
    inames = set()

    for n in onnx_model.graph.node:
        for iname in n.input:
            inames.add(iname)
        for oname in n.output:
            onames.add(oname)

    inames -= {""}

    names = onames.union(inames)

    outputs = []

    necessary_init = []

    for name in names:
        if name in name2input:
            outputs.append(name2input[name])
        elif name in name2output:
            outputs.append(name2output[name])
        elif name in name2value:
            outputs.append(name2value[name])
        elif name in name2initializer:
            necessary_init.append(name)
        else:
            raise ValueError("Unknown name: {}".format(name))

    graph_def = onnx.helper.make_graph(
        nodes=onnx_model.graph.node,  # nodes
        name="nnreduce-all-out",  # name
        inputs=onnx_model.graph.input,  # inputs
        outputs=outputs,  # onnx_model.graph.value_info,  # outputs
        initializer=[
            ini for ini in onnx_model.graph.initializer if ini.name in necessary_init
        ],  # initializer
        value_info=[],  # value_infos
    )

    model_def = onnx.helper.make_model(graph=graph_def, producer_name="nnreduce")
    skip_check = False  # ONNX Bug? Check will fail if one operator has no shape in value info (cannot be inferred neither)
    for v in outputs:
        if v.type.tensor_type.HasField("shape"):
            skip_check = True
            break

    if not skip_check:
        onnx.checker.check_model(model_def, full_check=True)

    return model_def


def readable_graph(model):
    if isinstance(model, GraphProto):
        return onnx.helper.printable_graph(model)
    elif isinstance(model, ModelProto):
        return onnx.helper.printable_graph(model.graph)

    raise ValueError("model must be a GraphProto or ModelProto")


def get_onnx_proto(model: Union[onnx.ModelProto, str]) -> onnx.ModelProto:
    if isinstance(model, str):
        onnx_model = onnx.load(model)
    else:
        assert isinstance(model, onnx.ModelProto)
        onnx_model = model
    return onnx_model


def is_dyn_shape(vinfo: ValueInfoProto) -> bool:
    if not vinfo.type.tensor_type.HasField("shape"):
        return True
    for dim in vinfo.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value < 0:
            return True
    return False


def check_well_formed(model: onnx.ModelProto) -> None:
    val_names = set()
    for n in model.graph.node:
        for o in n.output:
            val_names.add(o)

    vinfo_names = {vinfo.name for vinfo in model.graph.value_info}
    try:
        assert vinfo_names == val_names, f"{vinfo_names} != {val_names}"

        for v in model.graph.value_info:
            assert not is_dyn_shape(
                v
            ), f"Value {onnx.helper.printable_value_info(v)} has dynamic shape!"
        for i in model.graph.input:
            assert not is_dyn_shape(
                i
            ), f"Input {onnx.helper.printable_value_info(i)} has dynamic shape!"
        for o in model.graph.output:
            assert not is_dyn_shape(
                o
            ), f"Output {onnx.helper.printable_value_info(o)} has dynamic shape!"
    except AssertionError as e:
        if os.getenv("DEBUG"):
            traceback.print_exc()
            sys.exit(0)
        raise e
