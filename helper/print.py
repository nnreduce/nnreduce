import onnx
import onnx.helper

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="The onnx model to print."
    )
    args = parser.parse_args()

    model = onnx.load(args.model)
    model = onnx.shape_inference.infer_shapes(model=model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    print(onnx.helper.printable_graph(model.graph))
    print(
        f"# Real Op: { len([n for n in model.graph.node if n.op_type != 'Constant']) }"
    )
    print(f"{model.opset_import = }")

    pad_sizes = []
    new_ops = []
    for op in model.graph.node:
        if op.op_type == "Pad" and len(op.input) == 1:
            # make ONNX initializer
            pad_size_name = op.output[0] + "_pad_size"
            pad_val_name = op.output[0] + "_pad_val"
            pad_size = onnx.helper.make_tensor(
                name=pad_size_name,
                data_type=onnx.TensorProto.INT64,
                dims=(len(op.attribute[0].ints),),
                vals=op.attribute[0].ints,
            )
            pad_val = onnx.helper.make_tensor(
                name=pad_val_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=(1,),
                vals=[op.attribute[1].f],
            )
            pad_sizes.append(pad_size)
            pad_sizes.append(pad_val)
            new_pad = onnx.helper.make_node(
                "Pad",
                inputs=[op.input[0], pad_size_name, pad_val_name],
                outputs=op.output,
                name=op.name,
                mode="constant",
            )
            new_ops.append(new_pad)
            continue
        new_ops.append(op)

    new_graph = onnx.helper.make_graph(
        nodes=new_ops,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=pad_sizes + [ini for ini in model.graph.initializer],
        value_info=model.graph.value_info,
    )

    print("================ INPUTS =================")
    for vi in model.graph.input:
        print(onnx.helper.printable_value_info(vi))
    print("================ OUTPUT =================")
    for vi in model.graph.output:
        print(onnx.helper.printable_value_info(vi))
    print("================ VALUES =================")
    for vi in model.graph.value_info:
        print(onnx.helper.printable_value_info(vi))

    from onnxsim import simplify

    model_simp, check = simplify(
        model,
        perform_optimization=False,  # overwrite_input_shapes=input_shapes
    )

    print("================ ONNXSIM =================")
    print(onnx.helper.printable_graph(model_simp.graph))
    onnx.save(model_simp, args.model.replace(".onnx", "-simp.onnx"))
