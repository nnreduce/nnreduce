# Deduplication experiment
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

from onnx.helper import ValueInfoProto, TensorProto, NodeProto

from nnreduce.onnx_utils import is_dyn_shape

FLOATS = [TensorProto.FLOAT, TensorProto.DOUBLE]
INTS = [
    TensorProto.INT8,
    TensorProto.UINT8,
    TensorProto.INT16,
    TensorProto.UINT16,
    TensorProto.INT32,
    TensorProto.UINT32,
    TensorProto.INT64,
    TensorProto.UINT64,
]

OP_AGGREGATE = {
    "cmp": ["Equal", "Greater", "Less", "Not", "Or", "And"],
    # "reduce": ["ReduceMean", "ReduceSum", "ReduceProd", "ReduceMax", "ReduceMin"],
    # "aminmax": ["ArgMax", "ArgMin"],
    # "trivial-bin": ["Add", "Sub", "Mul"],
}


def check_op_aggre(opt1, opt2) -> bool:
    for v in OP_AGGREGATE.values():
        if opt1 in v and opt2 in v:
            return True
    return False


def check_dtype(d1, d2, strict=True) -> bool:
    # DOUBLE ~ FLOAT
    # INT64 ~ INT32
    if not strict:
        if d1 in FLOATS and d2 in FLOATS:
            return True
        if d1 in INTS and d2 in INTS:
            return True

    return d1 == d2


def check_isomorphism(
    m1, m2, op_aggr=True, strict_dtype=False, check_shape=False, check_attr=False
) -> bool:
    # Check graph isomorphism.
    # Relaxed equivalence.
    #   ~ Node: Node type;
    #   ~ Edge: Tensor data type;
    # Strict equivalence.
    #   ~ Node: Node type, Node attributes;
    #   ~ Edge: Tensor data type, Tensor shape;

    g1, g2 = m1.graph, m2.graph
    if (
        len(g1.node) != len(g2.node)
        or len(g1.input) != len(g2.input)
        or len(g1.output) != len(g2.output)
    ):
        return False

    @dataclass
    class NNInfo:
        n2op: Dict[str, ValueInfoProto]
        n2ini: Dict[str, TensorProto]
        n2inp: Dict[str, ValueInfoProto]
        n2out: Dict[str, ValueInfoProto]

    def make_data(g) -> NNInfo:
        n2op = {op.output[0]: op for op in g.node}
        n2ini = {ini.name: ini for ini in g.initializer}
        n2inp = {inp.name: inp for inp in g.input}
        n2out = {out.name: out for out in g.output}
        return NNInfo(n2op, n2ini, n2inp, n2out)

    ni1, ni2 = make_data(g1), make_data(g2)
    matched1 = {}
    matched2 = {}

    stage: List[Set[str]] = []

    def add_match(n1, n2):
        matched1[n1] = n2
        matched2[n2] = n1
        stage[-1].add(n1)

    def add_stack():
        stage.append(set())

    def undo():
        for n1 in stage[-1]:
            n2 = matched1[n1]
            matched2.pop(n2)
            matched1.pop(n1)

        stage.pop()

    def is_matched(n1, n2):
        return matched1.get(n1, None) == n2 and matched2.get(n2, None) == n1

    def check_match(n1, n2) -> bool:
        if len(n1) == len(n2) == 0:  # Resize bug.
            return True

        if is_matched(n1, n2):  # matched with each other.
            return True

        if n1 in matched1 or n2 in matched2:  # matched with other nodes.
            return False

        if n1 in ni1.n2out and n2 in ni2.n2out:
            o1, o2 = ni1.n2out[n1], ni2.n2out[n2]
            if not check_dtype(
                o1.type.tensor_type.elem_type,
                o2.type.tensor_type.elem_type,
                strict=strict_dtype,
            ):
                return False
            if check_shape and o1.type.tensor_type.shape != o2.type.tensor_type.shape:
                return False

        if n1 in ni1.n2op and n2 in ni2.n2op:
            op1, op2 = ni1.n2op[n1], ni2.n2op[n2]
            if not op_aggr and op1.op_type != op2.op_type:
                return False
            if op_aggr:
                if check_op_aggre(op1.op_type, op2.op_type):
                    pass
                elif op1.op_type != op2.op_type:
                    return False

            if check_attr and op1.attribute != op2.attribute:
                return False
            # Check operands.
            if len(op1.input) != len(op2.input):
                return False
            for i1, i2 in zip(op1.input, op2.input):
                if not check_match(i1, i2):
                    return False

        elif n1 in ni1.n2ini and n2 in ni2.n2ini:
            tp1, tp2 = ni1.n2ini[n1], ni2.n2ini[n2]
            if not check_dtype(
                tp1.data_type,
                tp2.data_type,
                strict=strict_dtype,
            ):
                return False
            if check_shape and tp1.dims != tp2.dims:
                return False
        elif n1 in ni1.n2inp and n2 in ni2.n2inp:
            i1, i2 = ni1.n2inp[n1], ni2.n2inp[n2]
            if not check_dtype(
                i1.type.tensor_type.elem_type,
                i2.type.tensor_type.elem_type,
                strict=strict_dtype,
            ):
                return False

            dyn1, dyn2 = is_dyn_shape(i1), is_dyn_shape(i2)
            if check_shape:
                dyn1, dyn2 = is_dyn_shape(i1), is_dyn_shape(i2)
                if dyn1 or dyn2:
                    # check rank: i.e., size of v.type.tensor_type.shape
                    if len(i1.type.tensor_type.shape.dim) != len(
                        i2.type.tensor_type.shape.dim
                    ):
                        return False
                elif i1.type.tensor_type.shape.dim != i2.type.tensor_type.shape.dim:
                    return False
        else:
            return False

        add_match(n1, n2)
        return True

    # Try unordered matching starting from output node:
    total_outs1 = [o.name for o in g1.output]
    total_outs2 = [o.name for o in g2.output]

    match_cache = set()

    def match_outs(outs1, outs2):
        assert len(outs1) == len(outs2)
        if len(outs1) == 0:
            return True

        o1 = outs1[0]
        for o2 in outs2:
            add_stack()
            if (o1, o2) in match_cache or check_match(o1, o2):
                match_cache.add((o1, o2))
                if match_outs(
                    [o for o in outs1 if o != o1], [o for o in outs2 if o != o2]
                ):
                    return True
            undo()

        return False

    return match_outs(total_outs1, total_outs2)


def try_match(
    mmin, mmax, op_aggr=True, strict_dtype=False, check_shape=False, check_attr=False
) -> Optional[List[NodeProto]]:

    # Check graph isomorphism.
    # Relaxed equivalence.
    #   ~ Node: Node type;
    #   ~ Edge: Tensor data type;
    # Strict equivalence.
    #   ~ Node: Node type, Node attributes;
    #   ~ Edge: Tensor data type, Tensor shape;

    gmin, gmax = mmin.graph, mmax.graph
    if len(gmin.node) > len(gmax.node):
        return False

    @dataclass
    class NNInfo:
        n2op: Dict[str, NodeProto]
        n2ini: Dict[str, TensorProto]
        n2inp: Dict[str, ValueInfoProto]
        n2out: Dict[str, ValueInfoProto]
        n2v: Dict[str, ValueInfoProto]

    def make_data(g) -> NNInfo:
        n2op = {op.output[0]: op for op in g.node}
        n2ini = {ini.name: ini for ini in g.initializer}
        n2inp = {inp.name: inp for inp in g.input}
        n2out = {out.name: out for out in g.output}
        n2v = {**n2inp, **n2out}
        n2v.update({v.name: v for v in g.value_info})
        return NNInfo(n2op, n2ini, n2inp, n2out, n2v)

    nnmin, nnmax = make_data(gmin), make_data(gmax)
    matched1 = {}
    matched2 = {}

    stage: List[Set[str]] = []

    def add_match(n1, n2):
        matched1[n1] = n2
        matched2[n2] = n1
        stage[-1].add(n1)

    def add_stack():
        stage.append(set())

    def undo():
        for n1 in stage[-1]:
            n2 = matched1[n1]
            matched2.pop(n2)
            matched1.pop(n1)

        stage.pop()

    def is_matched(n1, n2):
        return matched1.get(n1, None) == n2 and matched2.get(n2, None) == n1

    def check_match(opmin, opmax) -> bool:
        if len(opmin) == len(opmax) == 0:  # Resize bug.
            return True

        if is_matched(opmin, opmax):  # matched with each other.
            return True

        if opmin in matched1 or opmax in matched2:  # matched with other nodes.
            return False

        if opmin in nnmin.n2out and opmax in nnmax.n2v:
            o1, o2 = nnmin.n2out[opmin], nnmax.n2v[opmax]
            if not check_dtype(
                o1.type.tensor_type.elem_type,
                o2.type.tensor_type.elem_type,
                strict=strict_dtype,
            ):
                return False
            if check_shape and o1.type.tensor_type.shape != o2.type.tensor_type.shape:
                return False

        if opmin in nnmin.n2op and opmax in nnmax.n2op:
            op1, op2 = nnmin.n2op[opmin], nnmax.n2op[opmax]
            if not op_aggr and op1.op_type != op2.op_type:
                return False
            if op_aggr:
                if check_op_aggre(op1.op_type, op2.op_type):
                    pass
                elif op1.op_type != op2.op_type:
                    return False

            if check_attr and op1.attribute != op2.attribute:
                return False
            # Check operands.
            if len(op1.input) != len(op2.input):
                return False
            for i1, i2 in zip(op1.input, op2.input):
                if not check_match(i1, i2):
                    return False

        elif opmin in nnmin.n2ini and opmax in nnmax.n2ini:
            tp1, tp2 = nnmin.n2ini[opmin], nnmax.n2ini[opmax]
            if not check_dtype(
                tp1.data_type,
                tp2.data_type,
                strict=strict_dtype,
            ):
                return False
            if check_shape and tp1.dims != tp2.dims:
                return False
        elif opmin in nnmin.n2inp and opmax in nnmax.n2v:
            i1, i2 = nnmin.n2inp[opmin], nnmax.n2v[opmax]
            if not check_dtype(
                i1.type.tensor_type.elem_type,
                i2.type.tensor_type.elem_type,
                strict=strict_dtype,
            ):
                return False

            dyn1, dyn2 = is_dyn_shape(i1), is_dyn_shape(i2)
            if check_shape:
                dyn1, dyn2 = is_dyn_shape(i1), is_dyn_shape(i2)
                if dyn1 or dyn2:
                    # check rank: i.e., size of v.type.tensor_type.shape
                    if len(i1.type.tensor_type.shape.dim) != len(
                        i2.type.tensor_type.shape.dim
                    ):
                        return False
                elif i1.type.tensor_type.shape.dim != i2.type.tensor_type.shape.dim:
                    return False
        else:
            return False

        add_match(opmin, opmax)
        return True

    leaf_min = [
        (oname, op.op_type) for oname, op in nnmin.n2op.items() if oname in nnmin.n2out
    ]

    max_candidates = {
        oname: [
            maxname
            for (maxname, maxop) in nnmax.n2op.items()
            if maxop.op_type == optype
        ]
        for (oname, optype) in leaf_min
    }

    if any(len(cands) == 0 for cands in max_candidates.values()):
        return None  # Cannot be matched.

    match_cache = set()

    def match_outs(
        minlfs,
    ) -> bool:
        if len(minlfs) == 0:
            return True

        op1_oname = minlfs[0][0]
        op2_cands = max_candidates[op1_oname]
        for op2_oname in op2_cands:
            pair = (op1_oname, op2_oname)
            add_stack()
            if pair in match_cache or check_match(*pair):
                match_cache.add(pair)
                if match_outs(minlfs[1:]):
                    return True
            undo()
        return False

    if match_outs(leaf_min):
        return [
            nnmax.n2op[n2].name
            for n1, n2 in matched1.items()
            if n2 in nnmax.n2op and n1 in nnmin.n2op
        ]

    return None
