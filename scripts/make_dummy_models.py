#!/usr/bin/env python3
"""Create placeholder ONNX models with the exact file names and tensor shapes
the application expects.

These are NOT trained models - they exist so the full stack (backend + frontend
+ Docker) can be exercised before the real exports are downloaded:

    python scripts/make_dummy_models.py          # writes into ./models
    python scripts/make_dummy_models.py --force

Delete ``models/*.onnx`` and run ``scripts/download_models.py`` to swap in the
real weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = Path(__file__).resolve().parents[1]
SIZE = 128
OPSET = 17


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _conv(name, x, cin, cout, rng, k=3, pad=1, stride=1):
    w = numpy_helper.from_array(
        (rng.normal(0, 0.05, (cout, cin, k, k))).astype(np.float32), name + "_w"
    )
    node = helper.make_node("Conv", [x, w.name], [name], kernel_shape=[k, k],
                            pads=[pad, pad, pad, pad], strides=[stride, stride])
    return node, [name], [w]


def build_autoencoder(filename_seed: int, out_name: str, second_out: bool = False) -> onnx.ModelProto:
    """image [1,3,128,128] -> restored [1,3,128,128] (+ optional 4-wide vector)."""
    rng = _rng(filename_seed)
    nodes, inits, outs = [], [], []
    x = "image"

    n1, o1, i1 = _conv("enc1", x, 3, 8, rng)
    nodes += [n1, helper.make_node("Relu", o1, ["enc1_r"])]
    inits += i1
    cur = ["enc1_r"]
    n2, o2, i2 = _conv("enc2", cur[0], 8, 8, rng)
    nodes.append(n2)
    inits += i2
    cur = o2
    n3, o3, i3 = _conv("dec1", cur[0], 8, 3, rng)
    nodes.append(n3)
    inits += i3
    nodes.append(helper.make_node("Sigmoid", o3, [out_name]))
    outs.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, 3, SIZE, SIZE]))

    if second_out:
        pool = helper.make_node("GlobalAveragePool", ["enc1_r"], ["gap"])
        flat = helper.make_node("Flatten", ["gap"], ["feat"], axis=1)
        w = numpy_helper.from_array(rng.normal(0, 0.5, (4, 8)).astype(np.float32), "gate_w")
        b = numpy_helper.from_array(np.zeros(4, np.float32), "gate_b")
        gemm = helper.make_node("Gemm", ["feat", w.name, b.name], ["logits"],
                                alpha=1.0, beta=1.0, transB=1)
        nodes += [pool, flat, gemm]
        inits += [w, b]
        outs.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4]))

    graph = helper.make_graph(
        nodes, filename_seed.to_bytes(2, "big").hex(),
        [helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        outs, inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def build_classifier() -> onnx.ModelProto:
    rng = _rng(7)
    w1 = numpy_helper.from_array(rng.normal(0, 0.05, (8, 3, 3, 3)).astype(np.float32), "c1_w")
    conv = helper.make_node("Conv", ["image", w1.name], ["c1"], kernel_shape=[3, 3],
                            pads=[1, 1, 1, 1])
    relu = helper.make_node("Relu", ["c1"], ["c1_r"])
    pool = helper.make_node("GlobalAveragePool", ["c1_r"], ["gap"])
    flat = helper.make_node("Flatten", ["gap"], ["feat"], axis=1)
    gw = numpy_helper.from_array(rng.normal(0, 0.5, (4, 8)).astype(np.float32), "clf_w")
    gb = numpy_helper.from_array(np.zeros(4, np.float32), "clf_b")
    gemm = helper.make_node("Gemm", ["feat", gw.name, gb.name], ["logits"], transB=1)
    softmax = helper.make_node("Softmax", ["logits"], ["probs"], axis=1)
    graph = helper.make_graph(
        [conv, relu, pool, flat, gemm, softmax], "dummy_classifier",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 4])],
        [w1, gw, gb],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def build_generator() -> onnx.ModelProto:
    rng = _rng(11)
    w1 = numpy_helper.from_array(rng.normal(0, 0.05, (8, 3, 3, 3)).astype(np.float32), "g1_w")
    w2 = numpy_helper.from_array(rng.normal(0, 0.05, (8, 8, 3, 3)).astype(np.float32), "g2_w")
    w3 = numpy_helper.from_array(rng.normal(0, 0.05, (1, 8, 3, 3)).astype(np.float32), "g3_w")
    zero = numpy_helper.from_array(np.zeros(1, np.float32), "zero")

    nodes = [
        helper.make_node("Conv", ["photo", w1.name], ["g1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["g1"], ["g1_r"]),
        helper.make_node("Conv", ["g1_r", w2.name], ["g2"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["g2"], ["g2_r"]),
        helper.make_node("Conv", ["g2_r", w3.name], ["g3"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        # make the graph depend on the style condition (added as zero, so values are unaffected)
        helper.make_node("Cast", ["style"], ["style_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["style_f", zero.name], ["style_z"]),
        helper.make_node("Add", ["g3", "style_z"], ["g3_z"]),
        helper.make_node("Tanh", ["g3_z"], ["sketch"]),
    ]
    graph = helper.make_graph(
        nodes, "dummy_generator",
        [
            helper.make_tensor_value_info("photo", TensorProto.FLOAT, [1, 3, SIZE, SIZE]),
            helper.make_tensor_value_info("style", TensorProto.INT64, [1]),
        ],
        [helper.make_tensor_value_info("sketch", TensorProto.FLOAT, [1, 1, SIZE, SIZE])],
        [w1, w2, w3, zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


TARGETS = {
    "task1_universal_ae.onnx": lambda: build_autoencoder(1, "restored"),
    "task2_classifier.onnx": build_classifier,
    "task2_specialist_salt.onnx": lambda: build_autoencoder(2, "restored"),
    "task2_specialist_blur.onnx": lambda: build_autoencoder(3, "restored"),
    "task2_specialist_occlusion.onnx": lambda: build_autoencoder(4, "restored"),
    "task3_soft_moe.onnx": lambda: build_autoencoder(5, "restored", second_out=True),
    "task4_generator.onnx": build_generator,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=REPO / "models")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    for name, factory in TARGETS.items():
        path = args.dest / name
        if path.exists() and not args.force:
            print(f"  = {name}: exists (use --force to overwrite)")
            continue
        onnx.save(factory(), str(path))
        print(f"  + {name} ({path.stat().st_size / 1e3:.0f} kB, placeholder weights)")
    print("\nPlaceholder models written. Replace them with scripts/download_models.py before submitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
