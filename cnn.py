#!/usr/bin/env python3
import tensorflow as tf

from foolbox import foolbox
from robust_LVQ_models.utils.load_graph import load_graph


def create():
    graph = load_graph('./model_files/cnn.pb')

    for op in graph.get_operations():
        print(str(op.name))

    input = graph.get_tensor_by_name('prefix/input_1:0')
    output = graph.get_tensor_by_name('prefix/vae/dense_2/Softmax:0')

    fmodel = foolbox.models.TensorFlowModel(input, output, bounds=(0,1), preprocessing=(0, 255))
    return fmodel
