#!/usr/bin/env python3
import os

from foolbox import foolbox
from robust_LVQ_models.utils.load_graph import load_graph


def create():
    filename = os.path.join(os.path.dirname(__file__), 'model_files/cnn.pb')
    graph = load_graph(filename)

    for op in graph.get_operations():
        print(str(op.name))

    input = graph.get_tensor_by_name('prefix/input_1:0')
    output = graph.get_tensor_by_name('prefix/dense_2/BiasAdd:0')

    fmodel = foolbox.models.TensorFlowModel(input, output, bounds=(0,1), preprocessing=(0, 255))
    return fmodel
