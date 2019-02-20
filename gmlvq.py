#!/usr/bin/env python3
import os

import tensorflow as tf

from foolbox import foolbox
from robust_LVQ_models.utils.load_graph import load_graph

# The GMLVQ model with one prototype per class.
# See the paper for more implementation details.

def get_graph_and_tensors():
    filename = os.path.join(os.path.dirname(__file__), 'model_files/gmlvq.pb')
    graph = load_graph(filename)

    input = graph.get_tensor_by_name('prefix/encoder_input:0')
    output = graph.get_tensor_by_name('prefix/lambda_1/Neg:0')

    return graph, input, output


def create():
    graph, input, output = get_graph_and_tensors()
    fmodel = foolbox.models.TensorFlowModel(input, output, bounds=(0, 255), preprocessing=(0, 255))
    return fmodel
