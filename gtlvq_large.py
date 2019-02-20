#!/usr/bin/env python3
import os

from foolbox import foolbox
from robust_LVQ_models.utils.load_graph import load_graph


def create():
    filename = os.path.join(os.path.dirname(__file__), 'model_files/gtlvq_large.pb')
    graph = load_graph(filename)

    input = graph.get_tensor_by_name('prefix/encoder_input:0')
    output = graph.get_tensor_by_name('prefix/lambda_1/Neg:0')

    fmodel = foolbox.models.TensorFlowModel(input, output, bounds=(0, 255), preprocessing=(0, 255))
    return fmodel
