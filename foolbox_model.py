#!/usr/bin/env python3
import foolbox
from robust_LVQ_models.tensorflow_graph import get_tensorflow_graph


def create(model_name, large_model=False):
    '''
    Instantiates a foolbox model for the required model
    :param model_name: name of the model that should be initiated
    :param large_model: whether or not the model with more prototypes per class should be used
    :return: foolbox model
    '''

    graph, input, output = get_tensorflow_graph(model_name, large_model)
    fmodel = foolbox.models.TensorFlowModel(input, output, bounds=(0, 255), preprocessing=(0, 255))
    return fmodel
