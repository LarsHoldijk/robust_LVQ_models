#!/usr/bin/env python3

from robust_LVQ_models.models import cnn, glvq, glvq_large, gmlvq, gmlvq_large, gtlvq, gtlvq_large


def get_tensorflow_graph(model_name):
    '''
    Instantiates a tensorflow graph for the required model
    :param model_name: name of the model that should be initiated
    :return: foolbox graph, input tensor and output tensor of the graph
    '''
    if model_name == 'cnn':
        graph, input, output = cnn()
    elif model_name == 'glvq':
        graph, input, output = glvq()
    elif model_name == 'glvq_large':
        graph, input, output = glvq_large()
    elif model_name == 'gmlvq':
        graph, input, output = gmlvq()
    elif model_name == 'gmlvq_large':
        graph, input, output = gmlvq_large()
    elif model_name == 'gtlvq':
        graph, input, output = gtlvq()
    elif model_name == 'gtlvq_large':
        graph, input, output = gtlvq_large()
    else:
        raise Exception('Model_name unknown')
    return graph, input, output
