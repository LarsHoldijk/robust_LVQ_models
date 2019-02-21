#!/usr/bin/env python3
import foolbox
from robust_LVQ_models.models import cnn, glvq, glvq_large, gmlvq, gmlvq_large, gtlvq, gtlvq_large


def create(**kwargs):
    '''
    Instantiates a foolbox model for the required model
    :param model_name: name of the model that should be initiated
    :return: foolbox model
    '''
    if 'model_name' not in kwargs:
        raise Exception('Model_name parameter is required')
    model_name = kwargs['model_name']
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
    fmodel = foolbox.models.TensorFlowModel(input, output, bounds=(0, 255), preprocessing=(0, 255))
    return fmodel
