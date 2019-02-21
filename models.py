import os

import tensorflow as tf


def load_graph(frozen_graph_filename):
    '''
    Loads the tensorflow graph from a .pb file.
    :param frozen_graph_filename: Path to the .pb file
    :return: Tensorflow graph
    '''
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")

    return graph


def cnn():
    '''
    The baseline CNN with two convolutional and two fully connected layers.
    :return: Tensorflow graph, input tensor of graph and the graphs output tensor
    '''
    filename = os.path.join(os.path.dirname(__file__), 'model_files/cnn.pb')
    graph = load_graph(filename)

    input = graph.get_tensor_by_name('prefix/input_1:0')
    output = graph.get_tensor_by_name('prefix/dense_2/BiasAdd:0')

    return graph, input, output


def glvq():
    '''
    The GLVQ model with one prototype per class.
    :return: Tensorflow graph, input tensor of graph and the graphs output tensor
    '''
    filename = os.path.join(os.path.dirname(__file__), 'model_files/glvq.pb')
    graph = load_graph(filename)

    input = graph.get_tensor_by_name('prefix/encoder_input:0')
    output = graph.get_tensor_by_name('prefix/lambda_1/Neg:0')

    return graph, input, output


def glvq_large():
    '''
    The GLVQ model with 128 prototypes per class.
    :return: Tensorflow graph, input tensor of graph and the graphs output tensor
    '''
    filename = os.path.join(os.path.dirname(__file__), 'model_files/glvq_large.pb')
    graph = load_graph(filename)

    input = graph.get_tensor_by_name('prefix/encoder_input:0')
    output = graph.get_tensor_by_name('prefix/lambda_1/Neg:0')

    return graph, input, output


def gmlvq():
    '''
    The GMLVQ model with one prototype per class.
    :return: Tensorflow graph, input tensor of graph and the graphs output tensor
    '''
    filename = os.path.join(os.path.dirname(__file__), 'model_files/gmlvq.pb')
    graph = load_graph(filename)

    input = graph.get_tensor_by_name('prefix/encoder_input:0')
    output = graph.get_tensor_by_name('prefix/lambda_1/Neg:0')

    return graph, input, output


def gmlvq_large():
    '''
    The GMLVQ model with 49 prototypes per class.
    :return: Tensorflow graph, input tensor of graph and the graphs output tensor
    '''
    filename = os.path.join(os.path.dirname(__file__), 'model_files/gmlvq_large.pb')
    graph = load_graph(filename)

    input = graph.get_tensor_by_name('prefix/encoder_input:0')
    output = graph.get_tensor_by_name('prefix/lambda_1/Neg:0')

    return graph, input, output


def gtlvq():
    '''
    The GTLVQ model with one prototype for each class.
    :return: Tensorflow graph, input tensor of graph and the graphs output tensor
    '''
    filename = os.path.join(os.path.dirname(__file__), 'model_files/gtlvq.pb')
    graph = load_graph(filename)

    input = graph.get_tensor_by_name('prefix/encoder_input:0')
    output = graph.get_tensor_by_name('prefix/lambda_1/Neg:0')

    return graph, input, output


def gtlvq_large():
    '''
    The GTLVQ model with 10 prototypes per class.
    :return: Tensorflow graph, input tensor of graph and the graphs output tensor
    '''
    filename = os.path.join(os.path.dirname(__file__), 'model_files/gtlvq_large.pb')
    graph = load_graph(filename)

    input = graph.get_tensor_by_name('prefix/encoder_input:0')
    output = graph.get_tensor_by_name('prefix/lambda_1/Neg:0')

    return graph, input, output
