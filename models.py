import os

import tensorflow as tf

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")

    return graph

def cnn():
    filename = os.path.join(os.path.dirname(__file__), 'model_files/cnn.pb')
    graph = load_graph(filename)

    input = graph.get_tensor_by_name('prefix/input_1:0')
    output = graph.get_tensor_by_name('prefix/dense_2/BiasAdd:0')

    return graph, input, output

def glvq():
    pass

def glvq_large():
    pass

def gmlvq():
    pass

def gmlvq_large():
    pass

def gtlvq():
    pass

def gtvlq_large():
    pass