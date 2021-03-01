"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import tensorflow as tf

vers = (tf.__version__).split(".")
if int(vers[0]) == 2 or int(vers[0]) == 1 and int(vers[1]) > 12:
    tf = tf.compat.v1
else:
    tf = tf


def read_graph(file):
    """
    Loads the graph from a protobuf file

    Parameters
    -----------
    file : string
        path to the protobuf file

    Returns
    --------
    graph_def :class:`tensorflow.tf.compat.v1.GraphDef`
        The graph definition of the DeepLabCut model found at the object's path
    """

    with tf.io.gfile.GFile(file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def


def finalize_graph(graph_def):
    """
    Finalize the graph and get inputs to model

    Parameters
    -----------
    graph_def :class:`tensorflow.compat.v1.GraphDef`
        The graph of the DeepLabCut model, read using the :func:`read_graph` method

    Returns
    --------
    graph :class:`tensorflow.compat.v1.GraphDef`
        The finalized graph of the DeepLabCut model
    inputs :class:`tensorflow.Tensor`
        Input tensor(s) for the model
    """

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="DLC")
    graph.finalize()

    return graph


def get_output_nodes(graph):
    """
    Get the output node names from a graph

    Parameters
    -----------
    graph :class:`tensorflow.Graph`
        The graph of the DeepLabCut model

    Returns
    --------
    output : list
        the output node names as a list of strings
    """

    op_names = [str(op.name) for op in graph.get_operations()]
    if "concat_1" in op_names[-1]:
        output = [op_names[-1]]
    else:
        output = [op_names[-1], op_names[-2]]

    return output


def get_output_tensors(graph):
    """
    Get the names of the output tensors from a graph

    Parameters
    -----------
    graph :class:`tensorflow.Graph`
        The graph of the DeepLabCut model

    Returns
    --------
    output : list
        the output tensor names as a list of strings
    """

    output_nodes = get_output_nodes(graph)
    output_tensor = [out + ":0" for out in output_nodes]
    return output_tensor


def get_input_tensor(graph):

    input_tensor = str(graph.get_operations()[0].name) + ":0"
    return input_tensor


def extract_graph(graph, tf_config=None):
    """
    Initializes a tensorflow session with the specified graph and extracts the model's inputs and outputs

    Parameters
    -----------
    graph :class:`tensorflow.Graph`
        a tensorflow graph containing the desired model
    tf_config :class:`tensorflow.ConfigProto`

    Returns
    --------
    sess :class:`tensorflow.Session`
        a tensorflow session with the specified graph definition
    outputs :class:`tensorflow.Tensor`
        the output tensor(s) for the model
    """

    input_tensor = get_input_tensor(graph)
    output_tensor = get_output_tensors(graph)
    sess = tf.Session(graph=graph, config=tf_config)
    inputs = graph.get_tensor_by_name(input_tensor)
    outputs = [graph.get_tensor_by_name(out) for out in output_tensor]

    return sess, inputs, outputs
