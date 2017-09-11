"""
Usage:
python graph_optimizer.py \
--model folder/model_name \
--output_names "activation, accuracy" \
--input_names "x"
"""

import os, argparse

import freeze_graph
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib

dir = os.path.dirname(os.path.realpath(__file__))

fr_name = "_frozen.pb"
op_name = "_optimized.pb"


def graph_freez(model_folder, output_names):
    """
    # Retrieve checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    output_graph = input_checkpoint + fr_name

    output_node_names = output_names

    # Import the meta graph and retrieve a Saver
    clear_devices = True
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=clear_devices)

    # Retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # Start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        ) 

        with gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the frozen graph." % len(output_graph_def.node))

    return output_graph
    """
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    checkpoint_path = checkpoint.model_checkpoint_path
    output_graph_filename = checkpoint_path + fr_name
    print(checkpoint)
    
    input_saver_def_path = ""
    input_binary = True
#    checkpoint_path = model_folder
    output_node_names = output_names
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    clear_devices = False
    input_meta_graph = checkpoint_path + ".meta"

    freeze_graph.freeze_graph(
        "", input_saver_def_path, input_binary, checkpoint_path,
        output_node_names, restore_op_name, filename_tensor_name,
        output_graph_filename, clear_devices, "", "", input_meta_graph)
    
    return output_graph_filename
  

def graph_optimization(graph_file, input_names, output_names):
    output_file = graph_file[:-len(fr_name)] + op_name

    input_graph_def = tf.GraphDef()
    with gfile.Open(graph_file, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        input_names.split(","),
        output_names.split(","),
        tf.float32.as_datatype_enum)

    with gfile.FastGFile(output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the optimized graph." % len(output_graph_def.node))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_folder",
        type=str,
        help="Path of folder with model to freez")
    parser.add_argument(
        "--input_names",
        type=str,
        default="",
        help="Input node names, comma separated.")
    parser.add_argument(
        "--output_names",
        type=str,
        default="",
        help="Output node names, comma separated.")
    args = parser.parse_args()

    graph = graph_freez(args.model_folder, args.output_names)
    graph_optimization(graph, args.input_names, args.output_names)
