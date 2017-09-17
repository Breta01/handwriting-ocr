"""
Usage:
python graph_optimizer.py \
--tf_path ../../tensorflow/ \
--model_folder "path_to_the_model_folder" \
--output_names "activation, accuracy" \
--input_names "x"
"""

import os, argparse
from subprocess import call

import freeze_graph
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

fr_name = "_frozen.pb"
op_name = "_optimized.pb"


def graph_freez(model_folder, output_names):
    print("Model folder", model_folder)
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    print(checkpoint)
    checkpoint_path = checkpoint.model_checkpoint_path
    output_graph_filename = checkpoint_path + fr_name

    input_saver_def_path = ""
    input_binary = True
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


def graph_optimization(tf_path, graph_file, input_names, output_names):
    output_file = graph_file[:-len(fr_name)] + op_name
    tf_path += "bazel-bin/tensorflow/tools/graph_transforms/transform_graph"

    call([tf_path,
          "--in_graph=" + graph_file,
          "--out_graph=" + output_file,
          "--inputs=" + input_names,
          "--outputs=" + output_names,
          """--transforms=
          strip_unused_nodes(type=float, shape="1,299,299,3")
          fold_constants(ignore_errors=true)
          fold_batch_norms
          fold_old_batch_norms"""])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            "Script freezes graph and optimize it for mobile usage")
    parser.add_argument(
        "--model",
        type=str,
        help="Path of folder + model name (folder_path/model_name)")
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
    parser.add_argument(
        "--tf_path",
        type=str,
        default="../../tensorflow/",
        help="Path to the folder with tensorflow (requires bazel build of graph_transforms)")

    args = parser.parse_args()

    graph = graph_freez(args.model, args.output_names)
    graph_optimization(args.tf_path, graph, args.input_names, args.output_names)
