# -*- coding: utf-8 -*-
"""
Provide functions and classes:
Graph       = Class for loading and using trained models from tensorflow
create_cell = function for creatting RNN cells with wrappers
"""
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMCell, ResidualWrapper, DropoutWrapper, MultiRNNCell

class Graph():
    """ Loading and running isolated tf graph """
    def __init__(self, loc, operation='activation', input_name='x'):
        """
        loc: location of file containing saved model
        operation: name of operation for running the model
        input_name: name of input placeholder
        """
        self.input = input_name + ":0"
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
            saver.restore(self.sess, loc)
            self.op = self.graph.get_operation_by_name(operation).outputs[0]

    def run(self, data):
        """ Run the specified operation on given data """
        return self.sess.run(self.op, feed_dict={self.input: data})
    
    def eval_feed(self, feed):
        """ Run the specified operation with given feed """
        return self.sess.run(self.op, feed_dict=feed)
        
    
    
def create_single_cell(cell_fn, num_units, is_residual=False, is_dropout=False, keep_prob=None):
    """ Create single RNN cell based on cell_fn"""
    cell = cell_fn(num_units)
    if is_dropout:
        cell = DropoutWrapper(cell, input_keep_prob=keep_prob)
    if is_residual:
        cell = ResidualWrapper(cell)
    return cell


def create_cell(num_units, num_layers, num_residual_layers, is_dropout=False, keep_prob=None, cell_fn=LSTMCell):
    """ Create corresponding number of RNN cells with given wrappers"""
    cell_list = []
    
    for i in range(num_layers):
        cell_list.append(create_single_cell(
            cell_fn=cell_fn,
            num_units=num_units,
            is_residual=(i >= num_layers - num_residual_layers),
            is_dropout=is_dropout,
            keep_prob=keep_prob
        ))

    if num_layers == 1:
        return cell_list[0]
    return MultiRNNCell(cell_list)