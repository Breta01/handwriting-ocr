# -*- coding: utf-8 -*-
"""
Loading and using trained models from tensorflow
"""
import tensorflow as tf

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