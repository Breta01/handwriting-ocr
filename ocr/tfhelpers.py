#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loading and using trained models from tensorflow
"""
import tensorflow as tf

class Graph():
    """ Loading and running isolated tf graph """
    def __init__(self, loc):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
            saver.restore(self.sess, loc)
            self.activation = tf.get_collection('activation')[0]
    # To launch the graph
    def run(self, data):
        return self.sess.run(self.activation, feed_dict={"x:0": data})