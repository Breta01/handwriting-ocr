# -*- coding: utf-8 -*-
"""
Classes for controling machine learning processes
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import csv


class TrainingPlot:
    """
    Creating live plot during training
    REUIRES notebook backend: %matplotlib notebook
    @TODO Migrate to Tensorboard
    """
    trainLoss = []
    trainAcc = []
    validAcc = []
    testInterval = 0
    lossInterval = 0
    interval = 0
    ax1 = None
    ax2 = None
    fig = None

    def __init__(self, steps, testItr, lossItr):
        self.testInterval = testItr
        self.lossInterval = lossItr
        self.interval = steps

        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()
        self.ax1.set_autoscaley_on(True)
        plt.ion()

        self.updatePlot()

        # Description
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Train Loss')
        self.ax2.set_ylabel('Valid. Accuracy')

        # Axes limits
        self.ax1.set_ylim([0,10])


    def updatePlot(self):
        self.fig.canvas.draw()

    def updateCost(self, lossTrain, index):
        self.trainLoss.append(lossTrain)
        if len(self.trainLoss) == 1:
            self.ax1.set_ylim([0, min(10, math.ceil(lossTrain))])
        self.ax1.plot(self.lossInterval * np.arange(len(self.trainLoss)),
                      self.trainLoss, 'b', linewidth=1.0)

        self.updatePlot()

    def updateAcc(self, accVal, accTrain, index):
        self.validAcc.append(accVal)
        self.trainAcc.append(accTrain)

        self.ax2.plot(self.testInterval * np.arange(len(self.validAcc)),
                      self.validAcc, 'r', linewidth=1.0)
        self.ax2.plot(self.testInterval * np.arange(len(self.trainAcc)),
                      self.trainAcc, 'g',linewidth=1.0)

        self.ax2.set_title('Valid. Accuracy: {:.4f}'.format(self.validAcc[-1]))

        self.updatePlot()


class DataSet:
    """ Class for training data and feeding train function """
    images = None
    labels = None
    length = 0
    index = 0

    def __init__(self, img, lbl):
        """ Crate the dataset """
        self.images = img
        self.labels = lbl
        self.length = len(img)
        self.index = 0

    def next_batch(self, batchSize):
        """Return the next batch from the data set."""
        start = self.index
        self.index += batchSize

        if self.index > self.length:
            # Shuffle the data
            perm = np.arange(self.length)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index = batchSize


        end = self.index
        return self.images[start:end], self.labels[start:end]
