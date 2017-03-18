# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# Preloading trained model with activation function
# Loading is slow, prevent multiple loads
print("Loading tf model:")
sess = tf.Session()
saver = tf.train.import_meta_graph('models/CNN-CG.meta', clear_devices=True)
saver.restore(sess, 'models/CNN-CG')
activation = tf.get_collection("activation")[0]

def segmentation(img, slider = (30, 60), step = 2):
    """
    Take preprocessed image of word
    and return array of positions separating chars - gaps
    """
    gaps = []
    position = 0

    separate = False
    isPrevGap = True
    gapCount = 1
    gapPositionSum = position + slider[0] / 2

    while position < len(img[0]) - slider[0]:
        current = img[0:slider[1], position:position + slider[0]]
        # CharGapClassifier prediction
        # Pixel transform and rescale
        data = np.multiply(np.reshape(current, (1, 1800)).astype(np.float32),
                           1.0 / 255.0)

        if sess.run(activation, feed_dict={"x:0": data}) == 1:
            # If is GAP - add possition to sum
            gapPositionSum += position + slider[0] / 2
            gapCount += 1
            isPrevGap = True
            separate = False
        else:
            # Add gap position into array
            # only if two successive letter lines detected
            if not separate and isPrevGap:
                separate = True
            elif separate:
                gaps.append((int)(gapPositionSum / gapCount))
                gapPositionSum = 0
                gapCount = 0
                separate = False

            isPrevGap = False
        # Sliding forward
        position += step

    # Adding last line
    gapPositionSum += position + slider[0] / 2
    gapCount += 1
    gaps.append((int)(gapPositionSum / gapCount))
    # Drawing lines
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for gap in gaps:
        cv2.line(img,
                 ((int)(gap), 0),
                 ((int)(gap), slider[1]),
                 (0, 255, 0), 1)
    implt(img, t="Separated characters")
    return gaps


def implt(img, cmp=None, t=''):
    """ Show image using plt """
    plt.imshow(img, cmap=cmp)
    plt.title(t)
    plt.show()