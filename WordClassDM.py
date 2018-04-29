"""
Last: 5069

Script with simple UI for creating gaplines data
Run: python WordClassDM.py --index 0
Controls:
    setting gaplines  - click and drag
    saving gaplines   - 's' key
    reseting gaplines - 'r' key
    skip to next img  - 'n' key
    delete last line  - 'd' key
"""

import cv2
import os
import numpy as np
import glob
import argparse
import simplejson
from ocr.normalization import imageNorm
from ocr.viz import printProgressBar


def loadImages(dataloc, idx=0, num=None):
    """ Load images and labels """
    print("Loading words...")

    # Load images and short them from the oldest to the newest
    imglist = glob.glob(os.path.join(dataloc, u'*.jpg'))
    imglist.sort(key=lambda x: float(x.split("_")[-1][:-4]))
    tmpLabels = [name[len(dataloc):] for name in imglist]

    labels = np.array(tmpLabels)
    images = np.empty(len(imglist), dtype=object)

    if num is None:
        upper = len(imglist)
    else:
        upper = min(idx + num, len(imglist))
        num += idx

    for i, img in enumerate(imglist):
        # TODO Speed up loading - Normalization
        if i >= idx and i < upper:
            images[i] = imageNorm(
                cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB),
                height=60,
                border=False,
                tilt=True,
                hystNorm=True)
            printProgressBar(i-idx, upper-idx-1)
    print()
    return (images[idx:num], labels[idx:num])


def locCheck(loc):
    return loc + '/' if loc[-1] != '/' else loc


class Cycler:
    drawing = False
    scaleF = 4

    def __init__(self, idx, data_loc, save_loc):
        """ Load images and starts from given index """
        # self.images, self.labels = loadImages(loc, idx)
        # Create save_loc directory if not exists
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
            
        self.data_loc = locCheck(data_loc)
        self.save_loc = locCheck(save_loc)
        
        self.idx = 0
        self.org_idx = idx

        self.blockLoad()
        self.image_act = self.images[self.idx]

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.mouseHandler)
        self.nextImage()

        self.run()

    def run(self):
        while(1):
            self.imageShow()
            k = cv2.waitKey(1) & 0xFF
            if k == ord('d'):
                # Delete last line
                self.deleteLastLine()
            elif k == ord('r'):
                # Clear current gaplines
                self.nextImage()
            elif k == ord('s'):
                # Save gaplines with image
                if self.saveData():
                    self.idx += 1
                    if self.idx >= len(self.images):
                        if not self.blockLoad():
                            break
                    self.nextImage()
            elif k == ord('n'):
                # Skip to next image
                self.idx += 1
                if self.idx >= len(self.images):
                    if not self.blockLoad():
                        break
                self.nextImage()
            elif k == 27:
                cv2.destroyAllWindows()
                break

        print("End of labeling at INDEX: " + str(self.org_idx + self.idx))

    def blockLoad(self):
        self.images, self.labels = loadImages(
            self.data_loc, self.org_idx + self.idx, 100)
        self.org_idx += self.idx
        self.idx = 0
        return len(self.images) is not 0

    def imageShow(self):
        cv2.imshow(
            'image',
            cv2.resize(
                self.image_act,
                (0,0),
                fx=self.scaleF,
                fy=self.scaleF,
                interpolation=cv2.INTERSECT_NONE))

    def nextImage(self):
        self.image_act = cv2.cvtColor(self.images[self.idx], cv2.COLOR_GRAY2RGB)
        self.label_act = self.labels[self.idx][:-4]
        self.gaplines =  [0, self.image_act.shape[1]]
        self.redrawLines()

        print(self.org_idx + self.idx, ":", self.label_act.split("_")[0])
        self.imageShow();

    def saveData(self):
        self.gaplines.sort()
        print("Saving image with gaplines: ", self.gaplines)

        try:
            assert len(self.gaplines) - 1 == len(self.label_act.split("_")[0])

            cv2.imwrite(
                    self.save_loc + '%s.jpg' % (self.label_act),
                    self.images[self.idx])
            with open(self.save_loc + '%s.txt' % (self.label_act), 'w') as fp:
                simplejson.dump(self.gaplines, fp)
            return True
        except:
            print("Wront number of gaplines")
            return False

        print()
        self.nextImage()

    def deleteLastLine(self):
        if len(self.gaplines) > 0:
            del self.gaplines[-1]
        self.redrawLines()

    def redrawLines(self):
        self.image_act = cv2.cvtColor(self.images[self.idx], cv2.COLOR_GRAY2RGB)
        for x in self.gaplines:
            self.drawLine(x)

    def drawLine(self, x):
        cv2.line(
            self.image_act, (x, 0), (x, self.image_act.shape[0]), (0,255,0), 1)

    def mouseHandler(self, event, x, y, flags, param):
        # Clip x into image width range
        x = max(min(self.image_act.shape[1], x // self.scaleF), 0)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.tmp = self.image_act.copy()
            self.drawLine(x)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.image_act = self.tmp.copy()
                self.drawLine(x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if x not in self.gaplines:
                self.gaplines.append(x)
                self.image_act = self.tmp.copy()
                self.drawLine(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            "Script creating UI for gaplines classification")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of starting image")
    
    parser.add_argument(
        "--data",
        type=str,
        default='data/words_raw',
        help="Path to folder with images")
    
    parser.add_argument(
        "--save",
        type=str,
        default='data/words2',
        help="Path to folder for saving images with gaplines")

    args = parser.parse_args()
    Cycler(args.index, args.data, args.save)
