import os
import cv2
import json
import glob
import numpy as np
from random import randint
import uuid

#caffe_root = '/home/phani/dlpacks/caffe/'
#import sys
#sys.path.insert(0, caffe_root + 'python')

import caffe


def draw_label(im, true_label, pred_label):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (100, 100)
    fontScale              = 1
    fontColor              = (0, 0, 255)
    lineType               = 2
    cv2.putText(im, "true_label: " + true_label,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (100, 160)
    fontScale              = 1
    fontColor              = (0, 255, 0)
    lineType               = 2
    cv2.putText(im, "pred_label: " + pred_label,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)
    return im


caffe.set_mode_gpu()
pt = 'deploy.prototxt'
cm = 'snapshots/resnet-18_iter_1000.caffemodel'

nmodels = 1
image_resize = 224
nets = []
for i in range(nmodels):
    net = caffe.Net(pt, cm, caffe.TEST)
    net.blobs['data'].reshape(1,3,image_resize,image_resize)
    nets.append(net)

with open('labels.json', 'r') as fp:
    labels = json.load(fp)


import sys

fnames = glob.glob('data/nogo/*');

for fname in fnames:
        im = cv2.imread(fname)
        frame = im.copy()
	r, c = im.shape[:2]
        im = cv2.resize(im, (224, 224))
	im = np.rollaxis(im, 2)
	im = im.astype(np.float32) - 128.
	im = im[np.newaxis,:,:,:]
        ii = randint(0, nmodels-1)
        net = nets[ii]
	net.blobs['data'].data[...] = im
	prediction = net.forward()['prob'][0]
	index = prediction.argmax()
        
	conf = prediction[index]
        pred_label = labels[str(index)]
        true_label=''
        if conf>0.0:
           frame = draw_label(frame, true_label, pred_label)
        if pred_label.find('no')>=0:
            continue
        cv2.imshow('c', frame)
        cv2.waitKey(100)
