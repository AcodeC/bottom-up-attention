#!/usr/bin/env python
import numpy as np
import h5py
from util import *

# Change dir to caffe root or prototxt database paths won't work wrong
import os
print os.getcwd()
os.chdir('..')
print os.getcwd()

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
sys.path.insert(0, './caffe/python/')
sys.path.insert(0, './lib/')
sys.path.insert(0, './tools/')

import caffe
print os.getcwd()
data_path = './data/genome/1600-400-20'

# Load classes
classes = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

# Load attributes
attributes = ['__no_attribute__']
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())

# Check object extraction
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs

GPU_ID = 0   # if we have multiple GPUs, pick one
caffe.set_device(GPU_ID)
caffe.set_mode_gpu()
# net = None
cfg_from_file('experiments/cfgs/faster_rcnn_end2end_resnet.yml')

weights = 'data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel'
prototxt = 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'

net = caffe.Net(prototxt, caffe.TEST, weights=weights)

#########################

# DON'T FORGET CHANGE PATH!!!
dataset = 'flickr8k'
rpath = '/home/liudq/ParserCaption'

data_path = os.path.join(rpath, 'data')
gt_file_path = os.path.join(data_path, dataset, 'gt.token')
img_path = os.path.join(data_path, dataset, 'Flicker8k_Dataset')
feat_file_path = os.path.join(data_path, dataset, 'new_feat2.h5')

VFEAT_DIM = 2048
SFEAT_DIM = 5
n_box = 36

print "Begin extract image features..."
img_names = get_img_names_list(gt_file_path, img_path)

# Warmup on a dummy image
im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
for i in xrange(20):
    _, _, _, _ = im_detect(net, im)

# extract image features one by one
for idx, im_name in enumerate(img_names[20:]):
    vfeat, sfeat = extract_fea(net, im_name)

    # save each feature to a h5 file
    h5_flie = h5py.File(feat_file_path, 'w')
    vfeat_dataset = h5_flie['vfeat']
    sfeat_dataset = h5_flie['sfeat']

    vfeat_dataset.resize([idx + 1, n_box, VFEAT_DIM])
    sfeat_dataset.resize([idx + 1, n_box, SFEAT_DIM])

    vfeat_dataset[idx] = vfeat
    sfeat_dataset[idx] = sfeat

    h5_flie.close()

    if idx % 100 == 0:
        print '{:d}/{:d}'.format(idx, len(img_names))
    # print '{:d}/{:d}'.format(idx, len(img_names))
