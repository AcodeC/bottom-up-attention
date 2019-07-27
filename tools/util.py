import cv2
import os
import numpy as np

import _init_paths
import caffe
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms


def _boxes2sfeat(boxes, im):
    S_H = im.shape[0]
    S_W = im.shape[1]
    S_A = S_W * S_H

    boxes = np.asarray(boxes)
    # calculate sfeat
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    Sa = (x2 - x1) * (y2 - y1)
    sfeat = np.hstack(((x1/S_W)[:, np.newaxis],
                       (y1/S_H)[:, np.newaxis],
                       (x2/S_W)[:, np.newaxis],
                       (y2/S_H)[:, np.newaxis],
                       (Sa/S_A)[:, np.newaxis]))
    return sfeat


def get_img_names_list(gt_file_path, img_path):
    img_names = []

    gt_file = open(gt_file_path)

    for line in gt_file:
        # split, isfind, imgid, filename, raw, parser_gt = line.split('\t')
        filename = line.split('\t')[3]

        if filename[-2:] == '#0':
            img_names.append(os.path.join(img_path, filename[:-2]))

    return img_names


def extract_fea(net, im_file, conf_thresh=0.4, min_boxes=36, max_boxes=36):
    im = cv2.imread(im_file)

    scores, boxes, _, _ = im_detect(net, im)

    # Keep the original boxes, don't worry about the regression bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    # attr_prob = net.blobs['attr_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf)[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf)[::-1][:max_boxes]
    ############################

    boxes = cls_boxes[keep_boxes]
    # objects = np.argmax(cls_prob[keep_boxes][:,1:], axis=1)
    # attr_thresh = 0.1
    # attr = np.argmax(attr_prob[keep_boxes][:,1:], axis=1)
    # attr_conf = np.max(attr_prob[keep_boxes][:,1:], axis=1)

    vfeat = np.asarray(pool5[keep_boxes])
    sfeat = _boxes2sfeat(boxes, im)

    return vfeat, sfeat
