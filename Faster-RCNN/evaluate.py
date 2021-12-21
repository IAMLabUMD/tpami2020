#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Modified by Kyungjun Lee
# based on code from Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""
Evaluate script showing detections in test images.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import math
import xml.etree.ElementTree as ET

from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from lib.utils.util import get_idle_gpu
from sklearn.metrics import precision_recall_fscore_support


CLASSES = {
    'gtea': ('none',  # always index 0
            'box', 'plate', 'spoon', 'coffee', 'bag',
            'red cup', 'honey', 'ketchup', 'cheese', 'bread bag',
            'bread', 'hotdog', 'chocolate', 'mayonnaise',
            'sugar', 'cover', 'tea bag', 'sausage',
            'peanut', 'jam', 'water', 'mustard'),
    'gtea-gaze-plus': ('none',  # always index 0
                    'bacon', 'bottle', 'bowl', 'bread', 'cabinet', 'carrot',
                    'cereal', 'cheese', 'cheese bag', 'cream cheese', 'cup',
                    'egg', 'egg box', 'faucet', 'frying pan', 'glove', 'honey',
                    'jam', 'juice', 'ketchup', 'kettle', 'knife', 'mayonnaise',
                    'microwave', 'milk', 'mushroom', 'oil', 'onion', 'paper towel',
                    'pasta box', 'patty', 'peanut butter', 'pepper grinder', 'pizza',
                    'pizza bag', 'plastic bag', 'plate', 'pot', 'pot cover',
                    'refrigerator', 'salad dressing', 'salt', 'sausage', 'slice',
                    'spoon', 'stove', 'tea bag', 'tomato', 'turkey', 'vegetable',
                    'vinaigrette'),
    'tego': ('none',  # always index 0
            'airborne gum', 'baked cheetos', 'cheerios cereal',
            'diet coke bottle', 'extra dry skin moisturizer',
            'grill salt', 'hand soap', 'mountain dew can', 'oregano',
            'regular coke bottle', 'sprite bottle', 'aspirin',
            'baked lays chips', 'chicken soup can', 'dr pepper can', 
            'great grains cereal', 'hand sanitizer', 'mandarin can',
            'spf55 sunscreen'),
    'tor-feedback': ('none',  # always index 0
                    'baking soda', 'caramel coffee', 'cheetos', 'chewy bars',
                    'chicken broth', 'coca cola', 'diced tomatoes',
                    'diet coke', 'dill', 'fritos', 'lacroix apricot',
                    'lacroix mango', 'lays', 'oregano', 'pike place roast')
    }

NETS = {
  'vgg16': ('vgg16_faster_rcnn_iter_40000.ckpt','vgg16_faster_rcnn_iter_10000.ckpt',)
  }

DATASETS = {
  'gtea': ('gtea', 'test_GTEA_voc.txt',),
  'gtea-gaze-plus': ('gtea-gaze-plus', 'test_GTEA_GAZE_PLUS_voc.txt',),
  'tego': ('tego', 'test_TEgO_voc.txt',),
  'gtea_wholeBB': ('gtea', 'test_GTEA_voc_wholeBB.txt',),
  'gtea-gaze-plus_wholeBB': ('gtea-gaze-plus', 'test_GTEA_GAZE_PLUS_voc_wholeBB.txt',),
  'tego_wholeBB': ('tego', 'test_TEgO_voc_wholeBB.txt',),
  'tego_blind': ('tego', 'test_TEgO_blind_voc.txt',),
  'tego_sighted': ('tego', 'test_TEgO_sighted_voc.txt',),
  'tego_blind_wholeBB': ('tego', 'test_TEgO_blind_voc_wholeBB.txt',),
  'tego_sighted_wholeBB': ('tego', 'test_TEgO_sighted_voc_wholeBB.txt',),
  'tego_nohand': ('tego', 'test_TEgO_nohand_voc.txt',),
  'tego_nohand_wholeBB': ('tego', 'test_TEgO_nohand_voc_wholeBB.txt',),
  'tego_testing_nohand': ('tego', 'TEgO_testing_blind_nohand_voc.txt',),
  'tego_testing_nohand_wholeBB': ('tego', 'TEgO_testing_blind_nohand_voc_wholeBB.txt',),
  'tor-feedback': ('tor-feedback', 'test_TOR_feedback_voc.txt',),
  'tor-feedback_wholeBB': ('tor-feedback', 'test_TOR_feedback_voc_wholeBB.txt',),
  'tor-feedback_hand': ('tor-feedback', 'test_TOR_feedback+hand_voc.txt',),
  'tor-feedback_hand-wholeBB': ('tor-feedback', 'test_TOR_feedback+hand_voc_wholeBB.txt',),
  'tor-feedback_noshuffle': ('tor-feedback', 'test_TOR_feedback+noshuffle_voc.txt',),
  'tor-feedback_noshuffle-wholeBB': ('tor-feedback', 'test_TOR_feedback+noshuffle_voc_wholeBB.txt',),
  'tor-feedback_hand-noshuffle': ('tor-feedback', 'test_TOR_feedback+hand+noshuffle_voc.txt',),
  'tor-feedback_hand-noshuffle-wholeBB': ('tor-feedback', 'test_TOR_feedback+hand+noshuffle_voc_wholeBB.txt',),
  }


"""Parse input arguments."""
"""
import argparse

parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN evaluation')
parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                    choices=NETS.keys(), default='vgg16')
parser.add_argument('--dataset', dest='dataset', help='Trained dataset [gtea gtea-gaze-plus tego]',
                    choices=DATASETS.keys(), default='gtea')
args = parser.parse_args()
"""

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def vis_detections(im, bbox, score, class_name):
    """Draw detected bounding boxes."""
    #im = im[:, :, (2, 1, 0)]
    color = (0, 255, 0) # green
    thickness = 2
    # use opencv to draw a bounding box
    im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                    color, thickness)
    im = cv2.putText(im, '{:s} {:.3f}'.format(class_name, score),
                    (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    color, thickness, bottomLeftOrigin=False)
    return im


def get_ap(ious, threshold):
    precs = []
    for i, iou in enumerate(ious):
        if iou > threshold:
            prec_at = (len(precs) + 1) / (i + 1)
            precs.append(prec_at)
    
    return float(sum(precs) / len(precs)) if len(precs) > 0 else 0.


def calculate_map(ious, threshold=None):
    # if threshold == None, compute mAP
    # Otherwise, compute AP25, AP50, or AP75 according to the given threshold
    if threshold is None or threshold == "all":
        # do this
        thresholds = np.arange(0.5, 1.0, 0.05)
        aps = []
        for each in thresholds:
#             print(each)
            # calculate AP
            ap = get_ap(ious, each)
            aps.append(ap)
#         print(aps)
        ap = float(sum(aps) / len(aps))
    else:
        # AP25, AP50, or AP75
        # get AP
        ap = get_ap(ious, threshold)
    
    return ap


def compute_distance(p1, p2):
    return math.sqrt(((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2))


def get_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2


def get_area(bb):
    x1, y1, x2, y2 = bb
    w = np.maximum(0.0, x2 - x1 + 1)
    h = np.maximum(0.0, y2 - y1 + 1)
    return w * h


def compute_iou_and_iog(gt_bb, est_bb):
    # bb = (x1, y1, x2, y2)
    gt_area = get_area(gt_bb)
    est_area = get_area(est_bb)

    # Compute intersection
    xx1 = np.maximum(gt_bb[0], est_bb[0])
    yy1 = np.maximum(gt_bb[1], est_bb[1])
    xx2 = np.minimum(gt_bb[2], est_bb[2])
    yy2 = np.minimum(gt_bb[3], est_bb[3])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter_area = w * h

    return inter_area / (gt_area + est_area - inter_area), inter_area / gt_area


def keep_closest(dets, gt, thresh=0.5):
    inds = np.where(dets[:, 4] >= thresh)[0]
    if len(inds) == 0:
        return None, 0, 0
    elif len(inds) == 1:
        iou, iog = compute_iou_and_iog(gt["bbox"], dets[inds[0], :4])
        return dets[inds[0]], iou, iog
    else:
        print("more than one estimated bb, so choose one that's closest")
        
        gt_bbox = gt["bbox"]
        gt_center = get_center(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3])
        closest_i = -1
        closest_dist = -1
        for i in inds:
            bbox = dets[i, :4]
            est_center = get_center(bbox[0], bbox[1], bbox[2], bbox[3])
            dist = compute_distance(gt_center, est_center)
            if closest_dist == -1 or closest_dist > dist:
                closest_dist = dist
                closest_i = i
        
        iou, iog = compute_iou_and_iog(gt_bbox, dets[closest_i, :4])
        return dets[closest_i], iou, iog


def evaluate(sess, net, dataset, img_file, voc_file, output_dir, demo_dir):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(img_file)
    demo = im.copy()

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    #print(scores.shape, boxes.shape)
    #print(boxes)
    
    # Load the ground-truth bounding box
    # there is only one object of interest
    gt = parse_rec(voc_file)[0]
    #print(gt)
    # draw the gt bbox
    gt_bbox = gt["bbox"]
    gt_label = gt["name"]
    gt_color = (0, 0, 255)
    gt_thickness = 2
    """
    im = cv2.rectangle(im, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]),
                    gt_color, gt_thickness)
    im = cv2.putText(im, gt_label,
                    (gt_bbox[0], gt_bbox[3] - 15), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    gt_color, gt_thickness, bottomLeftOrigin=False)
    """

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    
    est_label = "None"
    ret_iou = 0
    ret_iog = 0
    whole_dets = None
    for cls_ind, cls in enumerate(CLASSES[DATASETS[dataset][0]][1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        cls_indices = np.array([cls_ind for i in range(len(cls_boxes))]).reshape(-1, 1)
        #print(cls_boxes, cls_scores)
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis],
                          cls_indices)).astype(np.float32)
        #print(dets)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        # draw the original estimated bbox(es)
        inds = np.where(dets[:, 4] >= CONF_THRESH)[0]
        for i in inds:
            bbox = dets[i, :4].astype(int)
            score = dets[i, 4]
            est_label = CLASSES[DATASETS[dataset][0]][int(dets[i, -1])]
            demo = vis_detections(demo, bbox, score, est_label)

        if whole_dets is None:
            whole_dets = dets
        else:
            whole_dets = np.append(whole_dets, dets, axis=0)
    
    if whole_dets is None:
        return gt_label, est_label, ret_iou, ret_iog

    # get only one that is closest to the bounding box
    #print(whole_dets)
    #print(whole_dets.shape)
    dets, ret_iou, ret_iog = keep_closest(whole_dets, gt, thresh=CONF_THRESH)
    if dets is not None: # no appropriately estimated label
        # do evaluation
        bbox = dets[:4].astype(int)
        score = dets[4]
        est_label = CLASSES[DATASETS[dataset][0]][int(dets[-1])]
        # visualization
        im = vis_detections(im, bbox, score, est_label)
    
    # write the debugging file here
    output_file = osp.join(output_dir, osp.basename(img_file))
    demo_file = osp.join(demo_dir, osp.basename(img_file))
    #plt.savefig(output_file)
    cv2.imwrite(output_file, im)
    cv2.imwrite(demo_file, demo)
    print("Output written to {} and {}".format(output_file, demo_file))
    
    return gt_label, est_label, ret_iou, ret_iog


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py [dataset]")
        sys.exit(0)
    
    dataset = str(sys.argv[1])
    print("Evaluating on {}".format(dataset))

    # make only one GPU visible
    gpu_to_use = get_idle_gpu(leave_unmasked=0)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    if gpu_to_use >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_to_use)
    else:
        # only using CPUs
        print("ERROR: no GPU to use for training")
        sys.exit()
    
    # model path
    demonet = "vgg16"
    #dataset = "tego"
    if "tego" in dataset:
        if "blind" in dataset or "sighted" in dataset:
            tfmodel = os.path.join('default', DATASETS[dataset][0], dataset, NETS[demonet][0])
        elif "wholeBB" in dataset:
            tfmodel = os.path.join('default', DATASETS[dataset][0], 'wholeBB', NETS[demonet][0])
        else:
            tfmodel = os.path.join('default', DATASETS[dataset][0], 'default', NETS[demonet][0])
    elif "tor-feedback" in dataset:
        if "hand" in dataset or "noshuffle" in dataset:
            tfmodel = os.path.join('default', DATASETS[dataset][0], dataset, NETS[demonet][0])
        elif "wholeBB" in dataset:
            tfmodel = os.path.join('default', DATASETS[dataset][0], 'wholeBB', NETS[demonet][0])
        else:
            tfmodel = os.path.join('default', DATASETS[dataset][0], 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    print("Loading a TF model from {}".format(tfmodel))
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError

    n_classes = len(CLASSES[DATASETS[dataset][0]])
    # create the structure of the net having a certain shape (which depends on the number of classes) 
    net.create_architecture(sess, "TEST", n_classes,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    data_dir = "data"
    testlist = DATASETS[dataset][1]
    img_files = []
    voc_files = []
    is_tor_feedback = "tor-feedback" in dataset
    print("Reading test files from {}".format(testlist))
    with open(osp.join(data_dir, testlist), 'r') as f:
        for line in f.readlines():
            # NOTE:
            # The list for TOR Feedback uses the tab character as a delimiter
            # The list for other datasets uses the space character as a delimiter
            img, voc = line.split() if not is_tor_feedback else line.split('\t')
            img_files.append(osp.join(data_dir, img.strip()))
            voc_files.append(osp.join(data_dir, voc.strip()))
    
    out_dir = "results"
    output_dir = osp.join(out_dir, dataset, testlist.split('.')[0])
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    
    demo_dir = "demos"
    demo_dir = osp.join(demo_dir, dataset, testlist.split('.')[0])
    if not osp.exists(demo_dir):
        os.makedirs(demo_dir)
    
    gt_labels = []
    est_labels = []
    ious = []
    iogs = []
    for img_file, voc_file in zip(img_files, voc_files):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Evaluate for {} with {}'.format(img_file, voc_file))
        assert osp.exists(img_file), "ERROR: {} NOT FOUND".format(img_file)
        assert osp.exists(voc_file), "ERROR: {} NOT FOUND".format(voc_file)
        gt_label, est_label, iou, iog = evaluate(sess, net, dataset, img_file,\
                    voc_file, output_dir, demo_dir)
        gt_labels.append(gt_label)
        est_labels.append(est_label)
        ious.append(iou)
        iogs.append(iog)
        #break
    
    # simple calculation
    prec, rec, f1, _ = precision_recall_fscore_support(gt_labels, est_labels, average="macro")
    miou = sum(ious) / len(ious)
    miog = sum(iogs) / len(iogs)
    # mAPs
    ap = calculate_map(ious, None)
    ap50 = calculate_map(ious, 0.5)
    ap75 = calculate_map(ious, 0.75)
    #print(gt_labels)
    #print(est_labels)
    #print(ious)
    print("==============================================================")
    print("Precision: {}, Recall: {}, F1: {}".format(prec, rec, f1))
    print("mIOU: {}, mIOG: {}".format(miou, miog))
    print("mAP: {}, AP50: {}, AP75: {}".format(ap, ap50, ap75))
    print("Done")