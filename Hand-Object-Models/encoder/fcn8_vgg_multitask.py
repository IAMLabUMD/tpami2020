"""
Utilize vgg_fcn8 as encoder.
------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from arch import fcn8_vgg_multitask

import tensorflow as tf

import os


def inference(hypes, images, train=True):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    vgg16_npy_path = os.path.join(hypes['dirs']['data_dir'], 'weights',
                                  "vgg16.npy")
    # vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path=vgg16_npy_path)
    vgg_fcn = fcn8_vgg_multitask.FCN8VGG(vgg16_npy_path=vgg16_npy_path)

    vgg_fcn.wd = hypes['wd']

    # num_classes = hypes['arch']['num_classes']
    
    is_recognition = "recog" in hypes['model']['input_file']
    is_shallow = False
    try:
        is_shallow = "shallow" == hypes['arch']['recog_depth']
    except KeyError:
        is_shallow = False

    if is_recognition:
        num_obj_classes = hypes['arch']['num_obj_classes']
    else:
        num_obj_classes = None

    vgg_fcn.build(images, train=train, num_classes=2, random_init_fc8=True,
                  num_obj_classes=num_obj_classes, is_recognition=is_recognition,
                  is_shallow=is_shallow)

    logits = {}

    logits['images'] = images

    if hypes['arch']['fcn_in'] == 'pool5':
        logits['fcn_in'] = vgg_fcn.pool5
    elif hypes['arch']['fcn_in'] == 'fc7':
        logits['fcn_in'] = vgg_fcn.fc7
    else:
        raise NotImplementedError

    logits['feed2'] = vgg_fcn.pool4
    logits['feed4'] = vgg_fcn.pool3

    logits['fcn_logits'] = vgg_fcn.upscore32
    logits['fcn_logits2'] = vgg_fcn.upscore

    logits['deep_feat'] = vgg_fcn.pool5
    logits['early_feat'] = vgg_fcn.conv4_3

    if is_recognition:
        logits['pred'] = vgg_fcn.pred

    # List what variables to save and restore for finetuning
    """
    vars_to_save = {"conv1_1": vgg_fcn.conv1_1, "conv1_2": vgg_fcn.conv1_2,
                    "pool1": vgg_fcn.pool1,
                    "conv2_1": vgg_fcn.conv2_1, "conv2_2": vgg_fcn.conv2_2,
                    "pool2": vgg_fcn.pool2,
                    "conv3_1": vgg_fcn.conv3_1, "conv3_2": vgg_fcn.conv3_2,
                    "conv3_3": vgg_fcn.conv3_3, "pool3": vgg_fcn.pool3,
                    "conv4_1": vgg_fcn.conv4_1, "conv4_2": vgg_fcn.conv4_2,
                    "conv4_3":  vgg_fcn.conv4_3, "pool4": vgg_fcn.pool4,
                    "conv5_1": vgg_fcn.conv5_1, "conv5_2": vgg_fcn.conv5_2,
                    "conv5_3": vgg_fcn.conv5_3, "pool5": vgg_fcn.pool5,
                    "fc6": vgg_fcn.fc6, "fc7": vgg_fcn.fc7}
    """
    vars_to_save = (vgg_fcn.conv1_1, vgg_fcn.conv1_2,
                    vgg_fcn.conv2_1, vgg_fcn.conv2_2,
                    vgg_fcn.conv3_1, vgg_fcn.conv3_2, vgg_fcn.conv3_3,
                    vgg_fcn.conv4_1, vgg_fcn.conv4_2, vgg_fcn.conv4_3,
                    vgg_fcn.conv5_1, vgg_fcn.conv5_2, vgg_fcn.conv5_3,
                    vgg_fcn.fc6, vgg_fcn.fc7)

    logits['saving_vars'] = vars_to_save

    return logits
