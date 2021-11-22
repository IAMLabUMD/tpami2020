#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
from seg_utils import seg_utils as seg

import tensorflow as tf
import time

import tensorvision
import tensorvision.utils as utils


def eval_image(hypes, gt_image, cnn_image):
    """."""
    thresh = np.array(range(0, 256))/255.0

    gt_bg = gt_image[:, :, 0]
    gt_hand = gt_image[:, :, 1]
    gt_local = gt_image[:, :, 2]

    valid_gt = gt_bg | gt_hand

    FN, FP, posNum, negNum = seg.evalExp(gt_local, cnn_image,
                                         thresh, validMap=None,
                                         validArea=valid_gt)

    return FN, FP, posNum, negNum


def resize_label_image(image, hand_gt_image, local_gt_image,
                       image_height, image_width):
    image = cv2.resize(image, (image_width, image_height), \
                    interpolation = cv2.INTER_CUBIC)
    hand_gt_image = cv2.resize(hand_gt_image, (image_width, image_height), \
                        interpolation = cv2.INTER_NEAREST)
    local_gt_image = cv2.resize(local_gt_image, (image_width, image_height), \
                        interpolation = cv2.INTER_NEAREST)

    return image, hand_gt_image, local_gt_image


def evaluate(hypes, sess, image_pl, inf_out):

    hand_out_layer = inf_out['output']
    local_out_layer = inf_out['output2']
    data_dir = hypes['dirs']['data_dir']

    eval_dict = {}
    
    background_color = np.array(hypes['data']['background_color'])
    hand_color1 = np.array(hypes['data']['hand_color1'])
    hand_color2 = np.array(hypes['data']['hand_color2'])
    hand_color3 = np.array(hypes['data']['hand_color3'])
    hand_color4 = np.array(hypes['data']['hand_color4'])
    hand_color5 = np.array(hypes['data']['hand_color5'])
    heatmap_color = np.array(hypes['data']['heatmap_color'])

    for phase in ['train', 'val']:
        data_file = hypes['data']['{}_file'.format(phase)]
        data_file = os.path.join(data_dir, data_file)
        image_dir = os.path.dirname(data_file)

        thresh = np.array(range(0, 256))/255.0
        total_fp = np.zeros(thresh.shape)
        total_fn = np.zeros(thresh.shape)
        total_posnum = 0
        total_negnum = 0

        image_list = []

        with open(data_file) as file:
            for i, datum in enumerate(file):
                datum = datum.rstrip()
                image_file, local_gt_file = datum.split("\t")
                if "BigObjects" in local_gt_file:
                    hand_gt_file = local_gt_file.replace("/BigObjects/", "/Masks/", 1)
                else:
                    hand_gt_file = local_gt_file.replace("/Objects/", "/Masks/", 1)

                if "GTEA" in hand_gt_file:
                    hand_gt_file = hand_gt_file.replace(".jpg", ".png", 1)

                image_file = os.path.join(image_dir, image_file)
                hand_gt_file = os.path.join(image_dir, hand_gt_file)
                local_gt_file = os.path.join(image_dir, local_gt_file)

                # image = scp.misc.imread(image_file, mode='RGB')
                # gt_image = scp.misc.imread(gt_file, mode='RGB')
                image = cv2.imread(image_file, cv2.IMREAD_COLOR)
                hand_gt_image = cv2.imread(hand_gt_file, cv2.IMREAD_COLOR)
                local_gt_image = cv2.imread(local_gt_file, cv2.IMREAD_COLOR)

                if hypes['jitter']['resize_image']:
                    image_height = hypes['jitter']['image_height']
                    image_width = hypes['jitter']['image_width']
                    image, hand_gt_image, local_gt_image = resize_label_image(
                                                            image,
                                                            hand_gt_image,
                                                            local_gt_image,
                                                            image_height,
                                                            image_width)
                    input_image = image

                """
                1st output channel: background (non-hand)
                2nd output channel: hand
                3rd output channel: object-center
                """
                # background
                gt_bg = np.all(hand_gt_image == background_color, axis=2)
                # hand
                gt_hand1 = np.all(hand_gt_image == hand_color1, axis=2)
                gt_hand2 = np.all(hand_gt_image == hand_color2, axis=2)
                gt_hand3 = np.all(hand_gt_image == hand_color3, axis=2)
                gt_hand4 = np.all(hand_gt_image == hand_color4, axis=2)
                gt_hand5 = np.all(hand_gt_image == hand_color5, axis=2)
                # non-object 
                gt_no_obj = np.all(local_gt_image == background_color, axis=2)
                # object location heatmap
                gt_obj = np.any(local_gt_image != background_color, axis=2)
                
                assert(gt_obj.shape == gt_bg.shape)

                shape = gt_bg.shape
                gt_bg = gt_bg.reshape(shape[0], shape[1], 1)
                # hand masking
                gt_hand1 = gt_hand1.reshape(shape[0], shape[1], 1)
                gt_hand2 = gt_hand2.reshape(shape[0], shape[1], 1)
                gt_hand3 = gt_hand3.reshape(shape[0], shape[1], 1)
                gt_hand4 = gt_hand4.reshape(shape[0], shape[1], 1)
                gt_hand5 = gt_hand5.reshape(shape[0], shape[1], 1)
                # hands concatenation
                gt_hand = gt_hand1 | gt_hand2 | gt_hand3 | gt_hand4 | gt_hand5
                # object center masking
                gt_obj = gt_obj.reshape(shape[0], shape[1], 1)
                # gt_image concatenation
                gt_image = np.concatenate((gt_bg, gt_hand, gt_obj), axis=2)

                if hypes['jitter']['fix_shape']:
                    shape = image.shape
                    image_height = hypes['jitter']['image_height']
                    image_width = hypes['jitter']['image_width']
                    assert(image_height >= shape[0])
                    assert(image_width >= shape[1])

                    offset_x = (image_height - shape[0])//2
                    offset_y = (image_width - shape[1])//2
                    new_image = np.zeros([image_height, image_width, 3])
                    new_image[offset_x:offset_x+shape[0],
                              offset_y:offset_y+shape[1]] = image
                    input_image = new_image

                else:
                    input_image = image

                shape = input_image.shape

                feed_dict = {image_pl: input_image}

                hand, local = sess.run([hand_out_layer, local_out_layer],
                                        feed_dict=feed_dict)
                # getting a hand estimation
                output_hand = hand[:, 1].reshape(shape[0], shape[1])
                # getting an object localization
                output_local = local[:, 1].reshape(shape[0], shape[1])

                if hypes['jitter']['fix_shape']:
                    gt_shape = gt_image.shape
                    output_hand = output_hand[offset_x:offset_x+gt_shape[0],
                                              offset_y:offset_y+gt_shape[1]]
                    output_local = output_local[offset_x:offset_x+gt_shape[0],
                                                offset_y:offset_y+gt_shape[1]]

                if phase == 'val':
                    # Saving RB Plot
                    ov_hand_image = seg.make_overlay(image, output_hand)
                    ov_local_image = seg.make_overlay(image, output_local)
                    name = os.path.basename(image_file)
                    name_hand = name.split('.')[0] + '_hand.png'
                    name_local = name.split('.')[0] + '_local.png'
                    image_list.append((name_hand, ov_hand_image))
                    image_list.append((name_local, ov_local_image))

                    name2_hand = name_hand.split('.')[0] + '_blue.png'
                    name2_local = name_local.split('.')[0] + '_red.png'
                    hard_hand = output_hand > 0.5
                    hard_local = output_local > 0.5
                    hand_image = utils.fast_overlay(image, hard_hand, \
                                    color=[0, 0, 255, 127])
                    local_image = utils.fast_overlay(image, hard_local, \
                                    color=[255, 0, 0, 127])
                    image_list.append((name2_hand, hand_image))
                    image_list.append((name2_local, local_image))

                """
                FN, FP, posNum, negNum = eval_image(hypes, gt_image,
                                                    output_local)

                total_fp += FP
                total_fn += FN
                total_posnum += posNum
                total_negnum += negNum
                """

        # eval_dict[phase] = seg.pxEval_maximizeFMeasure(
        #     total_posnum, total_negnum, total_fn, total_fp, thresh=thresh)

        if phase == 'val':
            start_time = time.time()
            for i in range(10):
                sess.run([hand_out_layer, local_out_layer], feed_dict=feed_dict)
            dt = (time.time() - start_time)/10

    eval_list = []
    """
    for phase in ['train', 'val']:
        eval_list.append(('[{}] MaxF1'.format(phase),
                          100*eval_dict[phase]['MaxF']))
        eval_list.append(('[{}] BestThresh'.format(phase),
                          100*eval_dict[phase]['BestThresh']))
        eval_list.append(('[{}] Average Precision'.format(phase),
                          100*eval_dict[phase]['AvgPrec']))
    """

    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))

    return eval_list, image_list
