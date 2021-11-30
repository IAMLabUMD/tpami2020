#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
import json
from seg_utils import seg_utils as seg

import matplotlib.cm as cm
import tensorflow as tf
import time

import tensorvision
import tensorvision.utils as utils

from sklearn.metrics import precision_recall_fscore_support


def eval_image(hypes, gt_image, cnn_image):
    """."""
    thresh = np.array(range(0, 256))/255.0

    heatmap_color = np.array(hypes['data']['heatmap_color'])
    background_color = np.array(hypes['data']['background_color'])
    # calculate the probability of an object of interest location in each pixel
    gt_bg = np.all(gt_image == background_color, axis=2)
    gt_obj = np.any(gt_image != background_color, axis=2)

    valid_gt = gt_obj + gt_bg

    FN, FP, posNum, negNum = seg.evalExp(gt_obj, cnn_image,
                                         thresh, validMap=None,
                                         validArea=valid_gt)

    return FN, FP, posNum, negNum


def resize_label_image(image, gt_image, image_height, image_width):
    # image = scp.misc.imresize(image, size=(image_height, image_width),
    #                           interp='cubic')
    # gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
    #                              interp='nearest')
    image = cv2.resize(image, (image_height, image_width),
                        interpolation = cv2.INTER_CUBIC)
    gt_image = cv2.resize(gt_image, (image_height, image_width),
                            interpolation = cv2.INTER_NEAREST)

    return image, gt_image


def get_gt_class(hypes, data_dir, image_path):
    """
    Get a ground-truth object class (one-hot vector)

    Args
        hypes: hypes
        data_dir: data directory path 
        image_path: image file path

    Returns
        index: index of object class
    """
    # get a list of whole classes
    with open(os.path.join(data_dir, hypes['data']['label_file']), 'r') as f:
        classes = f.read().splitlines()
    # label jsons
    with open(os.path.join(data_dir, hypes['data']['train_label']), 'r') as f:
        train_label = json.load(f)
    with open(os.path.join(data_dir, hypes['data']['test_label']), 'r') as f:
        test_label = json.load(f)
    # get its object class
    paths = image_path.split('/')
    # e.g.) in-the-vanilla/B1_PS_NH_test/B1_PS_NH_test_000086.jpg
    if 'train' in image_path:
        cl_obj = train_label[paths[-3]][paths[-2]][paths[-1]]
    else:
        cl_obj = test_label[paths[-3]][paths[-2]][paths[-1]]
    # get the whole class
    one_hot_vec = np.zeros(len(classes))
    index = classes.index(cl_obj)
    # one_hot_vec[index] = 1

    return index


def pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh = None):
    '''

    @param totalPosNum: scalar
    @param totalNegNum: scalar
    @param totalFN: vector
    @param totalFP: vector
    @param thresh: vector
    '''

    #Calc missing stuff
    totalTP = totalPosNum - totalFN
    totalTN = totalNegNum - totalFP


    valid = (totalTP>=0) & (totalTN>=0)
    assert valid.all(), 'Detected invalid elements in eval'

    recall = totalTP / float( totalPosNum )
    TNR    = totalTN / float( totalNegNum )
    precision =  totalTP / (totalTP + totalFP + 1e-10)

    accuracy = (totalTP + totalTN) / (float( totalPosNum ) + float( totalNegNum ))
    
    selector_invalid = (recall==0) & (precision==0)
    recall = recall[~selector_invalid]
    precision = precision[~selector_invalid]
        
    maxValidIndex = len(precision)
    
    #Pascal VOC average precision
    AvgPrec = 0
    counter = 0
    for i in np.arange(0,1.1,0.1):
        ind = np.where(recall>=i)
        if ind == None:
            continue
        pmax = max(precision[ind]) if np.any(precision[ind]) else 0
        AvgPrec += pmax
        counter += 1
    AvgPrec = AvgPrec/counter
    
    
    # F-measure operation point
    beta = 1.0
    betasq = beta**2
    F = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    index = F.argmax()
    MaxF= F[index]
    
    recall_bst = recall[index]
    precision_bst =  precision[index]

    TP = totalTP[index]
    TN = totalTN[index]
    FP = totalFP[index]
    FN = totalFN[index]
    valuesMaxF = np.zeros((1,4),'u4')
    valuesMaxF[0,0] = TP
    valuesMaxF[0,1] = TN
    valuesMaxF[0,2] = FP
    valuesMaxF[0,3] = FN

    #ACC = (totalTP+ totalTN)/(totalPosNum+totalNegNum)
    prob_eval_scores  = calcEvalMeasures(valuesMaxF)
    prob_eval_scores['AvgPrec'] = AvgPrec
    prob_eval_scores['MaxF'] = MaxF
    prob_eval_scores['accuracy'] = accuracy

    #prob_eval_scores['totalFN'] = totalFN
    #prob_eval_scores['totalFP'] = totalFP
    prob_eval_scores['totalPosNum'] = totalPosNum
    prob_eval_scores['totalNegNum'] = totalNegNum

    prob_eval_scores['precision'] = precision
    prob_eval_scores['recall'] = recall
    prob_eval_scores['TNR'] = TNR
    #prob_eval_scores['precision_bst'] = precision_bst
    #prob_eval_scores['recall_bst'] = recall_bst
    prob_eval_scores['thresh'] = thresh
    if thresh is not None:
        BestThresh= thresh[index]
        prob_eval_scores['BestThresh'] = BestThresh

    #return a dict
    return prob_eval_scores


def calcEvalMeasures(evalDict, tag  = '_wp'):
    '''
    
    :param evalDict:
    :param tag:
    '''
    # array mode!
    TP = evalDict[:,0].astype('f4')
    TN = evalDict[:,1].astype('f4')
    FP = evalDict[:,2].astype('f4')
    FN = evalDict[:,3].astype('f4')
    Q = TP / (TP + FP + FN)
    P = TP + FN
    N = TN + FP
    TPR = TP / P
    FPR = FP / N
    FNR = FN / P
    TNR = TN / N
    A = (TP + TN) / (P + N)
    precision = TP / (TP + FP)
    recall = TP / P
    #numSamples = TP + TN + FP + FN
    correct_rate = A

    # F-measure
    #beta = 1.0
    #betasq = beta**2
    #F_max = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    
    
    outDict = dict()

    outDict['TP'+ tag] = TP
    outDict['FP'+ tag] = FP
    outDict['FN'+ tag] = FN
    outDict['TN'+ tag] = TN
    outDict['Q'+ tag] = Q
    outDict['A'+ tag] = A
    outDict['TPR'+ tag] = TPR
    outDict['FPR'+ tag] = FPR
    outDict['FNR'+ tag] = FNR
    outDict['PRE'+ tag] = precision
    outDict['REC'+ tag] = recall
    outDict['correct_rate'+ tag] = correct_rate
    return outDict


def evaluate(hypes, sess, image_pl, inf_out):

    out_layer = inf_out['output']
    recog_layer = inf_out['recog']
    data_dir = hypes['dirs']['data_dir']

    eval_dict = {}
    # for phase in ['train', 'val']:
    for phase in ['val']:
        data_file = hypes['data']['{}_file'.format(phase)]
        data_file = os.path.join(data_dir, data_file)
        image_dir = os.path.dirname(data_file)

        thresh = np.array(range(0, 256))/255.0
        total_fp = np.zeros(thresh.shape)
        total_fn = np.zeros(thresh.shape)
        total_posnum = 0
        total_negnum = 0

        image_list = []
        gt_class_list = []
        pred_class_list = []

        with open(data_file) as file:
            for i, datum in enumerate(file):
                datum = datum.rstrip()
                image_file, gt_file = datum.split("\t")
                image_file = os.path.join(image_dir, image_file)
                gt_file = os.path.join(image_dir, gt_file)

                # image = scp.misc.imread(image_file, mode='RGB')
                # gt_image = scp.misc.imread(gt_file, mode='RGB')
                image = cv2.imread(image_file, cv2.IMREAD_COLOR)
                gt_image = cv2.imread(gt_file, cv2.IMREAD_COLOR)
                gt_class = get_gt_class(hypes, data_dir, image_file)

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
                elif hypes['jitter']['resize_image']:
                    image_height = hypes['jitter']['image_height']
                    image_width = hypes['jitter']['image_width']
                    gt_image_old = gt_image
                    image, gt_image = resize_label_image(image, gt_image,
                                                         image_height,
                                                         image_width)
                    input_image = image
                else:
                    input_image = image

                shape = input_image.shape

                feed_dict = {image_pl: input_image}

                output = sess.run([out_layer, recog_layer], feed_dict=feed_dict)
                # getting an inference of object of interest location
                output_im = output[0][:, 1].reshape(shape[0], shape[1])
                pred_class = output[1]

                # add gt class and predicted class to the lists
                gt_class_list.append(gt_class)
                pred_class_list.append(pred_class)

                if hypes['jitter']['fix_shape']:
                    gt_shape = gt_image.shape
                    output_im = output_im[offset_x:offset_x+gt_shape[0],
                                          offset_y:offset_y+gt_shape[1]]
                if phase == 'val':
                    # Saving RB Plot
                    ov_image = seg.make_overlay(image, output_im)
                    name = os.path.basename(image_file)
                    image_list.append((name, ov_image))

                    name2 = name.split('.')[0] + '_blue.png'

                    hard = output_im > 0.5
                    blue_image = utils.fast_overlay(image, hard, \
                                    color=[0, 0, 255, 127])
                    image_list.append((name2, blue_image))

                    FN, FP, posNum, negNum = eval_image(hypes,
                                                        gt_image, output_im)
                    total_fp += FP
                    total_fn += FN
                    total_posnum += posNum
                    total_negnum += negNum

        eval_dict[phase] = pxEval_maximizeFMeasure(
                total_posnum, total_negnum, total_fn, total_fp, thresh=thresh)
        # calculate precision, recall, fscore, (and support) of classification
        ret = precision_recall_fscore_support(
                gt_class_list, pred_class_list, average='macro')
        eval_dict[phase]['RecogPrec'] = ret[0]
        eval_dict[phase]['RecogRec'] = ret[1]
        eval_dict[phase]['RecogF1'] = ret[2]

        if phase == 'val':
            start_time = time.time()
            for i in range(10):
                sess.run([out_layer], feed_dict=feed_dict)
            dt = (time.time() - start_time)/10

    eval_list = []

    if hypes['arch']['output'] != "regress":
        # for phase in ['train', 'val']:
        for phase in ['val']:
            eval_list.append(('[{}] Recognition F1'.format(phase),
                              100*eval_dict[phase]['RecogF1']))
            eval_list.append(('[{}] Recognition Precision'.format(phase),
                              100*eval_dict[phase]['RecogPrec']))
            eval_list.append(('[{}] Recognition Recall'.format(phase),
                              100*eval_dict[phase]['RecogRec']))

            eval_list.append(('[{}] MaxF1'.format(phase),
                              100*eval_dict[phase]['MaxF']))
            eval_list.append(('[{}] BestThresh'.format(phase),
                              100*eval_dict[phase]['BestThresh']))
            eval_list.append(('[{}] Average Precision'.format(phase),
                              100*eval_dict[phase]['AvgPrec']))

    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))

    return eval_list, image_list
