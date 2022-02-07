"""
objecr recognizer based on the pose an location of hand(s)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import argparse

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import csv
import cv2
import json
import tensorflow as tf
import xml.etree.ElementTree as ET

from fractions import Fraction
from scipy.ndimage import measurements
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

import gpu_utils

# configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO)

sys.path.insert(1, 'incl')
DEBUG = True

# from seg_utils import seg_utils as seg

try:
  # Check whether setup was done correctly
  import tensorvision.utils as tv_utils
  import tensorvision.core as core
except ImportError:
  # You forgot to initialize submodules
  logging.error("Could not import the submodules.")
  logging.error("Please execute:"
                "'git submodule update --init --recursive'")
  exit(1)


red_color = [0, 0, 255, 127]
green_color = [0, 255, 0, 127]
blue_color = [255, 0, 0, 127]
gb_color = [255, 255, 0, 127]

parser = None
args = None


def load_model(model_dir, hypes, modules, image_width=450, image_height=450):
  # set to allocate memory on GPU as needed
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  # Create tf graph and build module.
  with tf.Graph().as_default():
    # with tf.device('/cpu:0'):
    # Create placeholder for input
    num_channels = hypes['arch']['num_channels']
    input_pl = tf.placeholder(tf.float32, [None, None, num_channels])
    image = tf.expand_dims(input_pl, 0)
    # set the pre-defined image size here
    image.set_shape([1, image_height, image_width, num_channels])

    # build Tensorflow graph using the model from logdir
    output_operation = core.build_inference_graph(
        hypes, modules, image=image)
    logging.info("Graph build successfully.")

    # Create a session for running Ops on the Graph.
    sess = tf.Session(config=config)
    # self.sess = tf.Session()
    saver = tf.train.Saver()
    # Load weights from logdir
    core.load_weights(model_dir, sess, saver)
    logging.info("Weights loaded successfully.")
    
    model = {}
    model["in"] = input_pl
    model["out"] = output_operation
    model["sess"] = sess
    return model


def get_class_name_with_index(classes, index):
  """ Return an object name associated with a given index

  Args
    classes: a list of class labels
    index: an estimated class index

  Returns
    the class name associated with the index
  """
  return classes[index]


def get_class_index(classes, name):
  """ Return a class index associated with a given name

  Args
    classes: a list of class labels
    name: a class label
  
  Returns
    the class index associated with the name
  """
  return classes.index[name]


def get_gt_class_from_voc(annot_path):
  """ Parse a PASCAL VOC xml file
  
  Args
      annot_file: a path of a VOC-format annotation file

  Returns
      the class label (string)
  """
  tree = ET.parse(annot_path)
  for obj in tree.findall("object"):
    # there should be only one object in the TEgO feedback dataset
    # so we just return the first one
    return obj.find("name").text.lower()


def get_gt_local(hm, bg_color=(0,0,0)):
  """ Return a ground-truth object localization

  Args
    hm: numpy array of ground-truth center heatmap image
    bg_color: background color (i.e., non-hand pixel color)
  
  Return
    numpy array in boolean
  """
  gt_local = np.any(hm != bg_color, axis=2)
  return gt_local


def estimate(model, image, model_type, local_threshold):
  """ Localize and recognize an object of interest in an image

  Args
    image: image nparray

  Returns
    
  """
  shape = image.shape
  if DEBUG:
    print("DEBUG: image shape = (", str(shape[0]), str(shape[1]), ")")

  # start localizing an object of interest 
  feed = {model["in"]: image}
  if model_type == "multitask":
    # output1 is for hand segmentation in this multi-task model
    local_pred = model["out"]['output2']
  else:
    local_pred = model["out"]['output']
  # for recognition (classification)
  class_pred = model["out"]['recog']
  pred = [local_pred, class_pred]
  
  logging.info("Running recognizer")
  output = model["sess"].run(pred, feed_dict=feed)

  #print("max prob: ", max(output[0]))
  # get an estimated localization
  if model_type == "multiclass":
    # localization should be at 2 on axis=2
    est_local = output[0][:, 2]
  else:
    est_local = output[0][:, 1]
  # reshape output from flat vector to 2D Image
  est_local = est_local.reshape(shape[0], shape[1])
  # get an estimated class
  est_class_idx = np.asscalar(output[1])

  # Accept all pixel with conf >= 0.5 as positive prediction
  # This creates a `hard` prediction result for localization
  logging.info("Given threshold value: %f" % local_threshold)
  logging.info("Max threshold value in localization: %f" % np.max(est_local))
  est_local = est_local > local_threshold

  return est_local, est_class_idx


def get_est(iou, threshold):
  return 1 if iou > threshold else 0


def get_ap(ious, threshold):
  precs = []
  for i, iou in enumerate(ious):
    res = get_est(iou, threshold)
    if res:
      prec_at = (len(precs) + 1) / (i + 1)
      precs.append(prec_at)
  
  return float(sum(precs) / len(precs)) if len(precs) > 0 else 0.


def get_biggest(pred):
  """ Get the biggest cluster if there are more than one cluster
      in the prediction

  Args
    prediction: (numpy array)

  Returns
    biggest prediction indices (numpy array)
  """
  # get labeled clusters
  labeled, num_features = measurements.label(pred)
  if num_features <= 1:
    return pred
  # calculate the size of each cluster
  area = measurements.sum(pred, labeled, index=range(num_features + 1))
  c_idx = np.argmax(area)
  biggest = np.where(labeled == c_idx, pred, 0)
  
  assert(biggest.shape == pred.shape)
  
  return biggest


def calculate_map(ious, threshold=None):
  # if threshold == None, compute mAP
  # Otherwise, compute AP25, AP50, or AP75 according to the given threshold
  if threshold is None or threshold == "all":
    # do this
    thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for each in thresholds:
      # calculate AP
      ap = get_ap(ious, each)
      aps.append(ap)
    ap = float(sum(aps) / len(aps))
  else:
    # AP25, AP50, or AP75
    # get AP
    ap = get_ap(ious, threshold)
  
  return ap


def calculate_iou(gt, pred):
  """ Calculate Intersection Over Union (IOU) score

  Args
    gt: (numpy array)
    pred: (numpy array)
  
  Returns
    IOU score (float)
  """
  if not np.any(pred):
    return 0.

  pred = get_biggest(pred)
  intersection = np.logical_and(gt, pred)
  union = np.logical_or(gt, pred)

  return np.sum(intersection) / np.sum(union)


def do_hand_analysis(ious, name):
  """ Perform hand analysis
  @Args
    ious(numpy array):
    name(string):
  
  @Return

  """
  if ious is None or ious.size == 0:
    logging.error("WRONG DATA FOR ANALYSIS")
    return

  miou = float(sum(ious) / len(ious))
  ap = calculate_map(ious, None)
  ap50 = calculate_map(ious, 0.5)
  ap75 = calculate_map(ious, 0.75)
  
  print("IOU: mIOU={}, AP={}, AP50={}, AP75={}".\
      format(round(miou, 2), round(ap, 2), round(ap50, 2), round(ap75, 2)))
  return [name, str(miou), str(ap), str(ap50), str(ap75)]


def do_local_analysis(ious):
  """ Perform local analysis
  
  Args
    ious: a list of IOUs
  
  Return
    (miou, ap, ap50, ap75)
  """
  if ious is None or ious.size == 0:
    logging.error("WRONG DATA FOR ANALYSIS")
    return

  miou = float(sum(ious) / len(ious))
  ap = calculate_map(ious, None)
  ap50 = calculate_map(ious, 0.5)
  ap75 = calculate_map(ious, 0.75)
  
  print("IOU: mIOU={}, AP={}, AP50={}, AP75={}".\
      format(round(miou, 2), round(ap, 2), round(ap50, 2), round(ap75, 2)))
  return (miou, ap, ap50, ap75)


def do_class_analysis(gt_classes, pred_classes):
  """ Perform classification analysis

  """
  if gt_classes is None or pred_classes is None:
    logging.error("Wrong data for classification analysis")
    return

  prec, recall, f1, _ = precision_recall_fscore_support(
      gt_classes, pred_classes, average='macro')
  print("Classification: F1={}, Precision={}, Recall={}".\
      format(round(f1, 2), round(prec, 2), round(recall, 2)))
  return (f1, prec, recall)


"""
main function
"""
def main(args):
  model_dir = args.model
  threshold = args.threshold
  width = args.width
  height = args.height
  test_file = args.input_file
  model_type = args.model_type

  if not model_dir or not test_file or not model_type:
    parser.print_help()
    return
  elif not os.path.exists(test_file):
    print("ERROR: failed to locate ", test_file)
    return
  
  if not width or not height:
    # default setting
    width = height = 450
  else:
    width = int(width)
    height = int(height)
  
  if not threshold:
    # default setting
    threshold = 0.5
  else:
    threshold = float(threshold)

  if DEBUG:
    print("DEBUG: model:", model_dir, ", threshold:", threshold, \
          ", width:", width, ", height:", height)

  hypes = tv_utils.load_hypes_from_logdir(model_dir, base_path='hypes')
  logging.info("Hypes loaded successfully.")
  # load modules from the model dir
  modules = tv_utils.load_modules_from_logdir(model_dir)
  logging.info("Modules loaded successfully. Starting to build tf graph.")

  # load a list of classes
  data_dir = hypes['dirs']['data_dir']
  with open(os.path.join(data_dir, hypes['data']['label_file']), 'r') as f:
    classes = f.read().splitlines()

  # load model here
  model = load_model(model_dir, hypes, modules, width, height)

  # create output dir
  output_dir = os.path.join(
      "outputs", os.path.basename(model_dir),
      os.path.basename(test_file).split('.')[0],
      str(threshold).replace('.', ''))
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  local_ious = []
  gt_classes = []
  est_classes = []
  # load a list of testing images
  with open(test_file, 'r') as f:
    for line in f:
      # each line has three components separated by the tab character
      # - original image path
      # - its center heatmap path
      # - its VOC-format annotation path
      image_path, gt_local_path = line.rstrip().split('\t')
      image_file = os.path.join(data_dir, image_path)
      gt_local_file = os.path.join(data_dir, gt_local_path)
      voc_annot_path = gt_local_path.replace(
          "Objects", "ObjectsVOC").replace(".jpg", ".xml")
      voc_annot_file = os.path.join(data_dir, voc_annot_path)

      # get ground-truth label
      gt_class_label = get_gt_class_from_voc(voc_annot_file)
      
      # read the image
      image = cv2.imread(image_file, cv2.IMREAD_COLOR)
      # resize the input image
      image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

      # estimate
      est_local, est_class_idx = estimate(model, image, model_type, threshold)
      # get the class name from index
      est_class_label = get_class_name_with_index(classes, est_class_idx)

      # calculate and store IOU of the localization
      # read ground-truth center heatmap image
      gt_local_image = cv2.imread(gt_local_file, cv2.IMREAD_COLOR)
      # resize this image as well
      gt_local_image = cv2.resize(gt_local_image, (width, height))
      # get ground-truth localization
      bg_color = hypes["data"]["background_color"]
      gt_local = get_gt_local(gt_local_image, bg_color)
      # calculate IOU
      local_iou = calculate_iou(gt_local, est_local)
      # store IOU
      local_ious.append(local_iou)

      # store ground-truth and estimated class labels 
      gt_classes.append(gt_class_label)
      est_classes.append(est_class_label)

    # do analysis here
    cl_analysis = do_class_analysis(gt_classes, est_classes)
    if "nohandlocal" not in model_type:
      local_ious = np.array(local_ious)
      local_analysis = do_local_analysis(local_ious)
    else:
      local_analysis = None

    # then store the results
    csv_name = "results.csv"
    csv_path = os.path.join(output_dir, csv_name)
    with open(csv_path, 'w') as f:
      writer = csv.writer(f, dialect='excel')
      # classification result
      writer.writerow(["F1", "Precision", "Recall"])
      writer.writerow(cl_analysis)
      if local_analysis is not None:
        # localization result if available
        writer.writerow(["mIOU", "AP", "AP50", "AP75"])
        writer.writerow(local_analysis)  
    
    print("Evaluation done for {} with {}".format(model_dir, test_file))

    return


if __name__ == '__main__':
  # global parser, args
  # allocating TF graph to only one GPU
  #https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter
  # os.environ["CUDA_VISIBLE_DEVICES"]="0"
  gpu_to_use = gpu_utils.get_idle_gpu(leave_unmasked=0)
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
  if gpu_to_use >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_to_use)
  else:
    # only using CPUs
    print("WARNING: no GPU to use for training")
    os.environ["CUDA_VISIBLE_DEVICES"]=""

  parser = argparse.ArgumentParser()
  parser.add_argument("--model", help="recognizer model")
  parser.add_argument("--threshold", help="threshold for the model")
  parser.add_argument("--width", help="image width")
  parser.add_argument("--height", help="image height")
  parser.add_argument("--input_file", help="file containing a list of images")
  parser.add_argument("--model_type", help="[fientune|multitask|multiclass|handprimed]")

  args = parser.parse_args()
  
  main(args)
