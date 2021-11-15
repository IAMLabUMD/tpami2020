"""
Fine-tune the TOR's hand segmentation model
to localize an object of interest from an image.

-------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
Modified by Kyungjun Lee

More details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import commentjson
import logging
import os
import sys

import collections
import gpu_utils


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.iteritems():
        if (k in dct and isinstance(dct[k], dict) and
                isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

import tensorvision.train as train
import tensorvision.utils as utils

flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('hypes', None,
                    'File storing model parameters.')

flags.DEFINE_string('mod', None,
                    'Modifier for model parameters.')

if 'TV_SAVE' in os.environ and os.environ['TV_SAVE']:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug, '
                       'hence it will get overwritten by further runs.'))
else:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug '
                       'hence it will get overwritten by further runs.'))


def _score_layer(self, bottom, name, num_classes):
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        shape = [1, 1, in_features, num_classes]
        # He initialization Sheme
        num_input = in_features
        stddev = (2 / num_input)**0.5
        # Apply convolution
        w_decay = self.wd
        weights = self._variable_with_weight_decay(shape, stddev, w_decay)
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
        # Apply bias
        conv_biases = self._bias_variable([num_classes], constant=0.0)
        bias = tf.nn.bias_add(conv, conv_biases)

        _activation_summary(bias)

        return bias

def _upscore_layer(self, bottom, shape,
                   num_classes, name, debug,
                   ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.stack(new_shape)

        logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
        f_shape = [ksize, ksize, num_classes, in_features]

        # create
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input)**0.5

        weights = self.get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

        if debug:
            deconv = tf.Print(deconv, [tf.shape(deconv)],
                              message='Shape of %s' % name,
                              summarize=4, first_n=1)


def load_trained_model(sess, hypes):
    """
    Load an exsiting model trained for hand segmentation

    Parameters
    ----------
    hypes : dict
        Hyperparameters

    """
    # load modules
    modules = utils.load_modules_from_hypes(hypes)

    # start loading Graph from the pretrained model
    model_dir = hypes['transfer']['model_folder']
    model_name = hypes['transfer']['model_name']
    # meta file
    meta_file = os.path.join(model_dir, model_name, ".meta")
    # restore
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))


def prepare_tv_session(hypes, sess, saver):
    """
    Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters

    Returns
    -------
    tuple
        (sess, saver, summary_op, summary_writer, threads)
    """
    # Build the summary operation based on the TF collection of Summaries.
    if FLAGS.summary:
        tf.contrib.layers.summarize_collection(tf.GraphKeys.WEIGHTS)
        tf.contrib.layers.summarize_collection(tf.GraphKeys.BIASES)
        summary_op = tf.summary.merge_all()
    else:
        summary_op = None

    # Run the Op to initialize the variables.
    if 'init_function' in hypes:
        _initalize_variables = hypes['init_function']
        _initalize_variables(hypes)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(hypes['dirs']['output_dir'],
                                           graph=sess.graph)

    tv_session = {}
    tv_session['sess'] = sess
    tv_session['saver'] = saver
    tv_session['summary_op'] = summary_op
    tv_session['writer'] = summary_writer
    tv_session['coord'] = coord
    tv_session['threads'] = threads

    return tv_session


def check_model(model_file):
    from tensorflow.python.tools import inspect_checkpoint as chkp

    # print all tensors in checkpoint file
    chkp.print_tensors_in_checkpoint_file(model_file, tensor_name='', all_tensors=True)


def check_weights(sess):
    vars_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(vars_names)

    for k, v in zip(vars_names, values):
        print(k, v)


def check_graph(sess):
    vars_names = [v.name for v in tf.trainable_variables()]
    vars_shapes = [v.get_shape() for v in tf.trainable_variables()]

    # values = sess.run(vars_names)

    for name, shape in zip(vars_names, vars_shapes):
        print(name, shape)


def restoring_vars(hypes, sess):
    retrain_from = hypes["arch"]["retrain_from"]
    if retrain_from == "pool5":
        restoring_vars_names = ["conv1_1", "conv1_2", "pool1", 
                    "conv2_1", "conv2_2", "pool2",
                    "conv3_1", "conv3_2", "conv3_3", "pool3",
                    "conv4_1", "conv4_2", "conv4_3", "pool4",
                    "conv5_1", "conv5_2", "conv5_3", "pool5"]
    elif retrain_from == "fc6":
        restoring_vars_names = ["conv1_1", "conv1_2", "pool1", 
                    "conv2_1", "conv2_2", "pool2",
                    "conv3_1", "conv3_2", "conv3_3", "pool3",
                    "conv4_1", "conv4_2", "conv4_3", "pool4",
                    "conv5_1", "conv5_2", "conv5_3", "pool5",
                    "fc6"]
    elif retrain_from == "fc7":
        restoring_vars_names = ["conv1_1", "conv1_2", "pool1", 
                    "conv2_1", "conv2_2", "pool2",
                    "conv3_1", "conv3_2", "conv3_3", "pool3",
                    "conv4_1", "conv4_2", "conv4_3", "pool4",
                    "conv5_1", "conv5_2", "conv5_3", "pool5",
                    "fc6", "fc7"]
    else:
        restoring_vars_names = ["conv1_1", "conv1_2", "pool1", 
                    "conv2_1", "conv2_2", "pool2",
                    "conv3_1", "conv3_2", "conv3_3", "pool3",
                    "conv4_1", "conv4_2", "conv4_3", "pool4",
                    "conv5_1", "conv5_2", "conv5_3", "pool5"]
                    # "fc6", "fc7"]
    
    vars_to_restore = []

    for name in restoring_vars_names:
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
        if var:
            vars_to_restore += var

    print("vars to restore", vars_to_restore)
    return vars_to_restore


def do_finetuning(hypes):
    """
    Finetune model for a number of steps.

    This finetunes the model for at most hypes['solver']['max_steps'].
    It shows an update every utils.cfg.step_show steps and writes
    the model to hypes['dirs']['output_dir'] every utils.cfg.step_eval
    steps.

    Paramters
    ---------
    hypes : dict
        Hyperparameters
    """
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.

    try:
        import tensorvision.core as core
    except ImportError:
        logging.error("Could not import the submodules.")
        logging.error("Please execute:"
                      "'git submodule update --init --recursive'")
        exit(1)

    modules = utils.load_modules_from_hypes(hypes)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Session() as sess:

        # build the graph based on the loaded modules
        with tf.name_scope("Queues"):
            queue = modules['input'].create_queues(hypes, 'train')

        tv_graph = core.build_training_graph(hypes, queue, modules)

        # restoring vars
        vars_to_restore = restoring_vars(hypes, sess)

        restorer = tf.train.Saver(vars_to_restore)

        # load pre-trained model of hand segmentation
        logging.info("Loading pretrained model's weights")
        model_dir = hypes['transfer']['model_folder']
        model_file = hypes['transfer']['model_name']
        # DEBUG: check the model file
        # check_model(os.path.join(model_dir, model_file))

        """
        # Get a list of vars to restore
        vars_to_restore = restoring_vars(sess)
        print("vars to restore:", vars_to_restore)
        # Create another Saver for restoring pre-trained vars
        saver = tf.train.Saver(vars_to_restore)
        """
        core.load_weights(model_dir, sess, restorer)
        # load_trained_model(sess, hypes)

        saver = tf.train.Saver(max_to_keep=int(utils.cfg.max_to_keep))

        # prepaire the tv session
        tv_sess = prepare_tv_session(hypes, sess, saver)

        # DEBUG: print weights
        # check_weights(tv_sess['sess'])
        # check_graph(tv_sess['sess'])
        
        with tf.name_scope('Validation'):
            tf.get_variable_scope().reuse_variables()
            image_pl = tf.placeholder(tf.float32)
            image = tf.expand_dims(image_pl, 0)
            image.set_shape([1, None, None, 3])
            inf_out = core.build_inference_graph(hypes, modules,
                                                 image=image)
            tv_graph['image_pl'] = image_pl
            tv_graph['inf_out'] = inf_out

        # Start the data load
        modules['input'].start_enqueuing_threads(hypes, queue, 'train', sess)

        # And then after everything is built, start the training loop.
        train.run_training(hypes, modules, tv_graph, tv_sess)

        # stopping input Threads
        tv_sess['coord'].request_stop()
        tv_sess['coord'].join(tv_sess['threads'])

        # now freeze the graph and save it
        # frozen_graph = TOR_utils.freeze_session(
        #                     sess,
        #                     output_names=[out.op.name for out in model.outputs]
        #                 )


def main(_):
    utils.set_gpus_to_use()

    if tf.app.flags.FLAGS.hypes is None:
        logging.error("No hype file is given.")
        logging.info("Usage: python train.py --hypes hypes/KittiClass.json")
        exit(1)

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = commentjson.load(f)

    utils.load_plugins()

    if tf.app.flags.FLAGS.mod is not None:
        import ast
        mod_dict = ast.literal_eval(tf.app.flags.FLAGS.mod)
        dict_merge(hypes, mod_dict)

    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'],
                                                 'KittiSeg')
    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    utils._add_paths_to_sys(hypes)

    train.maybe_download_and_extract(hypes)
    logging.info("Initialize training folder")
    train.initialize_training_folder(hypes)
    logging.info("Start finetuning")
    do_finetuning(hypes)


if __name__ == '__main__':
    # allocating TF graph to only one GPU
    #https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # only using CPUs
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    gpu_to_use = gpu_utils.get_idle_gpu(leave_unmasked=0)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    if gpu_to_use >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_to_use)
    else:
        # only using CPUs
        print("ERROR: no GPU to use for training")
        sys.exit()
    tf.app.run()
