"""
Train a model that learns all.
all = object localization + object recognition

Many codes are adapted from KittiSeg;

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import commentjson
#import json as commentjson
import logging
import os
import sys
import cv2

import collections
import gpu_utils
# for training
import time


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
    # logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
    #                     level=logging.INFO,
    #                     stream=sys.stdout)
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)
# to prevent double logging
# https://stackoverflow.com/questions/19561058/duplicate-output-in-simple-python-logging-configuration
# logging.propagate = False

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

import tensorvision.train as train
import tensorvision.utils as utils
import tensorvision.core as core

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


def build_training_graph(hypes, queue, modules):
    """
    Build the tensorflow graph out of the model files.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    queue: tf.queue
        Data Queue
    modules : tuple
        The modules load in utils.

    Returns
    -------
    tuple
        (q, train_op, loss, eval_lists) where
        q is a dict with keys 'train' and 'val' which includes queues,
        train_op is a tensorflow op,
        loss is a float,
        eval_lists is a dict with keys 'train' and 'val'
    """

    data_input = modules['input']
    encoder = modules['arch']
    objective = modules['objective']
    optimizer = modules['solver']

    learning_rate = tf.placeholder(tf.float32)

    # Add Input Producers to the Graph
    with tf.name_scope("Inputs"):
        image, labels, classes = data_input.inputs(hypes, queue, phase='train')

    # Run inference on the encoder network
    logits = encoder.inference(hypes, image, train=True)

    # Build decoder on top of the logits
    decoded_logits = objective.decoder(hypes, logits, train=True)

    # Add to the Graph the Ops for loss calculation.
    with tf.name_scope("Loss"):
        losses = objective.loss(hypes, decoded_logits,
                                labels, classes)

    # Add to the Graph the Ops that calculate and apply gradients.
    with tf.name_scope("Optimizer"):
        global_step = tf.Variable(0, trainable=False)
        # Build training operation
        train_op = optimizer.training(hypes, losses,
                                      global_step, learning_rate)

    with tf.name_scope("Evaluation"):
        # Add the Op to compare the logits to the labels during evaluation.
        eval_list = objective.evaluation(
            hypes, image, labels, classes, decoded_logits, losses, global_step)

        summary_op = tf.summary.merge_all()

    graph = {}
    graph['losses'] = losses
    graph['eval_list'] = eval_list
    graph['summary_op'] = summary_op
    graph['train_op'] = train_op
    graph['global_step'] = global_step
    graph['learning_rate'] = learning_rate
    graph['decoded_logits'] = decoded_logits

    return graph


def _write_eval_dict_to_summary(eval_dict, tag, summary_writer, global_step):
    summary = tf.Summary()
    for name, result in eval_dict:
        summary.value.add(tag=tag + '/' + name,
                          simple_value=result)
    summary_writer.add_summary(summary, global_step)
    return


def _write_images_to_summary(images, summary_writer, step):
    for name, image in images:
        image = image.astype('float32')
        shape = image.shape
        image = image.reshape(1, shape[0], shape[1], shape[2])
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                log_image = tf.summary.image(name, image)
            with tf.Session() as sess:
                summary_str = sess.run([log_image])
                summary_writer.add_summary(summary_str[0], step)
        break
    return


def _write_images_to_disk(hypes, images, step):

    new_dir = str(step) + "_images"
    image_dir = os.path.join(hypes['dirs']['image_dir'], new_dir)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    for name, image in images:
        file_name = os.path.join(image_dir, name)
        cv2.imwrite(file_name, image)


def _print_eval_dict(eval_names, eval_results, prefix=''):
    print_str = ", ".join([nam + ": %.2f" for nam in eval_names])
    print_str = "   " + prefix + "  " + print_str
    logging.info(print_str % tuple(eval_results))


class ExpoSmoother():
    """docstring for expo_smoother"""
    def __init__(self, decay=0.9):
        self.weights = None
        self.decay = decay

    def update_weights(self, l):
        if self.weights is None:
            self.weights = np.array(l)
            return self.weights
        else:
            self.weights = self.decay*self.weights + (1-self.decay)*np.array(l)
            return self.weights

    def get_weights(self):
        return self.weights.tolist()


class MedianSmoother():
    """docstring for expo_smoother"""
    def __init__(self, num_entries=50):
        self.weights = None
        self.num = 50

    def update_weights(self, l):
        l = np.array(l).tolist()
        if self.weights is None:
            self.weights = [[i] for i in l]
            return [np.median(w[-self.num:]) for w in self.weights]
        else:
            for i, w in enumerate(self.weights):
                w.append(l[i])
            if len(self.weights) > 20*self.num:
                self.weights = [w[-self.num:] for w in self.weights]
            return [np.median(w[-self.num:]) for w in self.weights]

    def get_weights(self):
        return [np.median(w[-self.num:]) for w in self.weights]


def run_training(hypes, modules, tv_graph, tv_sess, start_step=0):
    """Run one iteration of training."""
    # Unpack operations for later use
    summary = tf.Summary()
    sess = tv_sess['sess']
    summary_writer = tv_sess['writer']

    solver = modules['solver']

    display_iter = hypes['logging']['display_iter']
    write_iter = hypes['logging'].get('write_iter', 5*display_iter)
    eval_iter = hypes['logging']['eval_iter']
    save_iter = hypes['logging']['save_iter']
    image_iter = hypes['logging'].get('image_iter', 5*save_iter)

    py_smoother = MedianSmoother(20)
    dict_smoother = ExpoSmoother(0.95)

    n = 0

    eval_names, eval_ops = zip(*tv_graph['eval_list'])
    # Run the training Step
    start_time = time.time()
    for step in range(start_step, hypes['solver']['max_steps']):

        lr = solver.get_learning_rate(hypes, step)
        feed_dict = {tv_graph['learning_rate']: lr}

        if step % display_iter:
            sess.run([tv_graph['train_op']], feed_dict=feed_dict)

        # Write the summaries and print an overview fairly often.
        elif step % display_iter == 0:
            # Print status to stdout.
            _, loss_value = sess.run([tv_graph['train_op'],
                                      tv_graph['losses']['total_loss']],
                                     feed_dict=feed_dict)

            _print_training_status(hypes, step, loss_value, start_time, lr)

            eval_results = sess.run(eval_ops, feed_dict=feed_dict)

            _print_eval_dict(eval_names, eval_results, prefix='   (raw)')
            """
            dict_smoother.update_weights(eval_results)
            smoothed_results = dict_smoother.get_weights()

            _print_eval_dict(eval_names, smoothed_results, prefix='(smooth)')
            """
            # Reset timer
            start_time = time.time()

        if step % write_iter == 0:
            # write values to summary
            if FLAGS.summary:
                summary_str = sess.run(tv_sess['summary_op'],
                                       feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
            summary.value.add(tag='training/total_loss',
                              simple_value=float(loss_value))
            summary.value.add(tag='training/learning_rate',
                              simple_value=lr)
            summary_writer.add_summary(summary, step)
            # Convert numpy types to simple types.
            eval_results = np.array(eval_results)
            eval_results = eval_results.tolist()
            eval_dict = zip(eval_names, eval_results)
            _write_eval_dict_to_summary(eval_dict, 'Eval/raw',
                                        summary_writer, step)
            # eval_dict = zip(eval_names, smoothed_results)
            # _write_eval_dict_to_summary(eval_dict, 'Eval/smooth',
            #                             summary_writer, step)

        # Do a evaluation and print the current state
        if (step) % eval_iter == 0 and step > 0 or \
           (step + 1) == hypes['solver']['max_steps']:
            # write checkpoint to disk

            logging.info('Running Evaluation Script.')
            eval_dict, images = modules['eval'].evaluate(
                hypes, sess, tv_graph['image_pl'], tv_graph['inf_out'])

            _write_images_to_summary(images, summary_writer, step)
            logging.info("Evaluation Finished. All results will be saved to:")
            logging.info(hypes['dirs']['output_dir'])

            if images is not None and len(images) > 0:

                name = str(n % 10) + '_' + images[0][0]
                image_file = os.path.join(hypes['dirs']['image_dir'], name)
                # scp.misc.imsave(image_file, images[0][1])
                cv2.imwrite(image_file, images[0][1])
                n = n + 1

            logging.info('Raw Results:')
            utils.print_eval_dict(eval_dict, prefix='(raw)   ')
            _write_eval_dict_to_summary(eval_dict, 'Evaluation/raw',
                                        summary_writer, step)
            """
            logging.info('Smooth Results:')
            names, res = zip(*eval_dict)
            smoothed = py_smoother.update_weights(res)
            eval_dict = zip(names, smoothed)
            utils.print_eval_dict(eval_dict, prefix='(smooth)')
            _write_eval_dict_to_summary(eval_dict, 'Evaluation/smoothed',
                                        summary_writer, step)
            """
            # Reset timer
            start_time = time.time()

        # Save a checkpoint periodically.
        if (step) % save_iter == 0 and step > 0 or \
           (step + 1) == hypes['solver']['max_steps']:
            # write checkpoint to disk
            checkpoint_path = os.path.join(hypes['dirs']['output_dir'],
                                           'model.ckpt')
            tv_sess['saver'].save(sess, checkpoint_path, global_step=step)
            # Reset timer
            start_time = time.time()

        if step % image_iter == 0 and step > 0 or \
           (step + 1) == hypes['solver']['max_steps']:
            _write_images_to_disk(hypes, images, step)


def _print_training_status(hypes, step, loss_value, start_time, lr):

    info_str = utils.cfg.step_str

    # Prepare printing
    duration = (time.time() - start_time) / int(utils.cfg.step_show)
    examples_per_sec = hypes['solver']['batch_size'] / duration
    sec_per_batch = float(duration)

    logging.info(info_str.format(step=step,
                                 total_steps=hypes['solver']['max_steps'],
                                 loss_value=loss_value,
                                 lr_value=lr,
                                 sec_per_batch=sec_per_batch,
                                 examples_per_sec=examples_per_sec)
                 )


def do_training(hypes):
    """
    Train model for a number of steps.

    This trains the model for at most hypes['solver']['max_steps'].
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

    modules = utils.load_modules_from_hypes(hypes)

    # set to allocate memory on GPU as needed
    # For more details, look at
    # https://stackoverflow.com/questions/36927607/how-can-i-solve-ran-out-of-gpu-memory-in-tensorflow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Tell TensorFlow that the model will be built into the default Graph.
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:

        # build the graph based on the loaded modules
        with tf.name_scope("Queues"):
            queue = modules['input'].create_queues(hypes, 'train')

        tv_graph = build_training_graph(hypes, queue, modules)

        # prepaire the tv session
        tv_sess = core.start_tv_session(hypes)

        with tf.name_scope('Validation'):
            tf.get_variable_scope().reuse_variables()
            num_channels = hypes['arch']['num_channels']
            image_pl = tf.placeholder(tf.float32, [None, None, num_channels])
            image = tf.expand_dims(image_pl, 0)
            if hypes['jitter']['resize_image']:
                height = hypes['jitter']['image_height']
                width = hypes['jitter']['image_width']
                # set the pre-defined image size here
                image.set_shape([1, height, width, num_channels])
            inf_out = core.build_inference_graph(hypes, modules,
                                                 image=image)
            tv_graph['image_pl'] = image_pl
            tv_graph['inf_out'] = inf_out

        # Start the data load
        modules['input'].start_enqueuing_threads(hypes, queue, 'train', sess)

        # And then after everything is built, start the training loop.
        run_training(hypes, modules, tv_graph, tv_sess)

        # stopping input Threads
        tv_sess['coord'].request_stop()
        tv_sess['coord'].join(tv_sess['threads'])


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


def restoring_vars(hypes):
    retrain_from = hypes["arch"]["retrain_from"]
    if retrain_from == "cp":
        vars_names = ["conv1_1", "conv1_2", "pool1", 
            "conv2_1", "conv2_2", "pool2",
            "conv3_1", "conv3_2", "conv3_3", "pool3",
            "conv4_1", "conv4_2", "conv4_3", "pool4",
            "conv5_1", "conv5_2", "conv5_3", "pool5",
            "fc6", "fc7", "score_fr", "upscore2",
            "score_pool4", "upscore4", "score_pool3", "upscore32"]
    elif retrain_from == "pool5":
        vars_names = ["conv1_1", "conv1_2", "pool1", 
                    "conv2_1", "conv2_2", "pool2",
                    "conv3_1", "conv3_2", "conv3_3", "pool3",
                    "conv4_1", "conv4_2", "conv4_3", "pool4",
                    "conv5_1", "conv5_2", "conv5_3", "pool5"]
    elif retrain_from == "fc6":
        vars_names = ["conv1_1", "conv1_2", "pool1", 
                    "conv2_1", "conv2_2", "pool2",
                    "conv3_1", "conv3_2", "conv3_3", "pool3",
                    "conv4_1", "conv4_2", "conv4_3", "pool4",
                    "conv5_1", "conv5_2", "conv5_3", "pool5",
                    "fc6"]
    elif retrain_from == "fc7":
        vars_names = ["conv1_1", "conv1_2", "pool1", 
                    "conv2_1", "conv2_2", "pool2",
                    "conv3_1", "conv3_2", "conv3_3", "pool3",
                    "conv4_1", "conv4_2", "conv4_3", "pool4",
                    "conv5_1", "conv5_2", "conv5_3", "pool5",
                    "fc6", "fc7"]
    else:
        vars_names = ["conv1_1", "conv1_2", "pool1", 
                    "conv2_1", "conv2_2", "pool2",
                    "conv3_1", "conv3_2", "conv3_3", "pool3",
                    "conv4_1", "conv4_2", "conv4_3", "pool4",
                    "conv5_1", "conv5_2", "conv5_3", "pool5"]
    
    vars_to_restore = []
    for name in vars_names:
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

    # set to allocate memory on GPU as needed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Session(config=config) as sess:
    # with tf.Session() as sess:

        # build the graph based on the loaded modules
        with tf.name_scope("Queues"):
            queue = modules['input'].create_queues(hypes, 'train')

        tv_graph = build_training_graph(hypes, queue, modules)

        # restoring vars
        vars_to_restore = restoring_vars(hypes)

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
            num_channels = hypes['arch']['num_channels']
            image_pl = tf.placeholder(tf.float32, [None, None, num_channels])
            image = tf.expand_dims(image_pl, 0)
            if hypes['jitter']['resize_image']:
                height = hypes['jitter']['image_height']
                width = hypes['jitter']['image_width']
                # set the pre-defined image size here
                image.set_shape([1, height, width, num_channels])

            inf_out = core.build_inference_graph(hypes, modules,
                                                 image=image)
            tv_graph['image_pl'] = image_pl
            tv_graph['inf_out'] = inf_out

        # Start the data load
        modules['input'].start_enqueuing_threads(hypes, queue, 'train', sess)

        # And then after everything is built, start the training loop.
        run_training(hypes, modules, tv_graph, tv_sess)

        # stopping input Threads
        tv_sess['coord'].request_stop()
        tv_sess['coord'].join(tv_sess['threads'])


def start(hypes):
    """ Start training or fine-tuning a model

    Args
        hypes: a dictionary

    Returns
        void
    """
    model_type = hypes['arch']['type']
    if model_type == "cp" or model_type == "multi" or model_type == "ft":
        # fine-tuning for context-priming/multi-task approach
        do_finetuning(hypes)
    else:
        # training a model
        do_training(hypes)


def main(_):
    utils.set_gpus_to_use()

    try:
        import tensorvision.train
        import tensorflow_fcn.utils
    except ImportError:
        logging.error("Could not import the submodules.")
        logging.error("Please execute:"
                      "'git submodule update --init --recursive'")
        exit(1)

    if tf.app.flags.FLAGS.hypes is None:
        logging.error("No hype file is given.")
        logging.info("Usage: python train_recog.py --hypes hypes/KittiClass.json")
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
                                                 'TOR')
    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    utils._add_paths_to_sys(hypes)

    train.maybe_download_and_extract(hypes)
    logging.info("Initialize training folder")
    train.initialize_training_folder(hypes)
    logging.info("Start training")
    start(hypes)


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
