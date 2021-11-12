from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import tensorflow as tf


# def get_learning_rate(hypes, step):
#    lr = hypes['solver']['learning_rate']
#    lr_step = hypes['solver']['learning_rate_step']
#    if lr_step is not None:
#      adjusted_lr = (lr * 0.5 ** max(0, (step / lr_step) - 2))
#        return adjusted_lr
#    else:
#        return lr

def get_learning_rate(hypes, step):
    if "learning_rates" not in hypes['solver']:
        lr = hypes['solver']['learning_rate']
        lr_step = hypes['solver']['learning_rate_step']
        if lr_step is not None:
            adjusted_lr = (lr * 0.5 ** max(0, (step / lr_step) - 2))
            return adjusted_lr
        else:
            return lr

    for i, num in enumerate(hypes['solver']['steps']):
        if step < num:
            return hypes['solver']['learning_rates'][i]


def training(hypes, loss, global_step, learning_rate, opt=None):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.''
    sol = hypes["solver"]
    hypes['tensors'] = {}
    hypes['tensors']['global_step'] = global_step
    total_loss = loss['total_loss']
    with tf.name_scope('training'):

        if opt is None:

            if sol['opt'] == 'RMS':
                opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                decay=0.9,
                                                epsilon=sol['epsilon'])
            elif sol['opt'] == 'Adam':
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                             epsilon=sol['adam_eps'])
            elif sol['opt'] == 'SGD':
                lr = learning_rate
                opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
            else:
                raise ValueError('Unrecognized opt type')

        hypes['opt'] = opt

        # get the final layer and others if necessary
        # TODO: what layers should we finetune ?
        """ for FCN8
        train_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "upscore32") +\
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_pool3") +\
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "upscore4") +\
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_pool4") +\
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_fr")# +
                # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc7")
        """
        retrain_from = hypes["arch"]["retrain_from"]
        if retrain_from is "" or retrain_from == "all":
            grads_and_vars = opt.compute_gradients(total_loss)
            print("=== retraining all layers ===")

        else:
            arch_model = hypes["model"]["architecture_file"]
            is_recognition = "recog" in hypes["model"]["input_file"]
            is_shallow = False
            try:
                is_shallow = "shallow" == hypes['arch']['recog_depth']
            except KeyError:
                is_shallow = False
            if "fcn32" in arch_model:
                if retrain_from == "pool5":
                    train_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "up") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_fr") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc7") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc6")    
                elif retrain_from == "fc6":
                    train_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "up") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_fr") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc7")
                elif retrain_from == "fc7":
                    train_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "up") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_fr")
                elif retrain_from == "cp":
                    train_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "local")
                else:
                    train_layers = []
            elif "fcn8" in arch_model:
                if retrain_from == "pool5":
                    train_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "up") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_pool") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_fr") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc7") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc6")    
                elif retrain_from == "fc6":
                    train_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "up") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_pool") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_fr") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc7")
                elif retrain_from == "fc7":
                    train_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "up") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_pool") +\
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_fr")
                elif retrain_from == "cp":
                    train_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "local")
                else:
                    train_layers = []

                if is_recognition:
                    if is_shallow:
                        train_layers += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "flatten") +\
                            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pred")
                    else:
                        train_layers += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "flatten") +\
                            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dense6") +\
                            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dense7") +\
                            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pred")

            # DEBUG: print retraining layers
            print("retraining layers: ", train_layers)
            grads_and_vars = opt.compute_gradients(total_loss, var_list = train_layers)

        # train_op = opt.minimize(total_loss, var_list=train_layers)

        if hypes['clip_norm'] > 0:
            grads, tvars = zip(*grads_and_vars)
            clip_norm = hypes["clip_norm"]
            clipped_grads, norm = tf.clip_by_global_norm(grads, clip_norm)
            if sys.version_info[0] < 3:
                # python 2
                grads_and_vars = zip(clipped_grads, tvars)
            else:
                # python 3
                grads_and_vars = list(zip(clipped_grads, tvars))

        train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # DEBUG: print update operations
        print("update ops: ", update_ops)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(grads_and_vars,
                                           global_step=global_step)

    return train_op
