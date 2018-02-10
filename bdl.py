import tensorflow as tf
import numpy as np

def _concrete_dropout(x, p, eps=1e-8):
    temp = 0.1
    unif_noise = tf.random_uniform(shape=tf.shape(x))
    drop_prob = tf.log(p + eps) - tf.log(1 - p + eps)
    drop_prob += tf.log(unif_noise + eps) - tf.log(1. - unif_noise + eps)
    drop_prob = tf.sigmoid(drop_prob/temp)
    random_tensor = 1. - drop_prob
    retain_prob = 1. - p
    x *= random_tensor
    x /= retain_prob
    return x

def concrete_layer(inputs,
                  init_prob=0.25,
                  is_training=True,
                  scope=None,
                  reg_scale=1.,
                  eps=1e-6):
    init = (np.log(init_prob) - np.log(1. - init_prob)).astype(np.float32)
    with tf.variable_scope(scope, 'ConcreteDropout', [inputs]):
        dropout_logit = tf.get_variable("dropout_logit", dtype=tf.float32,
                                      initializer=tf.constant(init))
        dropout_prob = tf.sigmoid(dropout_logit, name="dropout_prob")

        dropout_regularizer = dropout_prob * tf.log(dropout_prob + eps)
        dropout_regularizer += (1. - dropout_prob) * tf.log(1. - dropout_prob + eps)

        tf.losses.add_loss(tf.identity(reg_scale*dropout_regularizer, name='dropout_reg'),
                            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

        with tf.device("/cpu:0"):
            tf.summary.scalar("dropout_prob", dropout_prob)
            tf.summary.scalar("dropout_reg", reg_scale*dropout_regularizer)
            tf.summary.scalar("dropout_reg_raw", dropout_regularizer)

        return  tf.cond(is_training,
                lambda: _concrete_dropout(inputs, dropout_prob, eps=eps),
                lambda: tf.layers.dropout(inputs, dropout_prob))
