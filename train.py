import os, shutil

import sys
import time

import tensorflow as tf
import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dc_models import SDDCNetwork

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datacenter'))
from datacenter.prism import PrismTFPipeline
import tf_utils

# model parameters
flags = tf.flags

# model hyperparamters
flags.DEFINE_string('hidden', '512,512', 'Number of units in hidden layer 1.')
flags.DEFINE_string('kernels', '9,3,5', 'Kernel size of layer 1.')
flags.DEFINE_integer('depth', 2, 'Number of input channels.')
flags.DEFINE_integer('upscale', 4, 'Upscale factor.')
flags.DEFINE_integer('lr_km', 64, 'Low spatial resolution in km.')
flags.DEFINE_integer('hr_km', 16, 'High spatial resolution in km.')
flags.DEFINE_integer('precip_threshold', 0.5, 'What consititutes precipitation mm?')
flags.DEFINE_boolean('residual', True, 'Should we learn the residual?')
flags.DEFINE_string('distribution', 'normal', 'Select a distribution, normal or gamma.')
flags.DEFINE_float('priorlengthscale', 1e1, 'Prior Length Scale for weight decay')
flags.DEFINE_float('tau', 1e-5, 'Regularization Parameter for dropout decay')

# Model training parameters
flags.DEFINE_integer('num_epochs', 100001, 'Number of epochs to run trainer.')
flags.DEFINE_float('learning_rate', 1e-4, 'Select a learning rate')
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_string('device', '/gpu:3', 'What device should I train on?')
flags.DEFINE_integer('mc_runs', 30, 'Number of monte carlo samples.')
flags.DEFINE_integer('patch_size', 64, 'How large should the patch size be')
flags.DEFINE_integer('stride', 48, 'how far to stride which making subimages')
flags.DEFINE_integer('experiment', 2, '1=Normal, 2=Normal DC, 3=Gamma DC, 4=Lognormal DC')
flags.DEFINE_float('training_N', 365*152*348, 'Number of training samples per year.')
flags.DEFINE_boolean('load_checkpoint', True, 'Try to load weights from a checkpoint file?')
flags.DEFINE_integer('N_train_years', 25, '(Optional, used in experiment 4) Number of training years.')

# when to save, plot, and test
flags.DEFINE_integer('save_step', 5000, 'How often should I save the model')
flags.DEFINE_integer('test_step',250, 'How often test steps are executed and printed')

# where to save things
flags.DEFINE_string('save_dir', 'results/your-results/', 'Where to save checkpoints.')

def _maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def density_moments(samples):
    #[[locs, scales, weights],...]
    # loc shape: [N, height, width, 3, K, mc_samples]
    loc_samples = np.concatenate([np.expand_dims(s[0], -1) for s in samples], axis=-1)
    scale_samples = np.concatenate([np.expand_dims(s[1], -1) for s in samples], axis=-1)
    weight_samples = np.concatenate([np.expand_dims(s[2], -1) for s in samples], axis=-1)

    EX = np.sum(np.sum(loc_samples * weight_samples, axis=-1), axis=-1) / loc_samples.shape[-1]
    EX[EX < 0] = 0.
    EX[EX > 1] = 1.

    plt.imshow(EX[0])
    plt.show()

def train(FLAGS, train_years=range(1981,2006), test_years=range(2006,2016)):
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        with tf.name_scope("reading_data"):
            data_dir = '/home/tj/repos/datacenter/datacenter/prism/data'
            N = int(FLAGS.training_N) * len(train_years)
            train_data_pipeline = PrismTFPipeline(data_dir, years=train_years, lr_km=FLAGS.lr_km,
                                         hr_km=FLAGS.hr_km, input_vars=['ppt'])
            train_data = train_data_pipeline.tf_patches(batch_size=FLAGS.batch_size,
                                               patch_size=FLAGS.patch_size,
                                               stride=FLAGS.stride)
            test_data_pipeline = PrismTFPipeline(data_dir, years=test_years, lr_km=FLAGS.lr_km,
                                        hr_km=FLAGS.hr_km, input_vars=['ppt'])
            test_data= test_data_pipeline.tf_patches(batch_size=1, is_training=False,
                                            #patch_size=FLAGS.patch_size,
                                            patch_size=FLAGS.patch_size,
                                            stride=FLAGS.stride)
            # set placeholders, at test time use placeholder
            is_training = tf.placeholder_with_default(True, (), name='is_training')

            x = tf.cond(is_training, lambda: train_data[0], lambda: test_data[0], name='x')
            aux = tf.cond(is_training, lambda: train_data[1], lambda: test_data[1], name='aux')
            y = tf.cond(is_training, lambda: train_data[2], lambda: test_data[2])
            y_filled = tf.nn.relu(tf.where(tf.is_nan(y), tf.zeros_like(y), y)) # ensure non-negativity

            # x needs to be interpolated to the shape of y
            h = tf.shape(x)[1] * FLAGS.upscale
            w = tf.shape(x)[2] * FLAGS.upscale
            x_interp = tf.nn.relu(tf.image.resize_bilinear(x, [h,w])) # force non-negativity
            x_aux = tf.concat([x_interp, aux], axis=3)

        HIDDEN_LAYERS = [int(x) for x in FLAGS.hidden.split(",")]
        KERNELS = [int(x) for x in FLAGS.kernels.split(",")]

        # build graph
        model = SDDCNetwork(x_aux, y_filled, HIDDEN_LAYERS, KERNELS,
                            is_training=is_training, input_depth=2, residual=FLAGS.residual,
                            upscale_factor=FLAGS.upscale, residual_channels=[0],
                            learning_rate=FLAGS.learning_rate, device=FLAGS.device, N=N,
                            distribution=FLAGS.distribution,
                            precip_threshold=FLAGS.precip_threshold,
                            tau=FLAGS.tau, priorlengthscale=FLAGS.priorlengthscale)

        with tf.name_scope("compute_some_stats"):

            pred = tf.nn.relu(model.prediction)
            def image_summary(name, img):
                tf.summary.image(name, img, max_outputs=1)

            image_summary('input', tf.expand_dims(x_aux[:,:100,:100,0], -1))
            image_summary('prediction', pred[:,:100,:100])
            image_summary('label', y[:,:100,:100])

        # initialize graph
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Create a session for running operations in the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Initialize the variables (the trained variables and the # epoch counter).
        sess.run(init_op)

        #if os.path.exists(SAVE_DIR):
        #    shutil.rmtree(SAVE_DIR)

        if FLAGS.load_checkpoint:
            try:
                checkpoint = tf.train.latest_checkpoint(SAVE_DIR)
                saver.restore(sess, checkpoint)
            except ValueError:
                print("Could not find checkpoint")


        # summary data
        summary_op = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(SAVE_DIR + '/test', sess.graph)
        train_writer = tf.summary.FileWriter(SAVE_DIR + '/train', sess.graph) 
        s0 = sess.run(model.global_step)
        xa = None
        ya = None
        t = None

        '''while 1:
            xa, ya, t = sess.run([train_data[0], y_filled, train_data[3]])
            if np.max(xa) > 1e20:
                print "very large number, fuck man, time:", t[0]
        return
        '''
        for step in range(s0, FLAGS.num_epochs):
            try:
                _, train_loss = sess.run([model.opt, model.loss])
                if step % FLAGS.test_step == 0:
                    stats = []
                    d = {is_training: False}
                    test_writer.add_summary(sess.run(summary_op, feed_dict=d), step)
                    train_writer.add_summary(sess.run(summary_op), step)
                    print("Step: %i, Train Loss: %2.4f" %\
                            (step, train_loss))
            # if there's a corrupted record, delete data and begin retraining
            except tf.errors.DataLossError as err:
                train_data_pipeline._save_patches(FLAGS.patch_size, FLAGS.stride, force=True)
                train(FLAGS)
            if step % FLAGS.save_step == 0:
                save_path = saver.save(sess, os.path.join(SAVE_DIR, "model_%08i.ckpt" % step))
        save_path = saver.save(sess, os.path.join(SAVE_DIR, "model_%08i.ckpt" % step))

def normal():
    FLAGS.distribution = 'normal'
    FLAGS.precip_threshold = -0.1

    file_dir = os.path.dirname(os.path.abspath(__file__))
    global SAVE_DIR
    SAVE_DIR = os.path.join(file_dir, FLAGS.save_dir, "%s_%s_%s" % (FLAGS.distribution,
                        FLAGS.hidden.replace(",", "-"), FLAGS.kernels.replace(",", "-")))
    _maybe_make_dir(os.path.dirname(SAVE_DIR))
    _maybe_make_dir(SAVE_DIR)
    train(FLAGS)

def normal_dc():
    FLAGS.distribution = 'normal'
    file_dir = os.path.dirname(os.path.abspath(__file__))
    global SAVE_DIR
    SAVE_DIR = os.path.join(file_dir, FLAGS.save_dir, "%s_%s_%s_dc" % (FLAGS.distribution,
                        FLAGS.hidden.replace(",", "-"), FLAGS.kernels.replace(",", "-")))
    _maybe_make_dir(os.path.dirname(SAVE_DIR))
    _maybe_make_dir(SAVE_DIR)
    train(FLAGS)

## This distribution is very unstable, uncertainty widths explode - need to regularize alpha and beta
def gamma_dc():
    FLAGS.distribution = 'gamma'
    file_dir = os.path.dirname(os.path.abspath(__file__))
    global SAVE_DIR
    SAVE_DIR = os.path.join(file_dir, FLAGS.save_dir, "%s_%s_%s" % (FLAGS.distribution,
                        FLAGS.hidden.replace(",", "-"), FLAGS.kernels.replace(",", "-")))

    _maybe_make_dir(os.path.dirname(SAVE_DIR))
    _maybe_make_dir(SAVE_DIR)
    train(FLAGS)

def lognormal():
    FLAGS.distribution = 'lognormal'
    file_dir = os.path.dirname(os.path.abspath(__file__))
    global SAVE_DIR
    SAVE_DIR = os.path.join(file_dir, FLAGS.save_dir, "%s_%s_%s" % (FLAGS.distribution,
                        FLAGS.hidden.replace(",", "-"), FLAGS.kernels.replace(",", "-")))

    _maybe_make_dir(os.path.dirname(SAVE_DIR))
    _maybe_make_dir(SAVE_DIR)
    train(FLAGS)

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS._parse_flags()
    if "gpu" in FLAGS.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.device[-1]
        FLAGS.device = '/gpu:0'

    if FLAGS.experiment == 1:
        normal()
    elif FLAGS.experiment == 2:
        normal_dc()
    elif FLAGS.experiment == 3:
        gamma_dc()
    elif FLAGS.experiment == 4:
        lognormal()
