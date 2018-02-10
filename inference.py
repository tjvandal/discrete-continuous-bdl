import os, sys
import tensorflow as tf
import ConfigParser
from tensorflow.python.framework import graph_util
import numpy as np
import xarray as xr
import cv2
import time
import matplotlib
import scipy
#matplotlib.use('agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datacenter'))
from datacenter.prism import PrismTFPipeline

flags = tf.flags
flags.DEFINE_integer('upscale', 4, 'Upscale factor.')
flags.DEFINE_string('base_model_dir', 'results/your-results/', 'Where checkpoints are saved.')
flags.DEFINE_string('data_dir', '/home/tj/repos/datacenter/datacenter/prism/data', 'Where did you save your data?')
flags.DEFINE_string('deepsd_model_name', 'yourgraph', 'Lets name our joined graph')
flags.DEFINE_string('years', None, 'What years would you like to infer :)? If None ill do all')
flags.DEFINE_integer('mc_runs', 50, 'Number of monte carlo runs')
flags.DEFINE_integer('gpu', 0, 'Which gpu do you want to use?')

# parse flags
FLAGS = flags.FLAGS
FLAGS._parse_flags()

def get_graph_def():
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        new_saver = tf.train.import_meta_graph(checkpoint + '.meta')
        new_saver.restore(sess, checkpoint)
        return sess.graph_def

def freeze_graph(model_folder, graph_name=None, distribution='normal'):
    # We start a session and restore the graph weights
    with tf.Session() as sess:
        # We retrieve our checkpoint fullpath
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path

        # We precise the file fullname of our freezed graph
        absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_folder + "/frozen_model.pb"
        if os.path.exists(output_graph):
            os.remove(output_graph)

        # Before exporting our graph, we need to precise what is our output node
        # This is how TF decides what part of the Graph he has to keep and what part it can dump
        # NOTE: this variable is plural, because you can have multiple output nodes
        if graph_name is not None:
            #output_node_names = "prediction/prediction"
            if distribution.lower() in ['normal', 'lognormal']:
                output_node_names = "output_layer/loc,output_layer/logvar,output_layer/precip_probs"
            elif distribution.lower() == 'gamma':
                output_node_names = "output_layer/alpha,output_layer/beta,output_layer/precip_probs"
        else:
            raise ValueError("Give me a graph_name")

        # We clear devices to allow TensorFlow to control on which device it will load operations
        clear_devices = False

        # We import the meta graph and retrieve a Saver
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                           clear_devices=clear_devices)

        # We retrieve the protobuf graph definition
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        black_list = []
        saver.restore(sess, input_checkpoint)

        # Retrieve the protobuf graph definition and fix the batch norm nodes
        gd = sess.graph.as_graph_def()
        for node in gd.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            gd, # The graph_def is used to retrieve the nodes 
            output_node_names.split(","), # The output node names are used to select the usefull nodes
            variable_names_blacklist=black_list
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        return output_graph

def load_graph(frozen_graph_filename, graph_name, x=None, aux=None, distribution='normal'):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    if distribution.lower() in ['normal', 'lognormal']:
        return_elements = ['output_layer/loc:0', 'output_layer/logvar:0',
                           'output_layer/precip_probs:0']
    elif distribution.lower() == 'gamma':
        return_elements = ['output_layer/alpha:0', 'output_layer/beta:0',
                           'output_layer/precip_probs:0']

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    #with tf.Graph().as_default() as graph:
        is_training = tf.constant(False)
        if x is None:
            x = tf.placeholder(tf.float32, shape=(None, None, None, 1), name="%s/new_x" % graph_name)
        y = tf.import_graph_def(
            graph_def,
            input_map={'reading_data/x/Switch_2': x, 'reading_data/is_training': is_training,
                       'reading_data/aux/Switch_2': aux},
            return_elements=return_elements,
            name=graph_name,
            op_dict=None,
            producer_op_list=None
        )
    return y #graph, y

def normal_moments(loc_samples, logvar_samples, prob_samples):
    #*_samples = (N,H,W,1,K) # K=MC Samples
    # E[X] = 1/K * \sum_{i=1}^K p(precip)_i * loc_i
    conditional_mean = np.mean(loc_samples*prob_samples, axis=-1) # (N,H,W,1)
    var_samples = np.exp(logvar_samples) # (N,H,W,1,K)
    var_samples *= prob_samples
    second_moment = var_samples + np.expand_dims(conditional_mean,-1)**2
    second_moment = second_moment.mean(axis=-1)
    conditional_sigma = (second_moment - conditional_mean**2)**0.5
    avg_probs = np.mean(prob_samples, -1)
    return conditional_mean, conditional_sigma, avg_probs

def gamma_moments(alpha_samples, beta_samples, prob_samples, eps=1e-6):
    prob = np.mean(prob_samples, axis=-1)
    loc_samples = prob_samples * alpha_samples / beta_samples
    first_moment = np.mean(loc_samples, axis=-1)
    second_moment = np.mean(prob_samples*alpha_samples / beta_samples**2, axis=-1)
    second_moment += np.mean(loc_samples**2, axis=-1)
    var = second_moment - first_moment ** 2
    #beta = first_moment / var
    #alpha = beta * first_moment
    return first_moment, (var**0.5), prob
    #return first_moment, (var**0.5)

def lognormal_moments(loc_samples, logvar_samples, prob_samples):
    var_samples = np.exp(logvar_samples)
    first_moment = (prob_samples * np.exp(loc_samples + var_samples / 2)).mean(axis=-1)
    second_moment = (prob_samples**2 * np.exp(2*loc_samples + 2*var_samples)).mean(axis=-1)
    var_estimate = second_moment - first_moment**2
    C = var_estimate / first_moment**2
    sigma_2 = np.log(0.5 * np.sqrt(4*C + 1) + 1)
    mu = np.log(first_moment) - sigma_2 / 2
    return first_moment, var_estimate**0.5, prob_samples.mean(axis=-1)
    return mu, sigma_2**0.5,prob_samples.mean(axis=-1)

def calibration(predicted_params, observations, distribution='normal',
               rainy_threshold=1.):
    """
    predicted_params: array of shape (N,H,W,2)
    observations: (N,H,W,1)
    """
    y = observations.flatten()
    rainy_days = np.where(y > rainy_threshold)[0]
    y_rainy = y[rainy_days]
    param1 = predicted_params[:,:,:,0].flatten()[rainy_days]
    param2 = predicted_params[:,:,:,1].flatten()[rainy_days]

    p_range = np.arange(0,1.,0.001)
    ratios  = []
    if distribution == 'normal':
        dist = scipy.stats.norm(param1, param2)
    elif distribution == 'gamma':
        beta = param1 / param2**2 # beta = E[X] / Var[x] 
        alpha = beta * param1   # alpha = beta*E[x]
        scale = beta
        # UNCLEAR: scale may equal 1/beta or beta 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
        dist = scipy.stats.gamma(param1, loc=0, scale=scale)
    elif distribution == 'lognormal':
        dist = scipy.stats.lognorm(param1, param2)

    cdf = dist.cdf(y_rainy)
    zs = []
    for p_interval in p_range:
        p = 0.5 + p_interval / 2.
        plow = 0.5 - p_interval / 2.
        phigh = 0.5 + p_interval / 2.

        ratios.append(np.nanmean((plow < cdf) & (phigh > cdf)))
    return p_range, np.array(ratios)

def main(frozen_graph, max_days=366,  output_node=None, year=1986, lr_km=8,
         upscale_factor=2, distribution='normal'):
    # read prism dataset
    ## resnet parameter will not re-interpolate X
    data_dir = FLAGS.data_dir 

    hr_km = int(lr_km/upscale_factor)
    dataset = PrismTFPipeline(data_dir, years=[year], lr_km=lr_km, hr_km=hr_km,
                              input_vars=['ppt'], output_vars=['ppt'])
    x, aux, y, lats, lons, t = dataset.get_patches(year)


    mask = (y[0,:,:,0]+1)/(y[0,:,:,0] + 1)
    aux_hr = aux[0,:,:,0] # all the elevations are the same, remove some data from memory

    #  resize x
    n, h, w, c = x.shape

    #now read in frozen graph, set placeholder for x, constant for elevs
    x_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    elev = tf.constant(aux_hr[np.newaxis,:,:,np.newaxis].astype(np.float32))

    # y1 = (loc or alpha), y2 = (logvar or beta)
    y1, y2, precip_prob = load_graph(frozen_graph, 'uq_srcnn', x=x_placeholder,
                                     aux=elev, distribution=distribution)
    #inverse_scale_precip = lambda x: x*100.
    if distribution == 'gamma':
        root = 3
        scale = 1./100
    elif distribution == 'normal':
        root = 1
        scale = 1./100
    elif distribution == 'lognormal': # the distribution handles transforming the data
        root = 1.
        scale = 1.

    scale_precip = lambda x: (x*scale) ** (1./root)
    inverse_scale_precip = lambda x: (x**root) / scale

    downscaled = []
    with tf.Session() as sess:
        rmses = []
        for i in range(0,min([x.shape[0], max_days])):
            _x = x[i,np.newaxis]
            # is_training=False removes padding at test time
            samples = [sess.run([y1, y2, precip_prob],feed_dict={x_placeholder: _x}) for _ in
                           range(FLAGS.mc_runs)]
            y1_samples = np.concatenate([np.expand_dims(s[0], -1) for s in samples], -1)
            y2_samples = np.concatenate([np.expand_dims(s[1], -1) for s in samples], -1)
            probs_samples = np.concatenate([np.expand_dims(s[2], -1) for s in samples], -1)
            if distribution.lower() == 'normal':
                loc, sigma, probs = normal_moments(y1_samples, y2_samples, probs_samples)
            elif distribution.lower() == 'gamma':
                loc, sigma, probs = gamma_moments(y1_samples, y2_samples, probs_samples)
            elif distribution.lower() == 'lognormal':
                loc, sigma, probs = lognormal_moments(y1_samples, y2_samples, probs_samples)

            prediction = inverse_scale_precip(loc)
            # loc, sigma are normalized
            predictive_parameters = np.concatenate([loc, sigma, probs], axis=-1)
            #predictive_parameters = inverse_scale_precip(predictive_parameters)
            downscaled.append(predictive_parameters)
            rmses.append(np.sqrt(np.nanmean((prediction[0,:,:,0] - y[i,:,:,0])**2)))

        print "RMSE", np.mean(rmses)

    downscaled = np.concatenate(downscaled, axis=0)
    downscaled *= mask[:,:,np.newaxis]
    uq_prange, uq_freq = calibration(downscaled, scale_precip(y[:downscaled.shape[0]]),
                                     distribution=distribution, rainy_threshold=scale_precip(0.1))
    print "Calbration RMSE:", np.nanmean((uq_prange - uq_freq)**2)**0.5
    da1 = xr.DataArray(downscaled[:,:,:,0], coords=[t[:len(downscaled)], lats[0], lons[0]],
                          dims=['time', 'lat', 'lon'])
    da2 = xr.DataArray(downscaled[:,:,:,1], coords=[t[:len(downscaled)], lats[0], lons[0]],
                          dims=['time', 'lat', 'lon'])
    da3 = xr.DataArray(downscaled[:,:,:,2], coords=[t[:len(downscaled)], lats[0], lons[0]],
                          dims=['time', 'lat', 'lon'])
    ds = xr.Dataset({'mu': da1, 'sigma': da2, 'probs': da3})
    '''
    fig, axs = plt.subplots(2,3)
    ymax = np.nanmax(y[i]) * 0.90
    axs = np.ravel(axs)
    axs[0].imshow(y[i,:,:,0], vmax=ymax)
    axs[0].axis('off')
    axs[0].set_title("Observed")
    axs[1].imshow(x[i,:,:,0], vmax=ymax)
    axs[1].axis('off')
    axs[1].set_title("Input")
    axs[2].imshow(inverse_scale_precip(downscaled[i,:,:,0]), vmax=ymax)
    axs[2].axis('off')
    axs[2].set_title("Downscaled Mean")
    axs[3].imshow(downscaled[i,:,:,1] * mask) #, vmax=ymax)
    axs[3].axis('off')
    axs[3].set_title("Downscaled Std")
    axs[4].plot(uq_prange, uq_freq)
    axs[4].plot([0,1],[0,1], ls='--')
    axs[4].set_title("Calibration")
    #plt.savefig('res.pdf')
    plt.show()
    #plt.close()
    '''
    return ds

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '%i' % (FLAGS.gpu)
    lr_km = 64
    hr_km = 16
    N = 1
    if FLAGS.years is None:
        years = range(2006,2016)
    else:
        years = [int(y) for y in FLAGS.years.split(",")]
    experiments = os.listdir(FLAGS.base_model_dir)
    for experiment in experiments:
        checkpoint = os.path.join(FLAGS.base_model_dir, experiment)
        joined_checkpoint = os.path.join(checkpoint, FLAGS.deepsd_model_name)
        if not os.path.exists(joined_checkpoint):
            os.mkdir(joined_checkpoint)
        distribution = experiment.split("_")[0]
        print '\n\ncheckpoint', checkpoint
        frozen_graph_file = freeze_graph(checkpoint, graph_name='srcnn', distribution=distribution)
        for y in years:
            ds = main(frozen_graph_file, max_days=366, year=y, distribution=distribution, lr_km=lr_km, upscale_factor=FLAGS.upscale)
            f = os.path.join(joined_checkpoint, 'precip_uq_%4i.nc' % y)
            ds.to_netcdf(f)
            tf.reset_default_graph()

