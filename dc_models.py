import tensorflow as tf
import bdl
import numpy as np
import tf_utils
import os, sys

def _maybe_pad_x(x, padding, is_training):
    if padding == 0:
       x_pad = x
    elif padding > 0:
        x_pad = tf.cond(is_training, lambda: x,
                        lambda: tf_utils.replicate_padding(x, padding))
    else:
        raise ValueError("Padding value %i should be greater than or equal to 1" % padding)
    return x_pad

class SDDCNetwork:
    def __init__(self, x, y, layer_sizes, filter_sizes, input_depth=1,
                 learning_rate=1e-4, N=2000, residual=True, residual_channels=[0],
                 device='/gpu:0', upscale_factor=2, output_depth=1, is_training=True,
                 distribution='normal', precip_threshold=0., tau=1e-5,
                 priorlengthscale=1e1):
        """
        Precipitation only!
        Args:
            layer_sizes: Sizes of each layer
            filter_sizes: List of sizes of convolutional filters
            input_depth: Number of channels in input
        """
        self.x = x
        self.y = y
        self.is_training = is_training
        self.upscale_factor = upscale_factor
        self.layer_sizes = layer_sizes
        self.layer_sizes += [3] # precip/no-precip (softmax), loc, and scale (regression)
        self.filter_sizes = filter_sizes
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.learning_rate = learning_rate
        self.residual = residual
        self.residual_channels = residual_channels
        self.device = device
        self.global_step = tf.Variable(1, trainable=False)
        self.N = N
        self.eps = 1e-6
        #self.learning_rate = learning_rate
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                                   100000, 0.95)
        self.distribution = distribution
        self.precip_threshold = precip_threshold
        self.tau = tau
        self.priorlengthscale = priorlengthscale
        self._build_graph()

    def _normalize(self):
        # normalize precip with a cube root
        # normalize other features with batch_norm
        # normalize y with cube root
        print "Distribution", self.distribution
        if self.distribution == 'gamma':
            root = 3.
            scale = 1./100
            self.scale_precip = lambda x: (x*scale) ** (1./root)
            self.inverse_scale_precip = lambda x: (x**root) / scale
        elif self.distribution == 'normal':
            root = 1.
            scale = 1./100
            self.scale_precip = lambda x: (x*scale) ** (1./root)
            self.inverse_scale_precip = lambda x: (x**root) / scale
        elif self.distribution == 'lognormal':
            self.scale_precip = lambda x: tf.log(x + 1e-5)
            self.inverse_scale_precip = lambda x: tf.exp(x) - 1e-5

        with tf.variable_scope("normalize_inputs") as scope:
            precip_cuberoot = tf.expand_dims(self.scale_precip(self.x[:,:,:,0]), -1)
            with tf.device('/cpu:0'):
                tf.summary.histogram('precip_cuberoot', precip_cuberoot)
                tf.summary.histogram('precip', self.x[:,:,:,0])
            other_x = tf.contrib.layers.batch_norm(tf.expand_dims(self.x[:,:,:,1], -1),
                                       trainable=False,
                                       epsilon=1e-6,
                                       updates_collections=None, center=False,
                                       scale=False, is_training=self.is_training)
            self.x_norm = tf.concat([precip_cuberoot, other_x], axis=3, name='x_norm')

    def _inference(self, X, eps=1e-6):
        shape = tf.to_float(tf.shape(X))
        # classification of precipitiation
        wd = self.priorlengthscale / (self.tau * self.N)
        dd = 2. / (self.tau * self.N)

        self.p = []
        h = X
        pad_amt = int((sum(self.filter_sizes)-len(self.filter_sizes))/2)
        X_reduced = tf.cond(self.is_training, lambda: h[:,pad_amt:-pad_amt,pad_amt:-pad_amt,:],
                            lambda: h)

        for i, k in enumerate(self.filter_sizes):
            with tf.variable_scope("hidden_%i" % i) as scope:
                if i == (len(self.filter_sizes)-1):
                    activation = None
                else:
                    activation = tf.nn.relu
                if i != 0:
                    with tf.variable_scope('concrete_layer') as inner_scope:
                        input_dim = k**2 * self.layer_sizes[i-1] #tf.to_float(tf.shape(h)[-1])
                        h = bdl.concrete_layer(h, is_training=self.is_training,
                                       reg_scale=dd*input_dim,
                                       init_prob=0.10)
                        inner_scope.reuse_variables()
                        self.p.append(tf.sigmoid(tf.get_variable("ConcreteDropout/dropout_logit")))
                        #self.p.append(tf.constant(0.2))
                        #h = tf.nn.dropout(h, 1-self.p[-1])
                else:
                    self.p.append(tf.constant(0.0))

                pad_amt = int((k-1)/2)
                h = _maybe_pad_x(h, pad_amt, self.is_training)
                w_regularizer = tf.contrib.layers.l2_regularizer(wd)
                h = tf.layers.conv2d(h, self.layer_sizes[i], k, activation=activation,
                                    kernel_regularizer=w_regularizer,
                                    bias_regularizer=w_regularizer)
                weights = []

        # shape(h) = (N, H, W, 3)
        with tf.variable_scope("output_layer") as scope:
            shape = tf.shape(h)
            self.logits = tf.expand_dims(h[:,:,:,0], -1, name='logits')
            self.probs = tf.sigmoid(self.logits, name='precip_probs')
            X_resid = tf.gather(X_reduced, self.residual_channels, axis=3)
            if self.distribution.lower() == 'gamma':
                # Mode = (alpha-1)/beta -> alpha = e^h[:,:,:,1] + beta * X_resid + 1
                # we want to learn the log so alpha > 0 and beta > 0
                #self.beta = tf.expand_dims(tf.exp(h[:,:,:,1]), -1, name='beta') + self.eps
                r = 10.
                self.beta = tf.expand_dims(tf.exp(h[:,:,:,1]), -1, name='beta') + self.eps
                self.alpha = tf.expand_dims(tf.exp(h[:,:,:,2]), -1) + self.eps
                #if self.residual:
                    #self.alpha += self.beta * X_resid + 1
                #    self.alpha += X_resid
                self.alpha = tf.identity(self.alpha, name='alpha')
                self.h = tf.tuple([X, h] + self.p)
                # using the mode as the prediction
                #self.loc = tf.nn.relu((1-self.alpha) / self.beta)
                self.loc = tf.divide(self.alpha+eps, self.beta+eps, name='loc')
                self.var = tf.divide(self.alpha+eps, self.beta**2, name='var')
            elif self.distribution.lower() in ['normal', 'lognormal']:
                # we could add X_resid here
                self.loc = tf.expand_dims(h[:,:,:,1], -1)
                if self.residual:
                    self.loc += X_resid
                self.loc = tf.identity(self.loc, name='loc')
                self.logvar = tf.expand_dims(h[:,:,:,2], -1, name='logvar')
                self.var = tf.exp(self.logvar, name='var')
                #self.loc += self.var / 2.

            with tf.device("/cpu:0"):
                tf.summary.image('precip_prob', self.logits, max_outputs=1)
                if hasattr(self, 'var'):
                    tf.summary.histogram('precip_var', self.var)
                if hasattr(self, 'alpha'):
                    tf.summary.histogram('alpha', self.alpha)
                    tf.summary.histogram('beta', self.beta)
                tf.summary.image('precip_amt', self.loc, max_outputs=1)
                tf.summary.histogram('precip_amt', self.loc)

            #y = tf.to_float(tf.greater(self.probs, 0.5)) * self.loc
            # return the expected value
            if self.distribution.lower() == 'lognormal':
                # HIGHBIAS and ERROR ON TENSORBOARD: probs*(loc + var/2)
                y = (self.loc + self.var / 2) # this is rescaled with inverse_scale_precip = exp(x) 
            else:
                y = self.loc
            y = self.probs * tf.identity(self.inverse_scale_precip(y), 'prediction')

        return y

    def _loss(self, eps=1e-4):
        """
            if self.distribution == 'normal':
                ll = 0
                for k in range(K):
                    dist = tf.contrib.distributions.Normal(loc=self.loc[:,k],
                                                           scale=self.scale[:,k])
                    ll += self.weights[:,k] * dist.prob(yflat)

                self.neglogloss = -tf.reduce_mean(tf.log(ll + eps))
        """
        with tf.name_scope("loss"):
            reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # if training then crop center of y, else, padding was applied
            slice_amt = (np.sum(self.filter_sizes) - len(self.filter_sizes)) / 2
            if slice_amt > 0:
                slice_y = self.y[:,slice_amt:-slice_amt, slice_amt:-slice_amt]
            else:
                slice_y = self.y

            # start by slicing if training
            _y = tf.cond(self.is_training, lambda: slice_y, lambda: self.y)
            # get classification labels
            classes = tf.to_float(tf.greater(_y, self.precip_threshold))
            #  gather normalized precipitation for learning
            indicies = tf.where(tf.equal(classes, 1))
            _y_cond = tf.gather_nd(_y, indicies)
            _y_norm_precip = self.scale_precip(_y_cond)

            if self.distribution.lower() == 'gamma':
                _alpha_cond = tf.gather_nd(self.alpha, indicies)
                _beta_cond = tf.gather_nd(self.beta, indicies)
                # log likelihood = gamma log likelihood + cross entropy of classifier
                dist = tf.distributions.Gamma(concentration=_alpha_cond, rate=_beta_cond)
                cond_logprob = dist.log_prob(_y_norm_precip + eps)
            elif self.distribution.lower() in ['normal', 'lognormal']:
                _loc_cond = tf.gather_nd(self.loc, indicies)
                _logvar_cond = tf.gather_nd(self.logvar, indicies)
                _precision_cond = tf.exp(-_logvar_cond) + eps
                with tf.device('/cpu:0'):
                    tf.summary.histogram('precision_cond', _precision_cond)
                    tf.summary.histogram('logvar_cond', _logvar_cond)
                #dist = tf.distributions.Normal(loc=_loc_cond, scale=_var_cond**0.5)
                #K.sum(precision * (true - mean)**2. + log_var, -1)
                cond_logprob = _precision_cond * (_y_norm_precip - _loc_cond)**2 + _logvar_cond
                cond_logprob *= -1
                #self.pc = tf.group(self.logvar, _logvar_cond)

            # if no pixels contain precip, then cond_logprob is empty and reduce_mean fails
            cond_logprob = tf.cond(tf.equal(tf.size(cond_logprob), 0),
                                   lambda: tf.constant(0.0), lambda: cond_logprob)
            # we don't need to learn logits if all classes are 1
            #if self.precip_threshold < 0:
            #    logprob_classifier = tf.constant(0.0)
            #else:
            logprob_classifier = classes * tf.log(self.probs+eps)
            logprob_classifier += (1 - classes) * tf.log(1-self.probs+eps)
            logprob = tf.reduce_mean(cond_logprob) + tf.reduce_mean(logprob_classifier)
            #self.pc = self.h #tf.group(_alpha_cond, _beta_cond, cond_logprob)

            self.neglogloss = -tf.reduce_mean(logprob, name='neglogloss')
            with tf.device("/cpu:0"):
                #tf.summary.scalar('loss/auc', tf.metrics.auc(classes, self.probs)[0])
                tf.summary.scalar('loss/class_mean', tf.reduce_mean(classes))
                tf.summary.scalar('loss/logit_loss', -tf.reduce_mean(logprob_classifier))
                tf.summary.scalar('loss/cond_logprob', -tf.reduce_mean(cond_logprob))
                tf.summary.scalar('loss/neglogloss', self.neglogloss)
                tf.summary.scalar('loss/regularizer', reg_losses)
                tf.summary.scalar('loss/logprobs', -tf.reduce_mean(tf.log(self.probs+eps)))
                tf.summary.scalar('loss/log1minusprobs', -tf.reduce_mean(tf.log(1-self.probs+eps)))

        with tf.name_scope("prediction"):
            slice_y = tf.cond(tf.equal(slice_amt, 0), lambda: self.y,
                              lambda: self.y[:,slice_amt:-slice_amt, slice_amt:-slice_amt])
            _y = tf.cond(self.is_training, lambda: slice_y, lambda: self.y)
            self.rmse = tf.sqrt(tf_utils.nanmean(tf.square(self.prediction - _y)),
                            name='rmse')
            self.bias = tf_utils.nanmean(self.prediction - _y)
            return self.neglogloss + reg_losses

    def _optimize(self):
        opt = tf.train.AdamOptimizer(self.learning_rate)
        #opt = tf.train.AdagradOptimizer(self.learning_rate)
        self.opt = opt.minimize(self.loss, global_step=self.global_step)

    def _summaries(self):
        with tf.device('/cpu:0'):
            tf.contrib.layers.summarize_tensors(tf.trainable_variables())
            tf.summary.scalar('rmse(mm)', self.rmse)
            tf.summary.scalar('bias(mm)', self.bias)

    def _build_graph(self):
        self._normalize()
        with tf.device(self.device):
            self.prediction = self._inference(self.x_norm)
            self.loss = self._loss()
            self._optimize()

        self._summaries()

class GaussianNetwork:
    def __init__(self, x, y, layer_sizes, filter_sizes, input_depth=1,
                 learning_rate=1e-4, N=2000, residual=True, residual_channels=[0],
                 device='/gpu:0', upscale_factor=2, output_depth=1, is_training=True,
                 distribution='normal', tau=1e-5, priorlengthscale=1e1):
        """
        Precipitation only!
        Args:
            layer_sizes: Sizes of each layer
            filter_sizes: List of sizes of convolutional filters
            input_depth: Number of channels in input
        """
        self.x = x
        self.y = y
        self.is_training = is_training
        self.upscale_factor = upscale_factor
        self.layer_sizes = layer_sizes
        self.layer_sizes += [2] #  loc and scale (regression)
        self.filter_sizes = filter_sizes
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.learning_rate = learning_rate
        self.residual = residual
        self.residual_channels = residual_channels
        self.device = device
        self.global_step = tf.Variable(0, trainable=False)
        self.N = N
        self.eps = 1e-6
        self.learning_rate = learning_rate
        self.distribution = distribution
        self.tau = tau
        self.priorlengthscale = priorlengthscale
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                                   100000, 0.95)
        self._build_graph()

    def _normalize(self):
        # normalize precip with a cube root
        # normalize other features with batch_norm
        # normalize y with cube root
        root = 1.
        scale = 1./100
        self.scale_precip = lambda x: (x*scale) ** (1./root)
        self.inverse_scale_precip = lambda x: (x**root) / scale
        with tf.variable_scope("normalize_inputs") as scope:
            precip_scaled = tf.expand_dims(self.scale_precip(self.x[:,:,:,0]), -1)
            with tf.device('/cpu:0'):
                tf.summary.histogram('precip_scaled', precip_scaled)
                tf.summary.histogram('precip', self.x[:,:,:,0])
            other_x = tf.contrib.layers.batch_norm(tf.expand_dims(self.x[:,:,:,1], -1),
                                       trainable=False,
                                       epsilon=1e-6,
                                       updates_collections=None, center=False,
                                       scale=False, is_training=self.is_training)
            self.x_norm = tf.concat([precip_scaled, other_x], axis=3, name='x_norm')

    def _inference(self, X, eps=1e-6):
        shape = tf.to_float(tf.shape(X))
        # classification of precipitiation
        wd = self.priorlengthscale / (self.tau * self.N)
        dd = 2. / (self.tau * self.N)

        self.p = []
        h = X
        pad_amt = int((sum(self.filter_sizes)-len(self.filter_sizes))/2)
        X_reduced = tf.cond(self.is_training, lambda: h[:,pad_amt:-pad_amt,pad_amt:-pad_amt,:],
                            lambda: h)

        for i, k in enumerate(self.filter_sizes):
            with tf.variable_scope("hidden_%i" % i) as scope:
                if i == (len(self.filter_sizes)-1):
                    activation = None
                else:
                    activation = tf.nn.relu
                if i != 0:
                    with tf.variable_scope('concrete_layer') as inner_scope:
                        input_dim = k**2 * self.layer_sizes[i-1] #tf.to_float(tf.shape(h)[-1])
                        h = bdl.concrete_layer(h, is_training=self.is_training,
                                       reg_scale=dd*input_dim,
                                       init_prob=0.10)
                        inner_scope.reuse_variables()
                        self.p.append(tf.sigmoid(tf.get_variable("ConcreteDropout/dropout_logit")))
                        #self.p.append(tf.constant(0.2))
                        #h = tf.nn.dropout(h, 1-self.p[-1])
                else:
                    self.p.append(tf.constant(0.0))

                pad_amt = int((k-1)/2)
                h = _maybe_pad_x(h, pad_amt, self.is_training)
                w_regularizer = tf.contrib.layers.l2_regularizer(wd)
                h = tf.layers.conv2d(h, self.layer_sizes[i], k, activation=activation,
                                    kernel_regularizer=w_regularizer,
                                    bias_regularizer=w_regularizer)
                weights = []

        # shape(h) = (N, H, W, 2)
        with tf.variable_scope("output_layer") as scope:
            shape = tf.shape(h)
            X_resid = tf.gather(X_reduced, self.residual_channels, axis=3)

            # we could add X_resid here
            self.loc = tf.expand_dims(h[:,:,:,0], -1)
            if self.residual:
                self.loc += X_resid
            self.loc = tf.identity(self.loc, name='loc')
            self.logvar = tf.expand_dims(h[:,:,:,1], -1, name='logvar')
            self.var = tf.exp(self.logvar, name='var')

            with tf.device("/cpu:0"):
                if hasattr(self, 'var'):
                    tf.summary.histogram('precip_var', self.var)
                if hasattr(self, 'alpha'):
                    tf.summary.histogram('alpha', self.alpha)
                    tf.summary.histogram('beta', self.beta)
                tf.summary.image('precip_amt', self.loc, max_outputs=1)
                tf.summary.histogram('precip_amt', self.loc)

            #y = tf.to_float(tf.greater(self.probs, 0.5)) * self.loc
            # return the expected value
            y = tf.identity(self.inverse_scale_precip(self.loc), 'prediction')

        return y

    def _loss(self, eps=1e-4):
        """
            if self.distribution == 'normal':
                ll = 0
                for k in range(K):
                    dist = tf.contrib.distributions.Normal(loc=self.loc[:,k],
                                                           scale=self.scale[:,k])
                    ll += self.weights[:,k] * dist.prob(yflat)

                self.neglogloss = -tf.reduce_mean(tf.log(ll + eps))
        """
        with tf.name_scope("loss"):
            reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # if training then crop center of y, else, padding was applied
            slice_amt = (np.sum(self.filter_sizes) - len(self.filter_sizes)) / 2
            if slice_amt > 0:
                slice_y = self.y[:,slice_amt:-slice_amt, slice_amt:-slice_amt]
            else:
                slice_y = self.y

            # start by slicing if training
            _y = tf.cond(self.is_training, lambda: slice_y, lambda: self.y)
            #  gather normalized precipitation for learning
            _y_norm_precip = self.scale_precip(_y)
            _precision= tf.exp(-self.logvar) + eps

            with tf.device('/cpu:0'):
                tf.summary.histogram('logvar', self.logvar)

            logprob = _precision * (_y_norm_precip - self.loc)**2 + self.logvar
            self.neglogloss = tf.reduce_mean(logprob, name='neglogloss')
            with tf.device("/cpu:0"):
                #tf.summary.scalar('loss/auc', tf.metrics.auc(classes, self.probs)[0])
                tf.summary.scalar('loss/neglogloss', self.neglogloss)
                tf.summary.scalar('loss/regularizer', reg_losses)

        with tf.name_scope("prediction"):
            slice_y = tf.cond(tf.equal(slice_amt, 0), lambda: self.y,
                              lambda: self.y[:,slice_amt:-slice_amt, slice_amt:-slice_amt])
            _y = tf.cond(self.is_training, lambda: slice_y, lambda: self.y)
            self.rmse = tf.sqrt(tf_utils.nanmean(tf.square(self.prediction - _y)),
                            name='rmse')
            self.bias = tf_utils.nanmean(self.prediction - _y)
            return self.neglogloss + reg_losses

    def _optimize(self):
        opt = tf.train.AdamOptimizer(self.learning_rate)
        #opt = tf.train.AdagradOptimizer(self.learning_rate)
        self.opt = opt.minimize(self.loss, global_step=self.global_step)

    def _summaries(self):
        with tf.device('/cpu:0'):
            tf.contrib.layers.summarize_tensors(tf.trainable_variables())
            tf.summary.scalar('rmse(mm)', self.rmse)
            tf.summary.scalar('bias(mm)', self.bias)

    def _build_graph(self):
        self._normalize()
        with tf.device(self.device):
            self.prediction = self._inference(self.x_norm)
            self.loss = self._loss()
            self._optimize()

        self._summaries()
