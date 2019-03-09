# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt

from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Reshape, Flatten, Activation
from keras.layers import Input, Reshape
from keras.models import Model
# from keras import initializations
# from keras import initializers
from keras import backend as K

from tensorflow.examples.tutorials.mnist import input_data

# ----------------------------------------------------------------------------

default_opt = { 'lr' : 5e-5, 'c' : 1e-2, 'n_critic' : 5 }

class WDCGAN(object):
  """Wasserstein Deep Convolutional Generative Adversarial Network"""

  def __init__(self, n_dim, n_chan=1, opt_alg='rmsprop', opt_params=default_opt):
    self.n_critic = opt_params['n_critic']
    self.c        = opt_params['c']
    n_lat = 100
    self.PutG_shape = (16, 64)

    # create session
    self.sess = tf.Session()
    K.set_session(self.sess) # pass keras the session

    # create generator
    with tf.name_scope('generator'):
      # Xk_g = Input(shape=(n_lat,))
      Xk_g = Input(shape=self.PutG_shape)
      g = make_dcgan_generator(Xk_g, n_lat, n_chan)

    # create discriminator
    with tf.name_scope('discriminator'):
      Xk_d = Input(shape=(n_chan, n_dim, n_dim))
      d = make_dcgan_discriminator(Xk_d)

    # instantiate networks
    g_net = Model(input=Xk_g, output=g)
    g_net.summary()
    d_net = Model(input=Xk_d, output=d)
    d_net.summary()

    # save inputs
    # X_g = tf.placeholder(tf.float32, shape=(None, n_lat), name='X_g')
    X_g = tf.placeholder(tf.float32, shape=(None, 16, n_dim), name='X_g')
    X_d = tf.placeholder(tf.float32, shape=(None, n_chan, n_dim, n_dim), name='X_d')
    self.inputs = X_g, X_d

    # get the weights
    self.w_g = [w for w in tf.global_variables() if 'generator' in w.name]
    self.w_d = [w for w in tf.global_variables() if 'discriminator' in w.name]

    # create predictions
    d_real = d_net(X_d)
    d_fake = d_net(g_net(X_g))
    self.P = g_net(X_g)

    # create losses
    self.loss_g = tf.reduce_mean(d_fake)
    self.loss_d = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)

    # compute and store discriminator probabilities
    self.d_real = tf.reduce_mean(d_real)
    self.d_fake = tf.reduce_mean(d_fake)
    self.p_real = tf.reduce_mean(tf.sigmoid(d_real))
    self.p_fake = tf.reduce_mean(tf.sigmoid(d_fake))

    # create an optimizer
    lr = opt_params['lr']
    optimizer_g = tf.train.RMSPropOptimizer(lr)
    optimizer_d = tf.train.RMSPropOptimizer(lr)

    # get gradients
    gv_g = optimizer_g.compute_gradients(self.loss_g, self.w_g)
    gv_d = optimizer_d.compute_gradients(self.loss_d, self.w_d)

    # create training operation
    self.train_op_g = optimizer_g.apply_gradients(gv_g)
    self.train_op_d = optimizer_d.apply_gradients(gv_d)

    # clip the weights, so that they fall in [-c, c]
    self.clip_updates = [w.assign(tf.clip_by_value(w, -self.c, self.c)) for w in self.w_d]

  def fit(self, X_train, y_train, X_val, y_val, n_epoch=10, n_batch=64, logdir='dcgan-run'):

    # initialize log directory                  
    if tf.gfile.Exists(logdir):
      tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)

    # mnist = input_data.read_data_sets('data/mnist')

    # init model
    init = tf.global_variables_initializer()
    self.sess.run(init)

    # train the model
    step, g_step, d_step, epoch = 0, 0, 0, 0
    for epoch in range(n_epoch):
      for step in range(X_train.shape[0] // n_batch):
        # idx = np.random.randint(0, X_train.shape[0], n_batch)
        # X_train_batch = X_train[idx]
        # idx = np.random.randint(0, X_train_batch[0], n_batch)
        # X_val_batch = X_val[idx]
        # y_val_batch = y_val[idx]

        # n_critic = 100 if g_step < 25 or (g_step + 1) % 500 == 0 else self.n_critic
        n_critic = 5
        start_time = time.time()
        for i in range(n_critic):
          losses_d = []
          # load the batch
          # X_batch = mnist.train.next_batch(n_batch)[0]
          # X_batch = X_batch.reshape((n_batch, 1, 28, 28))
          X_batch = X_train[d_step*64 : (d_step+1)*64]

          noise = np.random.rand(16, 64).astype('float32')
          for j in range(1,n_batch):
            b = np.random.rand(16, 64).astype('float32')
            noise = np.hstack((noise, b))
          # print(noise.shape)
          # noise = np.expand_dims(noise, axis=0)
          noise = noise.reshape((n_batch, 16, 64))

          feed_dict = self.load_batch(X_batch, noise)
          # train the critic/discriminator
          loss_d = self.train_d(feed_dict)
          losses_d.append(loss_d)
          d_step+=1

        loss_d = np.array(losses_d).mean()

        # train the generator
        noise = np.random.rand(16, 64).astype('float32')
        for k in range(1,n_batch):
          b = np.random.rand(16, 64).astype('float32')
          noise = np.hstack((noise, b))
        # print(noise.shape)
        noise = noise.reshape((n_batch, 16, 64))
        # noise = np.expand_dims(noise, axis=0)
        #tensor expending [none, 1, 16, 64]

        # noise = np.random.uniform(-1.0, 1.0, [n_batch, 100]).astype('float32')
        feed_dict = self.load_batch(X_batch, noise)
        loss_g = self.train_g(feed_dict)
        g_step += 1

        if g_step < 100 or g_step % 100 == 0:
          tot_time = time.time() - start_time
          print('Epoch: %3d, Gen step: %4d (%3.1f s), Disc loss: %.6f, Gen loss %.6f' % \
                (epoch, g_step, tot_time, loss_d, loss_g))


    # while mnist.train.epochs_completed < n_epoch:
    #   n_critic = 100 if g_step < 25 or (g_step+1) % 500 == 0 else self.n_critic
    #   start_time = time.time()
    #   for i in range(n_critic):
    #     losses_d = []
    #
    #     # load the batch
    #
    #     X_batch = mnist.train.next_batch(n_batch)[0]
    #     X_batch = X_batch.reshape((n_batch, 1, 28, 28))
    #     noise = np.random.rand(n_batch,100).astype('float32')
    #     feed_dict = self.load_batch(X_batch, noise)
    #
    #     # train the critic/discriminator
    #     loss_d = self.train_d(feed_dict)
    #     losses_d.append(loss_d)
    #
    #   loss_d = np.array(losses_d).mean()
    #
    #   #train the generator
    #   noise = np.random.rand(n_batch,100).astype('float32')#n_batch*100
    #   # noise = np.random.uniform(-1.0, 1.0, [n_batch, 100]).astype('float32')
    #   feed_dict = self.load_batch(X_batch, noise)
    #   loss_g = self.train_g(feed_dict)
    #   g_step += 1
    #
    #   if g_step < 100 or g_step % 100 == 0:
    #     tot_time = time.time() - start_time
    #     print ('Epoch: %3d, Gen step: %4d (%3.1f s), Disc loss: %.6f, Gen loss %.6f' % \
    #       (mnist.train.epochs_completed, g_step, tot_time, loss_d, loss_g))
    #
    #   # take samples
    #   if g_step % 100 == 0:
    #     noise = np.random.rand(n_batch,100).astype('float32')
    #     samples = self.gen(noise)
    #     samples = samples[:42]
    #     fname = logdir + '.mnist_samples-%d.png' % g_step
    #     plt.imsave(fname,
    #                (samples.reshape(6, 7, 28, 28)
    #                        .transpose(0, 2, 1, 3)
    #                        .reshape(6*28, 7*28)),
    #                cmap='gray')
    #   saver.save(self.sess, checkpoint_root, global_step=step)

  def gen(self, noise):
    X_g_in, X_d_in = self.inputs
    feed_dict = { X_g_in : noise, K.learning_phase() : False }
    return self.sess.run(self.P, feed_dict=feed_dict)

  def train_g(self, feed_dict):
    _, loss_g = self.sess.run([self.train_op_g, self.loss_g], feed_dict=feed_dict)
    return loss_g

  def train_d(self, feed_dict):
    # clip the weights, so that they fall in [-c, c]
    self.sess.run(self.clip_updates, feed_dict=feed_dict)

    # take a step of RMSProp
    self.sess.run(self.train_op_d, feed_dict=feed_dict)

    # return discriminator loss
    return self.sess.run(self.loss_d, feed_dict=feed_dict)

  def train(self, feed_dict):
    self.sess.run(self.train_op, feed_dict=feed_dict)

  def load_batch(self, X_train, noise, train=True):
    X_g_in, X_d_in = self.inputs
    return {X_g_in : noise, X_d_in : X_train, K.learning_phase() : train}

  def eval_err(self, X, n_batch=128):
    batch_iterator = iterate_minibatches(X, n_batch, shuffle=True)
    loss_g, loss_d, p_real, p_fake = 0, 0, 0, 0
    tot_loss_g, tot_loss_d, tot_p_real, tot_p_fake = 0, 0, 0, 0
    for bn, batch in enumerate(batch_iterator):
      noise = np.random.rand(n_batch,100)
      feed_dict = self.load_batch(batch, noise)
      loss_g, loss_d, p_real, p_fake \
        = self.sess.run([self.d_real, self.d_fake, self.p_real, self.p_fake], 
                        feed_dict=feed_dict)
      tot_loss_g += loss_g
      tot_loss_d += loss_d
      tot_p_real += p_real
      tot_p_fake += p_fake
    return tot_loss_g / (bn+1), tot_loss_d / (bn+1), \
           tot_p_real / (bn+1), tot_p_fake / (bn+1)

    
# ----------------------------------------------------------------------------

def conv2D_init(shape, dim_ordering='tf', name=None, dtype = None):
   # return initializers.normal(shape, scale=0.02, dim_ordering=dim_ordering, name=name)
   return K.random_normal(shape, dtype=dtype)

def make_dcgan_discriminator(Xk_d):
  x = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init,
        dim_ordering='th')(Xk_d)
  x = LeakyReLU(0.2)(x)

  x = Convolution2D(nb_filter=128, nb_row=4, nb_col=4, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init,
        dim_ordering='th')(x)
  # x = BatchNormalization(mode=2, axis=1)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(0.2)(x)

  x = Flatten()(x)
  x = Dense(1024, init=conv2D_init)(x)
  # x = BatchNormalization(mode=2)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(0.2)(x)

  d = Dense(1, activation=None)(x)

  # substitute_detector = Model(inputs=Xk_d, outputs=d, name='substitute_detector')
  # substitute_detector.summary()

  return d

def make_dcgan_generator(Xk_g, n_lat, n_chan=1):
  n_g_hid1 = 1024 # size of hidden layer in generator layer 1
  n_g_hid2 = 128  # size of hidden layer in generator layer 2


  # x = Dense(n_g_hid1, init=conv2D_init)(Xk_g)
  Xk_g = Reshape((-1, 16, 64))(Xk_g)
  x = Dense(n_g_hid2, init=conv2D_init)(Xk_g)
  # x = BatchNormalization(mode=2, )(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  # x = Dense(n_g_hid2*7*7, init=conv2D_init)(x)
  x = Dense(16, init=conv2D_init)(x)
  # x = Reshape((n_g_hid2, 7, 7))(x)
  # x = BatchNormalization(mode=2, axis=1)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  # x = Deconvolution2D(64, 5, 5, output_shape=(128, 64, 14, 14),
  #       border_mode='same', activation=None, subsample=(2,2),
  #       init=conv2D_init, dim_ordering='th')(x)
  x = Deconvolution2D(64, 5, 5, output_shape=(64, 64, 32, 32),
        border_mode='same', activation=None, subsample=(2,2),
        init=conv2D_init, dim_ordering='th')(x)
  # x = BatchNormalization(mode=2, axis=1)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  # g = Deconvolution2D(n_chan, 5, 5, output_shape=(128, n_chan, 28, 28),
  #       border_mode='same', activation='sigmoid', subsample=(2,2),
  #       init=conv2D_init, dim_ordering='th')(x)

  g = Deconvolution2D(n_chan, 5, 5, output_shape=(64, n_chan, 64, 64),
        border_mode='same', activation='sigmoid', subsample=(2,2),
        init=conv2D_init, dim_ordering='th')(x)

  # generator = Model(inputs=Xk_g, outputs=g, name='generator')
  # generator.summary()

  return g