# -*- coding: utf-8 -*-
"""
Created on Wed May  9 16:06:47 2018

@author: wangj
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:13:03 2018

@author: wangj
"""

import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function = None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices = [1]), name='MSE')
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    writer =tf.summary.FileWriter('D:/Document/GitHub/tensorflow/logs/', sess.graph)
    sess.run(init)
    
#看图的话：
#1、cmd定位到logs文件夹内
#2、输入tensorboard --logdir ./
#3、打开浏览器 输入cmd提示的网址