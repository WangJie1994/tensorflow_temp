# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:05:31 2018

@author: wangj
"""

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7], input2:[5]}))
