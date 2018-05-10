import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 0 to 9 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def  compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs:v_xs, ys})