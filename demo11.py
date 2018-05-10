import tensorflow as tf

## Save to file
# remember to define the same dtype and shape when restore
# W = tf.Variable([[1,2,3],[4,5,6]], dtype=tf.float32, name='weights')
# b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')
#
# init = tf.initialize_all_variables()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
# 	sess.run(init)
# 	save_path = saver.save(sess, "my_net/save_net.ckpt")
# 	print("Save to path:", save_path)

# restore variables
# redefine the same shape and same type for your variables
W = tf.Variable(tf.zeros([2, 3]), dtype=tf.float32, name="weights")
b = tf.Variable(tf.zeros([1, 3]), dtype=tf.float32, name="biases")

# do not init step
saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, "my_net/save_net.ckpt")
	print("weights:", sess.run(W))
	print("biases:", sess.run(b))