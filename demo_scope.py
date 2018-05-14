import tensorflow as tf

tf.set_random_seed(1)

with tf.name_scope('a_name_scope'):
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name)
    print(sess.run(var1))
    print(var2.name)
    print(sess.run(var2))
    print(var21.name)
    print(sess.run(var21))
    print(var22.name)
    print(sess.run(var22))

# 运行结果：
# var1:0
# [1.]
# a_name_scope/var2:0
# [2.]
# a_name_scope/var2_1:0
# [2.1]
# a_name_scope/var2_2:0
# [2.2]

with tf.variable_scope('a_variable_scope') as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
    var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var3.name)
    print(sess.run(var3))
    print(var4.name)
    print(sess.run(var4))
    print(var4_reuse.name)
    print(sess.run(var4_reuse))

# 运行结果
# a_variable_scope/var3:0
# [3.]
# a_variable_scope/var4:0
# [4.]
# a_variable_scope/var4_1:0
# [4.]

with tf.variable_scope('a_variable_scope') as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
    scope.reuse_variables()
    var3_reuse = tf.get_variable(name='var3')

with  tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var3.name)
    print(sess.run(var3))
    print(var3_reuse.name)
    print(sess.run(var3_reuse))