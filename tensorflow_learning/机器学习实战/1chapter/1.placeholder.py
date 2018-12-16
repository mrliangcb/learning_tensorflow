import numpy as np
import tensorflow as tf



x = tf.placeholder(tf.float32, shape=(4, 4),name='input_place')
rand_array = np.random.rand(4, 4)
a=np.arange(0,16).reshape(4,4)
y = tf.identity(a)#placeholder 输入为nparray

sess = tf.Session()

print(sess.run(y, feed_dict={x: rand_array}))#

#将图保存
merged = tf.summary.merge_all()
path=r'C:/lcb\learning_python_git/learning_tensorflow/tensorflow_learning/机器学习实战/1chapter'
writer = tf.summary.FileWriter(path+"/1_tmp/variable_logs", sess.graph)