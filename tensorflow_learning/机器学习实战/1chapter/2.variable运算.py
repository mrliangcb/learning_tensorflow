#数据类型 np,tensor(tf.constant……),tf.variable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#就不会出现not compiled to use

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


v1 = tf.Variable(np.array(1,dtype=np.float64),name="v1")# dtype=tf.float32
v11 = tf.cast(v1,dtype=tf.float32)#tf转化类型

v2 = tf.Variable(np.array(2,dtype=np.float32), name="v2")
v3=v11+v2#两者要同样类型才能相加#模拟建立模型


global_init=tf.global_variables_initializer()

#单个变量相加（对于variable记得要run(init)）
with tf.Session() as sess:
	sess.run(global_init)
	print(sess.run(v1).dtype)
	print(sess.run(v11).dtype)
	print(sess.run(v3))

	def custom_polynomial(x_val):
		# Return 3x^2 - x + 10
		return(tf.subtract(3 * tf.square(x_val), x_val) + 10)#tf.wquare换成**2也行
	print(sess.run(custom_polynomial(11)))


#矩阵运算
print('########################################')
print('矩阵运算')

matrix1 = tf.constant(np.array([[3, 3]]),dtype=tf.float32)#默认ditype=32
matrix2 = tf.constant(np.array([[2],[2]],dtype=np.float32))
product = tf.matmul(matrix1, matrix2)
product2=matrix1*matrix2
with tf.Session() as sess:
	result = sess.run(product)
	result2=sess.run(product2)
	print('tf.mul运算',result)
	print('*运算',result2)






















