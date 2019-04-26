#数据类型 np,tensor(tf.constant……),tf.variable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#就不会出现not compiled to use

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


############ 建立变量variable的各种方法 ############
#tf.Variable(tf.zeros([1,20]))  #tf.ones([])  
#tf.Variable(tf.zeros_like(其他variable)   #ones_like() #建立同样大小的矩阵
# fill_var = tf.Variable(tf.fill([3, 3], -1)) #自定义初始，global_init对这个不起作用
# const_var = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9])) #自定义初始
# const_fill_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim])) #自定义初始
# tf.Variable里可以写 tf.linspace(start=0.0, stop=1.0, num=3)  tf.range(start=6, limit=15, delta=3)
# rnorm_var = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0) #定义初始参数的mean和std

# tf.get_variable("c", dtype=tf.float32, shape=value.shape,initializer=tf.constant_initializer(value))  这个很常用  初始化为常量  value可以放np.array()
#   tf.truncated_normal_initializer  #随机初始化方法
#   xavier_initializer 也是随机初始化


#方法一：直接建立模型公式
v1 = tf.Variable(np.array(1,dtype=np.float64),name="v1")# dtype=tf.float32
v11 = tf.cast(v1,dtype=tf.float32)#tf转化数据类型
v2 = tf.Variable(np.array(2,dtype=np.float32), name="v2")
v3=v11+v2#两者要同样类型才能相加#模拟建立模型

#方法二：利用函数建立模型公式
def custom_polynomial(x_val):
		# Return 3x^2 - x + 10
		return(tf.subtract(3 * tf.square(x_val), x_val) + 10)#tf.square换成**2也行
dy = tf.gradients(custom_polynomial(v1),v1)
#ddy = tf.gradients(dy,x)  二次导  

global_init=tf.global_variables_initializer()

#单个变量相加（对于variable记得要run(init)）
with tf.Session() as sess:
	# sess.run(ones_similar.initializer) #给单个var进行初始
	# sess.run(tf.initialize_variables([variable_c])) #给单个var初始  
	#  sess.run(tf.initialize_all_variables())  初始化全部
	
	sess.run(global_init) #给全部var初始
	print(sess.run(v1).dtype)# 查看变量的性质，.dtype(数据类型)  .size(有多少个参数) .shape(几行几列)  type()
	print(sess.run(v11).dtype) #建立variable的时候，还没初始和运行的，当run的时候才做
	print(sess.run(v3))
	print('计算函数模型',sess.run(custom_polynomial(v1)))#可以输入variable，也可以输入常量数字
	print(sess.run(dy)) #求导 (3x^2-x+10)'=(6x-1)=5

	
	
##########  建立矩阵     ##########
#  tf.diag([1.0,1.0,1.0])  
#  
	
	
#矩阵运算
print('########################################')
print('矩阵运算')

matrix1 = tf.constant(np.array([[3, 3]]),dtype=tf.float32)#默认ditype=32  constant是tensor ，还不是variable
matrix2 = tf.constant(np.array([[2],[2]],dtype=np.float32))
product = tf.matmul(matrix1, matrix2) #横乘以竖
product2=matrix1*matrix2  #不知道是什么

print(product) 
print('还没run的variable',type(product))  #定义为tf.con  tensor型
print('还没run的v3:var',type(v3)) #定义为tf.var(tf.con)   tensor型

const_fill_var = tf.Variable(tf.constant(-1, shape=[3, 3]))

with tf.Session() as sess:
	result = sess.run(product)
	result2=sess.run(product2)
	print('tf.mul运算',result)
	print('*运算',result2)
	print(sess.run(matrix1).shape)
	print('run了var的结果',type(result))  #np型  因为var初始就是np型的  var只是一个载体，管道
	
#tf.constant    tf.variable      tf.placeholder()的区别

# tf.Variable()生成变量 变量才可导
# tf.constant()生成常量 常量不可导
# 变量需要初始化：


















