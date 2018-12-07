#保存参数

import tensorflow as tf
import numpy as np
train=False


if train:
	W = tf.Variable([[1,2,3],
					[3,2,5]],dtype=tf.float32,name='weights')
	b=tf.Variable([[5,8,3]],dtype=tf.float32,name='biases')
	print('权重',W)
	print('bias:',b)
	
	init=tf.initialize_all_variables()
	global_init=tf.global_variables_initializer()
	saver=tf.train.Saver()#会话之前定义
	with tf.Session() as sess:
		sess.run(global_init)#全局初始化
		sess.run(init)#局部初始化，这一步之后，W,B都初始化了
		print(sess.run(W))#建立W bariable的时候还没有值，run之后才初始化
		print(sess.run(b))
		save_path=saver.save(sess,r"C:\Users\mrliangcb\Desktop\note\module\tf\mnist\checkpoint\save.ckpt")
		print("save to path: ",save_path)#返回保存路径
	

else:
	#首先要创建一个跟前面形状一样的变量，然后再用之前的参数填充
	W_new = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')#这个名字一定要跟保存之前相同
	a=W_new
	
	saver=tf.train.Saver()#会话前定义saver
	with tf.Session() as sess:
		saver.restore(sess,r"C:\Users\mrliangcb\Desktop\note\module\tf\mnist\checkpoint\save.ckpt")#会话后提取
		print(sess.run(a))
	
#对于要保存整个tf的权重参数，先要定义tf框架，名字要相同
#print(sess.run(accuracy,feed_dict={}))
	
	