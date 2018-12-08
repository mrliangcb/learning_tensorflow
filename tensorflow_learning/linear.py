#简单线性回归
#训练过程有几种数据：1、placeholder接收numpy数据，float型。2,权重参数variable变量


import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
		7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
		2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# plt.figure(num='数据',figsize=(8,4))
# plt.scatter(train_X,train_Y,color='red')
# plt.legend()



# tf Graph Input
X = tf.placeholder("float")#float定义了数据类型
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

#定义图形
#模型op
pred = tf.add(tf.multiply(X, W), b)

# 代价函数op
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Adam
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

#训练会话
with tf.Session() as sess:
	sess.run(init)

	# Fit all training data
	for epoch in range(training_epochs):
		
		for (x, y) in zip(train_X, train_Y):
			#x,y分别是numpy float64
			
			#run optimizer相当于权重参数前进一步,torch.的step()
			sess.run(optimizer, feed_dict={X: x, Y: y})#float64可以喂给float32
			
		#Display logs per epoch step
		#显示cost值就先run一下
		if (epoch+1) % display_step == 0:
			c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
			print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
					"W=", sess.run(W), "b=", sess.run(b))

	print ("Optimization Finished!")
	training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
	print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
	
	#Graphic display
	plt.plot(train_X, train_Y, 'ro', label='Original data')
	plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
	plt.legend()
	plt.show()














