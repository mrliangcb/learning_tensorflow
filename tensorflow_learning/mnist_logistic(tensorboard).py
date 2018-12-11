import tensorflow as tf

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



# Parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 64
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

#定义模型权重
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

batch_xs, batch_ys = mnist.train.next_batch(batch_size)#数据loader


#模型结构
#顺便放到scopes里面，制作图(tensorboard):with ft.name_scope
with tf.name_scope('Model'):
	pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax  x1*w1+x2*w2+……+b
with tf.name_scope('test_Model'):
	pred_test = tf.nn.softmax(tf.matmul(x, W) + b)

# Minimize error using cross entropy代价函数
with tf.name_scope('Loss'):
	cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
#

# Gradient Descent
with tf.name_scope('Adam'):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	
with tf.name_scope('Accuracy'):
	acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	acc= tf.reduce_mean(tf.cast(acc, tf.float32))
	#tf.cast(x,dtype,name=None)tf.bool

#定义一个总结summary监视cost的tensor变化
tf.summary.scalar("loss", cost)#前者名任意取，后者是tensor名
tf.summary.scalar("accuracy", acc)
#将所有的监视放进一个信号op
merged_summary_op = tf.summary.merge_all()
	
	
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
init2=tf.initialize_all_variables()
total_batch = int(mnist.train.num_examples/batch_size)

# Start training
with tf.Session() as sess:
	sess.run(init)
	sess.run(init2)
	
	#将监视op写入tensorboard
	summary_writer = tf.summary.FileWriter(r'./logs/', graph=tf.get_default_graph())
	
	coord = tf.train.Coordinator()  
	threads = tf.train.start_queue_runners(coord=coord) 
	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		
		# Loop over all batches
		for i in range(total_batch):
			
			# Fit training using batch data
			_, c,summary = sess.run([optimizer, cost,merged_summary_op],feed_dict={x: batch_xs,
											y: batch_ys})#直接输入batch_loader
			#把每次迭代写入
			summary_writer.add_summary(summary, epoch * total_batch + i)
			
			#run实例化一下测试集
			p=sess.run(pred_test,feed_dict={x: batch_xs,y: batch_ys})
			#print('输出:',p.shape)#输出是np

			# Compute average loss
			avg_cost += c / total_batch
		# Display logs per epoch step
		if (epoch+1) % display_step == 0:
			print('准确率',sess.run(acc,feed_dict={x: batch_xs,y: batch_ys}))
			print( "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

	print ("Optimization Finished!")

	# Test model
	correct_prediction = tf.equal(tf.argmax(pred_test, 1), tf.argmax(y, 1))
	# Calculate accuracy for 3000 examples
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#将[1,0,1]算平均值，这里就是(1+1)/3
	# print ("Accuracy:", accuracy.eval())
	print('测试准确率:',sess.run(accuracy,feed_dict={x: batch_xs,y: batch_ys}))
	
	coord.request_stop()  
	coord.join(threads)

#调用tensorboard
# tensorboard --logdir=./logs#直接读这个文件夹，不用读文件夹里面的文件
#Then open http://0.0.0.0:6006/ into your web browser
#http://localhost:6006/
#上面一栏,scalars看监视tensor，graphs看图结构



