#

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.python import tensor_forest

# Ignore all GPUs, tf random forest does not benefit from it.
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=False)





num_steps = 50 # Total steps to train
batch_size = 1024 # The number of samples per batch
num_classes = 10 # The 10 digits
num_features = 784 # Each image is 28x28 pixels
num_trees = 10
max_nodes = 1000

total_batch = int(mnist.train.num_examples/batch_size)
batch_xs, batch_ys = mnist.train.next_batch(batch_size)#数据loader
X=batch_xs
Y=batch_ys

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
									num_features=num_features,
									num_trees=num_trees,
									max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
	resources.initialize_resources(resources.shared_resources()))

with tf.Session() as sess:
	
	coord = tf.train.Coordinator()  
	threads = tf.train.start_queue_runners(coord=coord) 
	# Training cycle
	sess.run(init_vars)#随机森林的初始化和别人的不同
	for i in range(1, num_steps + 1):
		# Prepare Data
		# Get the next batch of MNIST data (only images are needed, not labels)
		_, l = sess.run([train_op, loss_op])
		if i % 5 == 0 or i == 1:
			acc = sess.run(accuracy_op)
			print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))
	


#随机森林























